"""
CarSim 实时仿真环境 (视觉增强版)

环境返回多模态观测:
  obs = {
      "physics": np.ndarray (physics_dim,) float32, 归一化后的车辆动力学状态,
      "image":   np.ndarray (3, img_h, img_w) uint8, ROI裁剪后的道路图像 (CHW, RGB),
  }

图像处理管线:
  CarSim共享缓冲区 → RGB→BGR → 垂直翻转 → ROI裁剪(40%~90%高度) → Resize → BGR→RGB → CHW → uint8
"""
from ctypes import cdll, cast, POINTER, c_ubyte, c_ulong
import os
import numpy as np
import cv2
import time
from typing import Tuple, Dict

import GetSharedBufferInfo
import Simulation_with_LiveAnimation
from GetSharedBufferInfo import VSSBCT_RGB


class LiveCarsimEnv:
    """
    基于 CarSim 共享缓冲区的实时仿真环境。
    集成车辆动力学状态和道路图像两种观测，供多模态 RL 智能体使用。
    """

    IMG_HEIGHT = 48
    IMG_WIDTH = 64

    def __init__(
        self,
        simfile_path: str = "simfile.sim",
        vs_dll_path: str = r"E:\CarSim2024.1_Prog\Database\RL_VISION\RL_TC_VISION\vs_lv_ds_x64.dll",
        sim_time_s: float = 10.0,
        max_torque: float = 1000.0,
        target_slip_ratio: float = 0.06,
        reward_weights: dict = None,
        frame_skip: int = 5,
    ):
        self.simfile_path = simfile_path
        self.vs_dll_path = vs_dll_path
        self.sim_time_s = sim_time_s
        self.max_torque = max_torque
        self.target_slip_ratio = target_slip_ratio
        self.frame_skip = frame_skip

        default_weights = {
            'w_speed': 0.1, 'w_accel': 0.0, 'w_energy': 0.0,
            'w_consistency': 0.0, 'w_beta': 0.0, 'w_slip': -1.0,
            'w_smooth': 0.0, 'w_yaw': -50.0
        }
        self.weights = default_weights.copy()
        if reward_weights:
            self.weights.update(reward_weights)

        self.wheel_radius = 0.325
        self.veh_bf = 1.675
        self.veh_br = 1.675
        self.veh_l = 2.910
        self.veh_lf = 1.015
        self.veh_lr = self.veh_l - self.veh_lf

        self.vs = None
        self.configuration = None
        self.t_current = 0.0
        self.t_step = 0.0
        self.status = 0
        self.current_step = 0
        self.max_steps = 0
        self.last_torque = np.zeros(4)
        self.import_array = [0.0, 0.0, 0.0, 0.0]
        self.export_array = []
        self.episode_count = 0
        self.restart_interval = 100

        self.physics_dim = 8
        self.action_dim = 4
        self.img_shape = (3, self.IMG_HEIGHT, self.IMG_WIDTH)

        # 平滑度评估窗口大小 (用于计算多步扭矩变化)
        self.smooth_window_size = 10
        self.torque_history = []

        self._init_vs()

    # ================================================================
    # VS 接口
    # ================================================================
    def _init_vs(self):
        self.vs = Simulation_with_LiveAnimation.VehicleSimulationWithLiveAnimation()
        if not os.path.exists(self.vs_dll_path):
            raise FileNotFoundError(f"DLL not found: {self.vs_dll_path}")
        vs_dll = cdll.LoadLibrary(self.vs_dll_path)
        if vs_dll is None or not self.vs.get_api(vs_dll):
            raise RuntimeError("Failed to load VS DLL API")

    # ================================================================
    # reset / step
    # ================================================================
    def reset(self) -> Tuple[Dict, Dict]:
        """重置环境, 返回 (obs, info)"""
        self.episode_count += 1

        if self.episode_count > 0 and self.episode_count % self.restart_interval == 0:
            print(f"[Env] 定期重新初始化VS接口 (Episode {self.episode_count})")
            if self.configuration:
                try:
                    self.vs.terminate(self.t_current)
                    time.sleep(2)
                except Exception:
                    pass
            self._init_vs()
            self.configuration = None

        if self.configuration:
            self.vs.terminate(self.t_current)
            time.sleep(3)

        print(os.linesep + "++++++++++++++++ Starting New Episode ++++++++++++++++")

        self.configuration = self.vs.ReadConfiguration(self.simfile_path)
        self.t_current = self.configuration.get('t_start')
        self.t_step = self.configuration.get('t_step')
        n_export = self.configuration.get('n_export')
        print(f"[Env] CarSim Config: t_step={self.t_step}, t_start={self.t_current}")

        self.max_steps = int(self.sim_time_s / self.t_step)
        self.current_step = 0
        self.status = 0
        self.last_torque = np.zeros(4)
        self.torque_history = []  # 重置平滑度历史缓冲区

        self.import_array = [0.0, 0.0, 0.0, 0.0]
        self.export_array = [0.0] * n_export

        self.status, self.export_array = self.vs.updateonestep(
            self.t_current, self.import_array, self.export_array)
        time.sleep(2)

        physics = self._parse_observation(self.export_array)
        norm_physics = self._normalize_state(physics)
        image = self._grab_image(display=True, info=None, reward=0.0, step=0)

        obs = {"physics": norm_physics, "image": image}
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """执行一步, 返回 (obs, reward, done, info)"""
        action = np.clip(action, 0.0, 1.0)
        target_torque = action * self.max_torque

        self.import_array = [target_torque[i] for i in range(4)]

        #控制步减少
        for _ in range(self.frame_skip):
            self.t_current = (self.configuration.get('t_start')
                              + (self.current_step + 1) * self.t_step)
            self.status, self.export_array = self.vs.updateonestep(
                self.t_current, self.import_array, self.export_array)
            self.current_step += 1
            if self.status != 0:
                break

        physics = self._parse_observation(self.export_array)
        norm_physics = self._normalize_state(physics)

        # 更新扭矩历史缓冲区 (用于平滑度评估)
        self.torque_history.append(target_torque.copy())
        if len(self.torque_history) > self.smooth_window_size:
            self.torque_history.pop(0)

        reward, r_details = self._calculate_reward(
            physics, target_torque, self.last_torque)
        self.last_torque = target_torque

        if self.status != 0:
            print(f"[Env] Simulation stopped at step {self.current_step}, "
                  f"time {self.t_current:.4f}, status {self.status}")
        done = (self.current_step >= self.max_steps) or (self.status != 0)

        info = {
            "vx": physics[0], "ax": physics[1],
            "yaw": physics[6], "beta": physics[7],
            "slip_L1": physics[2], "slip_R1": physics[3],
            "slip_L2": physics[4], "slip_R2": physics[5],
            "trq_L1": target_torque[0], "trq_R1": target_torque[1],
            "trq_L2": target_torque[2], "trq_R2": target_torque[3],
            **r_details
        }

        display = (self.current_step % 10 == 0)
        image = self._grab_image(display=display, info=info,
                                 reward=reward, step=self.current_step)

        obs = {"physics": norm_physics, "image": image}
        return obs, reward, done, info

    # ================================================================
    # 图像获取与处理
    # ================================================================
    def _grab_image(self, display=False, info=None, reward=0.0, step=0) -> np.ndarray:
        """
        从 CarSim 共享缓冲区获取图像, 裁剪 ROI, 返回 (3, H, W) uint8 RGB。
        获取失败时返回全零图像。
        """
        image = np.zeros(self.img_shape, dtype=np.uint8)

        if not GetSharedBufferInfo.sbHandle:
            return image

        GetSharedBufferInfo.api_ver2_contents.Lock(GetSharedBufferInfo.sbHandle, 100)
        try:
            pageindex = c_ulong(0)
            pixel_ptr = GetSharedBufferInfo.api_ver2_contents.GetData(
                GetSharedBufferInfo.sbHandle, VSSBCT_RGB, pageindex)

            if pixel_ptr:
                width = GetSharedBufferInfo.width
                height = GetSharedBufferInfo.height

                img_data = np.ctypeslib.as_array(
                    cast(pixel_ptr, POINTER(c_ubyte)),
                    shape=(height, width, 3))
                frame = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
                frame = cv2.flip(frame, 0)

                h, w = frame.shape[:2]
                
                roi_x_start = int(w * 0.2)
                roi_x_end = int(w * 0.8)
                roi_y_start = int(h * 0.65)
                roi_y_end = int(h * 0.85)
                roi_frame = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

                roi_resized = cv2.resize(roi_frame,
                                         (self.IMG_WIDTH, self.IMG_HEIGHT))

                roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
                image = np.transpose(roi_rgb, (2, 0, 1)).copy()

                if display and info:
                    self._draw_hud(frame, roi_x_start, roi_x_end, roi_y_start, roi_y_end,
                                   info, reward, step)
        except Exception as e:
            print(f"[Env] Vision error: {e}")
        finally:
            GetSharedBufferInfo.api_ver2_contents.UnLock(GetSharedBufferInfo.sbHandle)

        return image

    def _draw_hud(self, frame, roi_x_start, roi_x_end, roi_y_start, roi_y_end,
                  info, reward, step):
        """在原始帧上绘制 HUD 信息并显示"""
        cv2.rectangle(frame, (roi_x_start, roi_y_start), (roi_x_end - 1, roi_y_end - 1),
                      (0, 255, 0), 2)
        cv2.putText(frame, "ROI", (5, roi_y_start + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        y = 35
        lines = [
            f"Time: {self.t_current:.2f}s | Step: {step} | Reward: {reward:.4f}",
            f"Vx: {info['vx']:.1f} km/h | Yaw: {info['yaw']:.3f} rad/s | Beta: {info['beta']:.2f} deg",
            f"Slip: {info['slip_L1']:.4f}/{info['slip_R1']:.4f} (F) | "
            f"{info['slip_L2']:.4f}/{info['slip_R2']:.4f} (R)",
            f"Torque: {info['trq_L1']:.0f}/{info['trq_R1']:.0f} (F) | "
            f"{info['trq_L2']:.0f}/{info['trq_R2']:.0f} (R)",
        ]
        for line in lines:
            cv2.putText(frame, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y += 35

        cv2.imshow('CarSim Live Training View', frame)
        cv2.waitKey(1)

    # ================================================================
    # 状态解析与归一化
    # ================================================================
    def _parse_observation(self, export_array) -> np.ndarray:
        if len(export_array) < 10:
            return np.zeros(8, dtype=np.float32)

        vx = export_array[0]
        vy = export_array[1]
        ax = export_array[2]
        avz = export_array[3] * np.pi / 180
        steer = export_array[4] * np.pi / 180 * 17.49

        vx_ms = vx / 3.6

        rpm_to_ms = (2 * np.pi / 60.0) * self.wheel_radius
        v_L1 = export_array[5] * rpm_to_ms * 3.6
        v_R1 = export_array[6] * rpm_to_ms * 3.6
        v_L2 = export_array[7] * rpm_to_ms * 3.6
        v_R2 = export_array[8] * rpm_to_ms * 3.6
        beta = export_array[9]

        v_L1_c = ((vx_ms - avz * 0.5 * self.veh_bf) * np.cos(steer) +
                   (vy + avz * self.veh_lf) * np.sin(steer)) * 3.6
        v_R1_c = ((vx_ms + avz * 0.5 * self.veh_bf) * np.cos(steer) +
                   (vy + avz * self.veh_lf) * np.sin(steer)) * 3.6
        v_L2_c = (vx_ms - avz * 0.5 * self.veh_br) * np.cos(steer) * 3.6
        v_R2_c = (vx_ms + avz * 0.5 * self.veh_br) * np.cos(steer) * 3.6

        def calc_slip(v_w, v_c):
            return (v_w - v_c) / max(abs(v_w), abs(v_c), 0.1) if max(abs(v_w), abs(v_c)) > 3.0 else 0.0

        s_L1 = calc_slip(v_L1, v_L1_c)
        s_R1 = calc_slip(v_R1, v_R1_c)
        s_L2 = calc_slip(v_L2, v_L2_c)
        s_R2 = calc_slip(v_R2, v_R2_c)

        return np.array([vx, ax, s_L1, s_R1, s_L2, s_R2, avz, beta],
                        dtype=np.float32)

    def _normalize_state(self, raw) -> np.ndarray:
        n = raw.copy()
        n[0] = raw[0] / 100.0
        n[1] = raw[1] / 1.0
        n[2:6] = raw[2:6] / 1.0
        n[6] = raw[6] / 1.5
        n[7] = raw[7] / 1.0
        return n

    # ================================================================
    # 奖励函数
    # ================================================================
    def _calculate_reward(self, state, current_torque, last_torque):
        vx = state[0]
        ax = state[1]
        slips = state[2:6]
        yaw_rate = state[6]
        beta = state[7]
        w = self.weights

        r_speed = w['w_speed'] * vx
        r_accel = w['w_accel'] * ax

        r_slip = 0.0
        if vx > 3.0:
            for s in slips:
                r_slip += (max(0.0, s - self.target_slip_ratio))**2
        r_slip = w['w_slip'] * r_slip *2

        r_energy = w['w_energy'] * np.mean(np.abs(current_torque / self.max_torque))
        
        r_consistency = w['w_consistency'] * (
            abs(current_torque[0] - current_torque[1]) +
            abs(current_torque[2] - current_torque[3])) / self.max_torque

        # 平滑度奖励：评估窗口内的扭矩变化方差
        if len(self.torque_history) >= 3:
            # 计算窗口内扭矩变化的方差 (衡量平滑度)
            torque_array = np.array(self.torque_history)  # shape: (N, 4)
            # 计算每列(每个轮子)的方差，然后求平均
            torque_variance = np.mean(np.var(torque_array, axis=0))
            # 归一化到 [0, 1] 范围 (假设合理最大方差为 0.25 * max_torque^2)
            normalized_variance = torque_variance / (0.25 * self.max_torque ** 2)
            # 限制范围并计算惩罚 (方差越大越不平滑)
            r_smooth = w['w_smooth'] * min(normalized_variance, 1.0)
        else:
            # 历史不足时退化为单步变化率
            r_smooth = w['w_smooth'] * np.mean(((current_torque - last_torque) / self.max_torque) ** 2)

        r_yaw = 0.0
        if abs(yaw_rate) > 0.01:
            r_yaw = w['w_yaw'] * (abs(yaw_rate) - 0.01)

        r_beta = w['w_beta'] * abs(beta)

        total = (r_speed + r_accel + r_slip + r_energy +
                 r_consistency + r_smooth + r_yaw + r_beta)
        total = np.clip(total, -10.0, 10.0)

        details = {
            "R_Spd": r_speed, "R_Acc": r_accel, "R_Slp": r_slip,
            "R_Eng": r_energy, "R_Cns": r_consistency,
            "R_Yaw": r_yaw, "R_Beta": r_beta, "R_Smooth": r_smooth
        }
        return total, details

    # ================================================================
    # 辅助方法
    # ================================================================
    def close(self):
        if self.vs and self.configuration:
            self.vs.terminate(self.t_current)
        cv2.destroyAllWindows()

    def get_physics_dim(self):
        return self.physics_dim

    def get_action_dim(self):
        return self.action_dim

    def get_img_shape(self):
        return self.img_shape

    # ================================================================
    # 环境配置测试主函数
    # ================================================================
if __name__ == "__main__":
    env = LiveCarsimEnv()
    try:
        obs, _ = env.reset()
        print(f"Physics dim: {obs['physics'].shape}")
        print(f"Image shape: {obs['image'].shape}, dtype: {obs['image'].dtype}")
        step_count = 0
        while True:
            t = env.t_current
            torque_val = min(t, 0.8)
            action = np.array([torque_val, torque_val,
                               torque_val * 0.70, torque_val * 0.70])
            obs, reward, done, info = env.step(action)
            if step_count % 100 == 0:
                print(f"Step {step_count}, Time {t:.2f}s, Vx {info['vx']:.2f}")
            step_count += 1
            if done:
                break
    finally:
        env.close()
