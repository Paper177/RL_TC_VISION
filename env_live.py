import argparse
from ctypes import cdll
import os
import struct
import platform
import numpy as np
import cv2
import math
import sys
import time
from typing import Tuple, Dict, Optional

import GetSharedBufferInfo
import Simulation_with_LiveAnimation
from GetSharedBufferInfo import VSSBCT_RGB, VSSBCT_DEPTH
from ctypes import cast, POINTER, c_ubyte, c_ulong

class LiveCarsimEnv:
    """
    基于 TorqueControl.py 的实时 CarSim 环境
    集成了实时图像显示和 DDPG 训练接口
    """
    
    # ================= CarSim 变量名配置 (参考 env_pc.py) =================
    # 这里的变量名主要用于文档说明，实际交互通过 import_array/export_array 顺序进行
    # 假设 CarSim 配置的 Import 顺序为: [Torque_L1, Torque_R1, Torque_L2, Torque_R2]
    # 假设 CarSim 配置的 Export 顺序为: [Vx, Vy,Ax,AVz, , Steer_SW,AVy_L1, AVy_R1, AVy_L2, AVy_R2, Beta]
    
    def __init__(
        self,
        simfile_path: str = "simfile.sim",
        vs_dll_path: str = r"E:\CarSim2024.1_Prog\Database\RL_VISION\RL_TC_VISION\vs_lv_ds_x64.dll",
        sim_time_s: float = 10.0,
        max_torque: float = 1000.0,
        target_slip_ratio: float = 0.06,
        reward_weights: dict = None,
        frame_skip: int = 1  # 新增：跳帧参数，默认为1（不跳帧）
    ):
        self.simfile_path = simfile_path
        self.vs_dll_path = vs_dll_path
        self.sim_time_s = sim_time_s
        self.max_torque = max_torque
        self.target_slip_ratio = target_slip_ratio
        self.frame_skip = frame_skip # 保存跳帧设置
        
        # 初始化奖励权重
        default_weights = {
            'w_speed': 0.1, 'w_accel': 0.0, 'w_energy': 0.0,
            'w_consistency': 0.0, 'w_beta': 0.0, 'w_slip': -1.0, 'w_smooth': 0.0,
            'w_yaw': -50.0
        }
        self.weights = default_weights.copy()
        if reward_weights:
            self.weights.update(reward_weights)
            
        # 车辆物理参数 (参考 env_pc.py)
        self.wheel_radius = 0.325 
        self.veh_bf = 1.675
        self.veh_br = 1.675
        self.veh_l = 2.910
        self.veh_lf = 1.015
        self.veh_lr = self.veh_l - self.veh_lf

        # 内部状态
        self.vs = None
        self.configuration = None
        self.t_current = 0.0
        self.status = 0
        self.current_step = 0
        self.max_steps = 0
        self.last_torque = np.zeros(4)
        self.import_array = [0.0, 0.0, 0.0, 0.0]
        self.export_array = []
        self.episode_count = 0
        self.restart_interval = 100
        
        # 状态空间维度: [Vx, Ax, S_L1, S_R1, S_L2, S_R2, YawRate, Beta]
        self.state_dim = 8
        # 动作空间维度: [T_L1, T_R1, T_L2, T_R2]
        self.action_dim = 4
        
        # 初始化 VS 接口
        self._init_vs()

    def _init_vs(self):
        """初始化 VehicleSimulationWithLiveAnimation 接口"""
        self.vs = Simulation_with_LiveAnimation.VehicleSimulationWithLiveAnimation()
        
        if not os.path.exists(self.vs_dll_path):
            raise FileNotFoundError(f"DLL not found: {self.vs_dll_path}")
            
        vs_dll = cdll.LoadLibrary(self.vs_dll_path)
        if vs_dll is None or not self.vs.get_api(vs_dll):
            raise RuntimeError("Failed to load VS DLL API")

    def reset(self) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        self.episode_count += 1
        
        # 定期重新初始化VS接口，防止资源累积导致的问题
        if self.episode_count > 0 and self.episode_count % self.restart_interval == 0:
            print(f"[Env] 定期重新初始化VS接口 (Episode {self.episode_count})")
            if self.configuration:
                try:
                    self.vs.terminate(self.t_current)
                    time.sleep(2)
                except:
                    pass
            self._init_vs()
            self.configuration = None
        
        # 如果之前有运行，先终止
        if self.configuration:
            self.vs.terminate(self.t_current)
            time.sleep(3) # 等待终止完成
            
        print(os.linesep + "++++++++++++++++ Starting New Episode ++++++++++++++++")
        
        # 读取配置                            
        # 使用原始路径，避免重复replace导致路径损坏
        self.configuration = self.vs.ReadConfiguration(self.simfile_path)
        self.t_current = self.configuration.get('t_start')
        self.t_step = self.configuration.get('t_step')
        print(f"[Env] CarSim Config Loaded: t_step={self.t_step}, t_start={self.t_current}")
        n_export = self.configuration.get('n_export')

        self.max_steps = int(self.sim_time_s / self.t_step)
        self.current_step = 0
        self.status = 0
        self.last_torque = np.zeros(4)
        
        # 初始化数组
        self.import_array = [0.0, 0.0, 0.0, 0.0]
        self.export_array = [0.0] * n_export
        
        # 运行一步以获取初始状态
        self.status, self.export_array = self.vs.updateonestep(self.t_current, self.import_array, self.export_array)
        time.sleep(2) # 等待初始步计算和渲染更新
        
        # 解析状态
        state = self._parse_observation(self.export_array)
        norm_state = self._normalize_state(state)
        
        # 构造初始 info 以便显示
        info = {
            "vx": state[0],
            "ax": state[1],
            "yaw": state[6],
            "beta": state[7],
            "slip_L1": state[2],
            "slip_R1": state[3],
            "slip_L2": state[4],
            "slip_R2": state[5],
            "trq_L1": 0.0,
            "trq_R1": 0.0,
            "trq_L2": 0.0,
            "trq_R2": 0.0
        }
        
        # 立即更新一次图像，显示初始状态
        self._update_live_view(np.zeros(4), info, 0.0, 0)
        
        return norm_state, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行一步"""
        # 1. 动作处理
        action = np.clip(action, 0.0, 1.0)
        target_torque = action * self.max_torque
        
        # 更新输入数组 [L1, R1, L2, R2]
        self.import_array = [target_torque[0], target_torque[1], target_torque[2], target_torque[3]]
        
        # 2. 物理仿真步进 (Frame Skip)
        # 循环执行 frame_skip 次 CarSim 仿真步，但只在最后一步返回状态
        # 这样可以保持 CarSim 内部高精度 (0.001s)，同时降低 RL 控制频率 (如 0.005s)
        
        for _ in range(self.frame_skip):
            # 使用整数步数计算时间，避免浮点数累积误差
            self.t_current = self.configuration.get('t_start') + (self.current_step + 1) * self.t_step
            self.status, self.export_array = self.vs.updateonestep(self.t_current, self.import_array, self.export_array)
            
            # 步数增加
            self.current_step += 1
            
            # 如果仿真出错，立即停止
            if self.status != 0:
                break
                
        # 4. 状态解析 (提前解析以便显示)
        raw_state = self._parse_observation(self.export_array)
        next_state = self._normalize_state(raw_state)
        
        # 5. 计算奖励
        reward, r_details = self._calculate_reward(raw_state, target_torque, self.last_torque)
        
        self.last_torque = target_torque
        # self.current_step += 1 # 已经在循环中增加
        
        # 6. 判断结束
        # 增加 Yaw Rate 约束：如果 |YawRate| > 0.5 rad/s，则终止并给予惩罚
        '''
        yaw_rate = raw_state[6]
        if abs(yaw_rate) > 0.2:
            done = True
            reward -= 20000.0 # 给予极大惩罚
            r_details["R_Yaw"] = -20000.0
            print(f"Episode terminated due to excessive Yaw Rate: {yaw_rate:.3f} rad/s")
        else:
            done = (self.current_step >= self.max_steps) or (self.status != 0)
        '''

        if self.status != 0:
            print(f"[Env] Simulation stopped early at step {self.current_step}, time {self.t_current:.4f}, status {self.status}")
        done = (self.current_step >= self.max_steps) or (self.status != 0)

        # 7. Info
        info = {
            "vx": raw_state[0],
            "ax": raw_state[1],
            "yaw": raw_state[6],
            "beta": raw_state[7], # Beta (deg)
            "slip_L1": raw_state[2],
            "slip_R1": raw_state[3],
            "slip_L2": raw_state[4],
            "slip_R2": raw_state[5],
            "trq_L1": target_torque[0],
            "trq_R1": target_torque[1],
            "trq_L2": target_torque[2],
            "trq_R2": target_torque[3],
            **r_details
        }

        # 3. 实时图像显示 (每 10 步更新一次，避免过于卡顿)
        # 将 info 传入以便显示更多信息
        if self.current_step % 10 == 0:
            self._update_live_view(target_torque, info, reward, self.current_step)
        
        return next_state, reward, done, info

    def _update_live_view(self, torque, info, reward, step):
        """更新实时图像显示"""
        if GetSharedBufferInfo.sbHandle:
            GetSharedBufferInfo.api_ver2_contents.Lock(GetSharedBufferInfo.sbHandle, 100)
            try:
                pageindex = c_ulong(0)
                pixeldatatemp = GetSharedBufferInfo.api_ver2_contents.GetData(GetSharedBufferInfo.sbHandle, VSSBCT_RGB, pageindex)
                
                if pixeldatatemp:
                    width = GetSharedBufferInfo.width
                    height = GetSharedBufferInfo.height
                    
                    # 获取图像数据
                    img_data = np.ctypeslib.as_array(cast(pixeldatatemp, POINTER(c_ubyte)), shape=(height, width, 3))
                    
                    # 转换颜色空间 RGB -> BGR
                    frame = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
                    
                    # 垂直翻转
                    frame = cv2.flip(frame, 0)
                    
                    # 显示信息
                    # 第一行：时间、步数、奖励
                    cv2.putText(frame, f"Time: {self.t_current:.2f}s | Step: {step} | Reward: {reward:.4f}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # 第二行：速度、Yaw、Beta
                    cv2.putText(frame, f"Vx: {info['vx']:.1f} km/h | Yaw: {info['yaw']:.3f} rad/s | Beta: {info['beta']:.2f} deg", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # 第三行：滑移率
                    slip_str = f"Slip: {info['slip_L1']:.4f}/{info['slip_R1']:.4f} (F) | {info['slip_L2']:.4f}/{info['slip_R2']:.4f} (R)"
                    cv2.putText(frame, slip_str, (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # 第四行：扭矩
                    trq_str = f"Torque: {torque[0]:.0f}/{torque[1]:.0f} (F) | {torque[2]:.0f}/{torque[3]:.0f} (R)"
                    cv2.putText(frame, trq_str, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    cv2.imshow('CarSim Live Training View', frame)
                    cv2.waitKey(1)
            except Exception as e:
                print(f"Live view error: {e}")
            finally:
                GetSharedBufferInfo.api_ver2_contents.UnLock(GetSharedBufferInfo.sbHandle)

    def _parse_observation(self, export_array) -> np.ndarray:
        """
        解析 CarSim 输出数组
        假设 Export 变量顺序: 
        0: Vx (km/h)
        1: Vy (km/h)
        2: Ax (g)
        3: AVz (deg/s)
        4: Steer_SW (deg)
        5: AVy_L1 (rpm)
        6: AVy_R1 (rpm)
        7: AVy_L2 (rpm)
        8: AVy_R2 (rpm)
        9: Beta (deg)
        ...
        注意：这个顺序必须与 CarSim 中的 Generic Export 设置一致！
        如果顺序不同，需要调整索引。
        """
        # 安全检查
        if len(export_array) < 10:
            # 如果输出变量不足，返回全0状态防止崩溃
            return np.zeros(7, dtype=np.float32)

        vx = export_array[0]
        vy = export_array[1]
        ax = export_array[2]
        avz = export_array[3] * np.pi / 180
        steer = export_array[4] * np.pi / 180 * 17.49 # 假设转向比
        
        vx_ms = vx / 3.6
        
        # 轮速 RPM -> m/s
        rpm_to_ms = (2 * np.pi / 60.0) * self.wheel_radius
        v_L1 = export_array[5] * rpm_to_ms * 3.6
        v_R1 = export_array[6] * rpm_to_ms * 3.6
        v_L2 = export_array[7] * rpm_to_ms * 3.6
        v_R2 = export_array[8] * rpm_to_ms * 3.6
        beta = export_array[9] # Beta (deg)
        
        # 计算轮心速度 (同 env_pc.py)
        v_L1_c = ((vx_ms - avz*0.5*self.veh_bf)*np.cos(steer)  + (vy+avz*self.veh_lf)*np.sin(steer)) * 3.6
        v_R1_c = ((vx_ms + avz*0.5*self.veh_bf)*np.cos(steer)  + (vy+avz*self.veh_lf)*np.sin(steer)) * 3.6
        v_L2_c = ((vx_ms - avz*0.5*self.veh_br)*np.cos(steer) ) * 3.6 
        v_R2_c = ((vx_ms + avz*0.5*self.veh_br)*np.cos(steer) ) * 3.6
        # 计算滑移率
        def calc_slip(v_w, v_c):
            return (v_w - v_c) / max(abs(v_w), abs(v_c), 0.1) if max(abs(v_w), abs(v_c)) > 3.0 else 0.0
            
        s_L1 = calc_slip(v_L1, v_L1_c)
        s_R1 = calc_slip(v_R1, v_R1_c)
        s_L2 = calc_slip(v_L2, v_L2_c)
        s_R2 = calc_slip(v_R2, v_R2_c)
        return np.array([vx, ax, s_L1, s_R1, s_L2, s_R2, avz, beta], dtype=np.float32)

    def _normalize_state(self, raw_state):
        n_s = raw_state.copy()
        n_s[0] = raw_state[0] / 100.0
        n_s[1] = raw_state[1] / 1.0
        n_s[2:6] = raw_state[2:6] / 1.0
        n_s[6] = raw_state[6] / 1.5
        # Beta 归一化 (假设范围 -10 到 10 度)excessive Yaw Rate
        n_s[7] = raw_state[7] / 1.0 
        return n_s

    def _calculate_reward(self, state, current_torque, last_torque):
        vx = state[0]
        ax = state[1]
        slips = state[2:6]
        yaw_rate = state[6]
        beta = state[7]
        w = self.weights

        '''
        r1 = 0.0
        for j in range(0,4):
            if slips[j] > self.target_slip_ratio:
                r1=r1 + w['w_slip'] * (slips[j] - self.target_slip_ratio) - w["w_energy"] * np.abs(current_torque[j]/self.max_torque)
            else:
                r1=r1 + w["w_energy"] * np.abs(current_torque[j]/self.max_torque)
        r2= w["w_beta"] * abs(beta)+ w["w_yaw"] * (abs(yaw_rate) - 0.1)
        '''
        
        r_speed = w['w_speed'] * vx
        r_accel = w['w_accel'] * ax  

        r_slip = 0.0
        if vx > 3.0:
            for i in range(4):
                r_slip += max(0.0, slips[i] - self.target_slip_ratio)
        r_slip = w['w_slip'] * r_slip  

        r_energy = w['w_energy'] * np.mean(np.abs(current_torque/self.max_torque))
        r_consistency = w['w_consistency'] * (abs(current_torque[0] - current_torque[1])+abs(current_torque[2] - current_torque[3]))/self.max_torque
        r_smooth = w['w_smooth'] * np.mean(((current_torque - last_torque)/self.max_torque)**2)
        
        # Yaw Rate Penalty
        r_yaw = 0.0
        if abs(yaw_rate) > 0.01:
            r_yaw = w['w_yaw'] * (abs(yaw_rate) - 0.01)

        r_beta = w['w_beta'] * abs(beta)

        total = r_speed + r_accel + r_slip + r_energy + r_consistency + r_smooth + r_yaw + r_beta
        # 限制奖励范围，防止Q值估计过大
        total = np.clip(total, -10.0, 10.0)
        details = {"R_Spd": r_speed, "R_Acc": r_accel, "R_Slp": r_slip, "R_Eng": r_energy, "R_Cns": r_consistency, "R_Yaw": r_yaw, "R_Beta": r_beta, "R_Smooth": r_smooth}
        
        return total, details

    def close(self):
        if self.vs and self.configuration:
            self.vs.terminate(self.t_current)
        cv2.destroyAllWindows()
        
    def get_state_dim(self): return self.state_dim
    def get_action_dim(self): return self.action_dim

if __name__ == "__main__":
    #测试
    env = LiveCarsimEnv()
    try:
        state, _ = env.reset()
        step_count = 0
        while True:
            # 根据时间控制扭矩 (示例：线性增加)
            t = env.t_current
            torque_val = min(t, 0.8) # 随时间增加，最大 80%
            
            action = np.array([torque_val, torque_val, torque_val*0.70, torque_val*0.70]) 
            next_state, reward, done, info = env.step(action)
            
            if step_count % 100 == 0:
                print(f"Step {step_count}, Time {t:.2f}s, Action {torque_val:.2f}, Vx {info['vx']:.2f}")
            
            step_count += 1
            
            if done:
                print("Episode finished.")
                break
    finally:
        env.close()
