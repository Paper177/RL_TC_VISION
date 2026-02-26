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

import torch
import torch.nn as nn
import torch.nn.functional as F

import GetSharedBufferInfo
import Simulation_with_LiveAnimation
from GetSharedBufferInfo import VSSBCT_RGB, VSSBCT_DEPTH
from ctypes import cast, POINTER, c_ubyte, c_ulong

class VisionFeatureExtractor(nn.Module):
    """
    视觉特征提取网络，输入为BGR格式的numpy数组，输出为一维特征向量。
    """
    def __init__(self, output_dim=128):
        super(VisionFeatureExtractor, self).__init__()
        # 假设输入为(3, 120, 160)，可根据实际分辨率调整
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 8 * 10, output_dim)  # 假设输入为(3,120,160)

    def forward(self, x):
        # x: (B, 3, H, W)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x

    @staticmethod
    def preprocess(np_img, resize_hw=(120, 160)):
        # np_img: (H, W, 3), BGR, uint8
        img = cv2.resize(np_img, resize_hw[::-1])
        img = img.astype(np.float32) / 255.0
        img = img[..., ::-1]  # BGR to RGB
        img = np.transpose(img, (2, 0, 1)).copy()  # (3, H, W)
        return torch.from_numpy(img).unsqueeze(0)  # (1, 3, H, W)

    def get_features_from_np(self, np_img, device="cpu"):
        """直接输入numpy图像，返回特征向量 (1, output_dim)"""
        self.eval()
        with torch.no_grad():
            x = self.preprocess(np_img).to(device)
            feat = self.forward(x)
        return feat.cpu().numpy().squeeze()

class LiveCarsimEnv:
    """
    基于 TorqueControl.py 的实时 CarSim 环境
    集成了实时图像显示和 DDPG 训练接口
    """
    
    def __init__(
        self,
        simfile_path: str = "simfile.sim",
        vs_dll_path: str = r"E:\CarSim2024.1_Prog\Database\RL_VISION\RL_TC_VISION\vs_lv_ds_x64.dll",
        sim_time_s: float = 10.0,
        max_torque: float = 1000.0,
        target_slip_ratio: float = 0.06,
        reward_weights: dict = None,
        frame_skip: int = 1,
        vision_feature_dim: int = 128,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.simfile_path = simfile_path
        self.vs_dll_path = vs_dll_path
        self.sim_time_s = sim_time_s
        self.max_torque = max_torque
        self.target_slip_ratio = target_slip_ratio
        self.frame_skip = frame_skip
        self.vision_feature_dim = vision_feature_dim
        self.device = device
        
        # 初始化奖励权重
        default_weights = {
            'w_speed': 0.1, 'w_accel': 0.0, 'w_energy': 0.0,
            'w_consistency': 0.0, 'w_beta': 0.0, 'w_slip': -1.0, 'w_smooth': 0.0,
            'w_yaw': -50.0
        }
        self.weights = default_weights.copy()
        if reward_weights:
            self.weights.update(reward_weights)
            
        # 车辆物理参数
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
        
        # 状态空间维度: 物理状态(8) + 视觉特征(vision_feature_dim)
        self.physics_state_dim = 8
        self.state_dim = self.physics_state_dim + self.vision_feature_dim
        # 动作空间维度: [T_L1, T_R1, T_L2, T_R2]
        self.action_dim = 4
        
        # 视觉特征提取器
        self.vision_extractor = VisionFeatureExtractor(output_dim=vision_feature_dim).to(self.device)
        self.vision_extractor.eval() # 推理模式
        
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
        
        # 获取视觉特征
        vision_feat, _ = self._get_vision_data(display=True, info=None, reward=0.0, step=0)
        if vision_feat is None:
            vision_feat = np.zeros(self.vision_feature_dim, dtype=np.float32)
        
        # 解析状态
        physics_state = self._parse_observation(self.export_array)
        norm_physics_state = self._normalize_state(physics_state)
        
        # 拼接状态
        state = np.concatenate([norm_physics_state, vision_feat])
        
        # 构造初始 info 以便显示
        info = {
            "vx": physics_state[0],
            "ax": physics_state[1],
            "yaw": physics_state[6],
            "beta": physics_state[7],
            "slip_L1": physics_state[2],
            "slip_R1": physics_state[3],
            "slip_L2": physics_state[4],
            "slip_R2": physics_state[5],
            "trq_L1": 0.0,
            "trq_R1": 0.0,
            "trq_L2": 0.0,
            "trq_R2": 0.0
        }
        
        return state, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行一步"""
        # 1. 动作处理
        action = np.clip(action, 0.0, 1.0)
        target_torque = action * self.max_torque
        
        # 更新输入数组 [L1, R1, L2, R2]
        self.import_array = [target_torque[0], target_torque[1], target_torque[2], target_torque[3]]
        
        # 2. 物理仿真步进 (Frame Skip)
        for _ in range(self.frame_skip):
            self.t_current = self.configuration.get('t_start') + (self.current_step + 1) * self.t_step
            self.status, self.export_array = self.vs.updateonestep(self.t_current, self.import_array, self.export_array)
            self.current_step += 1
            if self.status != 0:
                break
                
        # 4. 状态解析
        physics_state = self._parse_observation(self.export_array)
        norm_physics_state = self._normalize_state(physics_state)
        
        # 5. 计算奖励
        reward, r_details = self._calculate_reward(physics_state, target_torque, self.last_torque)
        
        self.last_torque = target_torque
        
        if self.status != 0:
            print(f"[Env] Simulation stopped early at step {self.current_step}, time {self.t_current:.4f}, status {self.status}")
        done = (self.current_step >= self.max_steps) or (self.status != 0)

        # 7. Info
        info = {
            "vx": physics_state[0],
            "ax": physics_state[1],
            "yaw": physics_state[6],
            "beta": physics_state[7],
            "slip_L1": physics_state[2],
            "slip_R1": physics_state[3],
            "slip_L2": physics_state[4],
            "slip_R2": physics_state[5],
            "trq_L1": target_torque[0],
            "trq_R1": target_torque[1],
            "trq_L2": target_torque[2],
            "trq_R2": target_torque[3],
            **r_details
        }

        # 3. 获取视觉特征并显示
        # 每步都获取特征，但只每10步显示一次以节省资源
        display = (self.current_step % 10 == 0)
        vision_feat, _ = self._get_vision_data(display=display, info=info, reward=reward, step=self.current_step)
        
        if vision_feat is None:
            vision_feat = np.zeros(self.vision_feature_dim, dtype=np.float32)
            
        # 拼接状态
        next_state = np.concatenate([norm_physics_state, vision_feat])
        
        return next_state, reward, done, info

    def _get_vision_data(self, display=False, info=None, reward=0.0, step=0):
        """获取视觉数据并提取特征，可选显示"""
        vision_feat = None
        frame = None
        
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
                    
                    # 提取特征
                    vision_feat = self.vision_extractor.get_features_from_np(frame, device=self.device)
                    
                    # 显示信息
                    if display and info:
                        # 第一行：时间、步数、奖励
                        cv2.putText(frame, f"Time: {self.t_current:.2f}s | Step: {step} | Reward: {reward:.4f}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        # 第二行：速度、Yaw、Beta
                        cv2.putText(frame, f"Vx: {info['vx']:.1f} km/h | Yaw: {info['yaw']:.3f} rad/s | Beta: {info['beta']:.2f} deg", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        # 第三行：滑移率
                        slip_str = f"Slip: {info['slip_L1']:.4f}/{info['slip_R1']:.4f} (F) | {info['slip_L2']:.4f}/{info['slip_R2']:.4f} (R)"
                        cv2.putText(frame, slip_str, (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        # 第四行：扭矩
                        trq_str = f"Torque: {info['trq_L1']:.0f}/{info['trq_R1']:.0f} (F) | {info['trq_L2']:.0f}/{info['trq_R2']:.0f} (R)"
                        cv2.putText(frame, trq_str, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        cv2.imshow('CarSim Live Training View', frame)
                        cv2.waitKey(1)
            except Exception as e:
                print(f"Vision error: {e}")
            finally:
                GetSharedBufferInfo.api_ver2_contents.UnLock(GetSharedBufferInfo.sbHandle)
                
        return vision_feat, frame

    def _parse_observation(self, export_array) -> np.ndarray:
        """
        解析 CarSim 输出数组
        """
        # 安全检查
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
        
        v_L1_c = ((vx_ms - avz*0.5*self.veh_bf)*np.cos(steer)  + (vy+avz*self.veh_lf)*np.sin(steer)) * 3.6
        v_R1_c = ((vx_ms + avz*0.5*self.veh_bf)*np.cos(steer)  + (vy+avz*self.veh_lf)*np.sin(steer)) * 3.6
        v_L2_c = ((vx_ms - avz*0.5*self.veh_br)*np.cos(steer) ) * 3.6 
        v_R2_c = ((vx_ms + avz*0.5*self.veh_br)*np.cos(steer) ) * 3.6
        
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
        n_s[7] = raw_state[7] / 1.0 
        return n_s

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
            for i in range(4):
                r_slip += max(0.0, slips[i] - self.target_slip_ratio)
        r_slip = w['w_slip'] * r_slip  

        r_energy = w['w_energy'] * np.mean(np.abs(current_torque/self.max_torque))
        r_consistency = w['w_consistency'] * (abs(current_torque[0] - current_torque[1])+abs(current_torque[2] - current_torque[3]))/self.max_torque
        r_smooth = w['w_smooth'] * np.mean(((current_torque - last_torque)/self.max_torque)**2)
        
        r_yaw = 0.0
        if abs(yaw_rate) > 0.01:
            r_yaw = w['w_yaw'] * (abs(yaw_rate) - 0.01)

        r_beta = w['w_beta'] * abs(beta)

        total = r_speed + r_accel + r_slip + r_energy + r_consistency + r_smooth + r_yaw + r_beta
        total = np.clip(total, -10.0, 10.0)
        details = {"R_Spd": r_speed, "R_Acc": r_accel, "R_Slp": r_slip, "R_Eng": r_energy, "R_Cns": r_consistency, "R_Yaw": r_yaw, "R_Beta": r_beta}
        
        return total, details

    def close(self):
        if self.vs and self.configuration:
            self.vs.terminate(self.t_current)
        cv2.destroyAllWindows()
        
    def get_state_dim(self): return self.state_dim
    def get_action_dim(self): return self.action_dim

if __name__ == "__main__":
    env = LiveCarsimEnv()
    try:
        state, _ = env.reset()
        print(f"State dim: {len(state)}")
        step_count = 0
        while True:
            t = env.t_current
            torque_val = min(t, 0.8)
            action = np.array([torque_val, torque_val, torque_val*0.70, torque_val*0.70]) 
            next_state, reward, done, info = env.step(action)
            
            if step_count % 100 == 0:
                print(f"Step {step_count}, Time {t:.2f}s, Action {torque_val:.2f}, Vx {info['vx']:.2f}")
            
            step_count += 1
            if done:
                break
    finally:
        env.close()
