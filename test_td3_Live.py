#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TD3-Live Testing Script
用于测试训练好的 TD3 模型在 CarSim 环境中的表现
"""
import numpy as np
import torch
import os
import random
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from td3_agent import TD3Agent
from env_live import LiveCarsimEnv
import shutil
import argparse

def save_episode_data(episode_num, history, reward, info, save_dir):
    if not save_dir:
        return
        
    data_dir = os.path.join(save_dir, "test_data")
    os.makedirs(data_dir, exist_ok=True)
    
    episode_data = {
        'episode': episode_num,
        'total_reward': float(reward),
        'num_steps': len(history.get('velocity', [])),
    }
    
    for key, value in history.items():
        if isinstance(value, list) and len(value) > 0:
            try:
                float_values = []
                for x in value:
                    if hasattr(x, 'item'):
                        float_values.append(float(x.item()))
                    elif hasattr(x, 'tolist'):
                        float_values.append(float(x))
                    else:
                        float_values.append(float(x))
                episode_data[key] = float_values
            except (ValueError, TypeError, AttributeError):
                episode_data[key] = value
    
    for key, value in info.items():
        if isinstance(value, (int, float, str, np.integer, np.floating)):
            if hasattr(value, 'item'):
                episode_data[f'info_{key}'] = value.item()
            else:
                episode_data[f'info_{key}'] = value
    
    save_path = os.path.join(data_dir, f'test_episode_{episode_num}.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(episode_data, f, indent=2, ensure_ascii=False)
    
    print(f"Episode data saved to: {save_path}")
    return save_path

def test_td3_Live(
    model_path: str,
    num_episodes: int = 5,
    max_torque: float = 500.0,
    target_slip_ratio: float = 0.08,
    log_dir: str = "logs_TD3_Test"
):
    reward_weights = {
        'w_speed': 0.00,
        'w_accel': 0.15,
        'w_energy': 0,
        'w_consistency': 0,
        'w_beta': -0.02,
        'w_slip': -0.25,
        'w_smooth': -0.01,
        'w_yaw': -2.0,
    }
    
    # 只需要加载 Actor 网络所需的参数
    hyperparams = {
        'Action Bound': 1.0,   
        'Hidden Dim': 256,
        # 其他参数在测试时不需要，但为了初始化 Agent 保持一致
        'gamma': 0.99,
        'Actor LR': 2e-4,
        'Critic LR': 2e-4,
        'Buffer Capacity': 1000, # 测试不需要大 Buffer
        'Batch Size': 512,
        'Elite Ratio': 0.3,    
        'Elite Capacity': 1000,
        'Policy Noise': 0.1,
        'Noise Clip': 0.25,
        'Policy Freq': 2,
    }
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"TD3_Test_{current_time}")
    os.makedirs(log_path, exist_ok=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"工作目录: {os.getcwd()}")
    
    SIMFILE_PATH = os.path.join(script_dir, "simfile.sim")
    VS_DLL_PATH = os.path.join(script_dir, "vs_lv_ds_x64.dll")
    
    if not os.path.exists(SIMFILE_PATH):
        print(f"找不到 simfile.sim: {SIMFILE_PATH}")
        return

    env = LiveCarsimEnv(
        simfile_path=SIMFILE_PATH,
        vs_dll_path=VS_DLL_PATH,
        sim_time_s=10.0,       
        max_torque=max_torque,
        target_slip_ratio=target_slip_ratio,
        reward_weights=reward_weights,
        frame_skip=1
    )
    
    agent = TD3Agent(
        state_dim=env.get_state_dim(),
        action_dim=env.get_action_dim(),
        action_bound=hyperparams['Action Bound'],
        hidden_dim=hyperparams['Hidden Dim'],
        actor_lr=hyperparams['Actor LR'],
        critic_lr=hyperparams['Critic LR'],
        buffer_capacity=hyperparams['Buffer Capacity'],
        batch_size=hyperparams['Batch Size'],
        gamma=hyperparams['gamma'],
        policy_noise=hyperparams['Policy Noise'],
        noise_clip=hyperparams['Noise Clip'],
        policy_freq=hyperparams['Policy Freq'],
        elite_ratio=hyperparams['Elite Ratio'],
        elite_capacity=hyperparams['Elite Capacity']   
    )
    
    if os.path.exists(model_path):
        print(f"加载模型: {model_path}")
        agent.load_model(model_path)
    else:
        print(f"错误: 找不到模型文件 {model_path}")
        return

    print(f"\n========== Start TD3 Testing ({num_episodes} episodes) ==========")
    
    try:
        for episode in range(num_episodes):
            state, info = env.reset()
            episode_reward = 0
            
            episode_visual_history = {
                'T_L1': [], 'T_R1': [], 'T_L2': [], 'T_R2': [],
                'S_L1': [], 'S_R1': [], 'S_L2': [], 'S_R2': [],
                'velocity': [], 
                'r_total': [], 
                'target_slip': []
            }
            final_info = {}
            
            while True:
                # 测试时不需要噪声，直接输出确定性动作
                action = agent.select_action(state, noise_scale=0.0)

                next_state, reward, done, info = env.step(action)
                final_info = info
                
                episode_visual_history['T_L1'].append(info.get('trq_L1', 0))
                episode_visual_history['T_R1'].append(info.get('trq_R1', 0))
                episode_visual_history['T_L2'].append(info.get('trq_L2', 0))
                episode_visual_history['T_R2'].append(info.get('trq_R2', 0))
                
                episode_visual_history['S_L1'].append(info.get('slip_L1', 0))
                episode_visual_history['S_R1'].append(info.get('slip_R1', 0))
                episode_visual_history['S_L2'].append(info.get('slip_L2', 0))
                episode_visual_history['S_R2'].append(info.get('slip_R2', 0))
                episode_visual_history['target_slip'].append(target_slip_ratio)
                
                episode_visual_history['velocity'].append(info.get('vx', 0))
                episode_visual_history['r_total'].append(reward)
                
                for k, v in info.items():
                    if k.startswith("R_"):
                        if k not in episode_visual_history: episode_visual_history[k] = []
                        episode_visual_history[k].append(v)

                if done:
                    final_v = info.get('vx', 0.0)
                    r_terminal = final_v * 0.2 
                    reward += r_terminal
                    episode_visual_history['r_total'][-1] = reward

                state = next_state
                episode_reward += reward
                
                if done: break
            
            final_speed = final_info.get('vx', 0.0)
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"\033[92m{'='*60}\033[0m")
            print(f"\033[92m[{timestamp}] TEST EPISODE {episode+1}/{num_episodes} | Reward: {episode_reward:.0f} | Final Speed: {final_speed:.1f} km/h\033[0m")
            print(f"\033[92m{'='*60}\033[0m")

            save_episode_data(episode, episode_visual_history, episode_reward, final_info, log_path)
            
    except KeyboardInterrupt:
        print("=====停止测试=====")
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TD3 Live Testing")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model (.pt)")
    parser.add_argument("--episodes", type=int, default=1, help="Number of test episodes")
    
    args = parser.parse_args()
    
    test_td3_Live(
        model_path=args.model,
        num_episodes=args.episodes
    )
