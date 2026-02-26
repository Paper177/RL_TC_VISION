#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DDPG-Live Testing Script
Loads a trained DDPG model and runs it in the LiveCarsimEnv environment without noise or training.
"""
import numpy as np
import torch
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from ddpg_agent import DDPGAgent
from env_live import LiveCarsimEnv
import argparse

#================= 绘图工具 =================
def log_episode_visuals(episode_num, history, save_dir=None):
    plt.close('all')
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Test Episode {episode_num} Analysis', fontsize=16)
    
    steps = range(len(history['velocity']))
    
    # 1. Torques
    ax = axes[0, 0]
    ax.plot(steps, history['T_L1'], label='L1 (FL)', alpha=0.8, linewidth=1)
    ax.plot(steps, history['T_R1'], label='R1 (FR)', alpha=0.8, linewidth=1)
    ax.plot(steps, history['T_L2'], label='L2 (RL)', alpha=0.8, linewidth=1)
    ax.plot(steps, history['T_R2'], label='R2 (RR)', alpha=0.8, linewidth=1)
    ax.set_title('Wheel Torques (Nm)')
    ax.set_ylabel('Torque')
    ax.legend(loc='upper right', fontsize='small')
    ax.grid(True, alpha=0.3)

    # 2. Slips
    ax = axes[0, 1]
    ax.plot(steps, history['S_L1'], label='L1', alpha=0.8)
    ax.plot(steps, history['S_R1'], label='R1', alpha=0.8)
    ax.plot(steps, history['S_L2'], label='L2', alpha=0.8)
    ax.plot(steps, history['S_R2'], label='R2', alpha=0.8)
    
    if 'target_slip' in history and len(history['target_slip']) > 0:
        target_val = history['target_slip'][0]
        ax.axhline(y=target_val, color='r', linestyle='--', alpha=0.5, label=f'Target ({target_val})')
        
    ax.set_title('Slip Ratios')
    ax.set_ylabel('Slip Ratio')
    ax.set_ylim(-0.05, 0.2)
    ax.legend(loc='upper right', fontsize='small')
    ax.grid(True, alpha=0.3)

    # 3. Velocity
    ax = axes[1, 0]
    ax.plot(steps, history['velocity'], 'b-', label='Actual Speed', linewidth=2)
    ax.set_title('Vehicle Velocity (km/h)')
    ax.set_ylabel('Speed (km/h)')
    ax.set_ylim(0, 100)
    ax.set_xlabel('Step')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Rewards
    ax = axes[1, 1]
    ax.plot(steps, history['r_total'], 'k-', label='Total Reward', linewidth=2, alpha=0.9)
    ax.set_title('Reward History')
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'test_episode_{episode_num}.png')
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")
        
    plt.close(fig)

def test_ddpg(model_path, max_episodes=1, target_speed=100):
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    # --- 配置 (需与训练时一致) ---
    hyperparams = {
        'Action Bound': 1.0,   
        'Hidden Dim': 256,
        # 以下参数在测试时主要用于初始化Agent结构
        'Actor LR': 0,      
        'Critic LR': 0,
        'Buffer Capacity': 1000,
        'Batch Size': 64,
        'Elite Ratio': 0.4,    
        'Elite Capacity': 2000,
    }
    
    # --- 环境初始化 ---
    SIMFILE_PATH = os.path.join(os.getcwd(), "simfile.sim")
    VS_DLL_PATH = r"E:\CarSim2024.1_Prog\Database\RL_VISION\RL_TC_VISION\vs_lv_ds_x64.dll"
    
    if not os.path.exists(SIMFILE_PATH):
        print(f"Error: simfile.sim not found at {SIMFILE_PATH}")
        return

    # 测试时通常不需要 reward_weights，但环境初始化需要
    reward_weights = {
        'w_speed': 0.0, 'w_accel': 0.0, 'w_energy': 0.0,
        'w_consistency': 0, 'w_beta': 0.0, 'w_slip': 0.0,
        'w_smooth': 0.0, 'w_yaw': 0.0,
    }

    print("Initializing Environment...")
    env = LiveCarsimEnv(
        simfile_path=SIMFILE_PATH,
        vs_dll_path=VS_DLL_PATH,
        sim_time_s=10.0,       
        max_torque=500.0,
        target_slip_ratio=0.04, # 仅用于绘图参考
        target_speed=target_speed,
        reward_weights=reward_weights
    )
    
    # --- Agent 初始化 ---
    print("Initializing Agent...")
    agent = DDPGAgent(
        state_dim=env.get_state_dim(),
        action_dim=env.get_action_dim(),
        action_bound=hyperparams['Action Bound'],
        hidden_dim=hyperparams['Hidden Dim'],
        actor_lr=hyperparams['Actor LR'],
        critic_lr=hyperparams['Critic LR'],
        buffer_capacity=hyperparams['Buffer Capacity'],
        batch_size=hyperparams['Batch Size'],
        elite_ratio=hyperparams['Elite Ratio'],
        elite_capacity=hyperparams['Elite Capacity']   
    )
    
    # --- 加载模型 ---
    print(f"Loading model from: {model_path}")
    try:
        # 注意：如果你的 pt 文件只包含 actor，这里需要调整
        # 假设 pt 文件是 train_ddpg_Live.py 保存的完整字典
        agent.load_model(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting to load only actor state_dict if keys mismatch...")
        try:
            checkpoint = torch.load(model_path)
            if 'actor' in checkpoint:
                agent.actor.load_state_dict(checkpoint['actor'])
            else:
                agent.actor.load_state_dict(checkpoint) # 假设整个文件就是 state_dict
            print("Actor model loaded successfully.")
        except Exception as e2:
            print(f"Failed to load model: {e2}")
            return

    # --- 测试循环 ---
    save_dir = "test_results"
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n========== Start Testing ==========")
    
    try:
        for episode in range(max_episodes):
            state, info = env.reset()
            episode_reward = 0
            
            # 数据记录
            history = {
                'T_L1': [], 'T_R1': [], 'T_L2': [], 'T_R2': [],
                'S_L1': [], 'S_R1': [], 'S_L2': [], 'S_R2': [],
                'velocity': [], 
                'r_total': [], 
                'target_slip': []
            }
            
            print(f"Episode {episode+1} running...")
            
            while True:
                # Select Action (No Noise for Testing)
                action = agent.select_action(state, noise_scale=0.0)

                # Step
                next_state, reward, done, info = env.step(action)
                
                # 记录
                history['T_L1'].append(info.get('trq_L1', 0))
                history['T_R1'].append(info.get('trq_R1', 0))
                history['T_L2'].append(info.get('trq_L2', 0))
                history['T_R2'].append(info.get('trq_R2', 0))
                
                history['S_L1'].append(info.get('slip_L1', 0))
                history['S_R1'].append(info.get('slip_R1', 0))
                history['S_L2'].append(info.get('slip_L2', 0))
                history['S_R2'].append(info.get('slip_R2', 0))
                history['target_slip'].append(0.04) # 假设目标滑移率
                
                history['velocity'].append(info.get('vx', 0))
                history['r_total'].append(reward)

                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            final_speed = info.get('vx', 0.0)
            print(f"Episode {episode+1} Finished. Total Reward: {episode_reward:.2f}, Final Speed: {final_speed:.2f} km/h")
            
            # 绘图
            log_episode_visuals(episode+1, history, save_dir=save_dir)

    except KeyboardInterrupt:
        print("Testing interrupted by user.")
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("Environment closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test DDPG Model')
    parser.add_argument('--model', type=str, required=True, help='Path to the .pt model file')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to run')
    
    args = parser.parse_args()
    
    test_ddpg(args.model, args.episodes)
