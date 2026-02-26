#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DDPG-Live Training Script
Integrates TorqueControl.py simulation loop with DDPG training
"""
import numpy as np
import torch
import os
import random
import io
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from dashboard import create_live_monitor
from ddpg_agent import DDPGAgent
from env_live import LiveCarsimEnv # 使用新的实时环境
import gc
import shutil

#================= 绘图与日志 (复用 train_ddpg_PC.py) =================
def log_episode_visuals(writer, episode_num, history, save_dir=None, reward=None):
    plt.close('all')
    gc.collect()
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Detailed Analysis', fontsize=16)
    
    steps = range(len(history['velocity']))
    
    ax = axes[0, 0]
    ax.plot(steps, history['T_L1'], label='L1 (FL)', alpha=0.8, linewidth=1)
    ax.plot(steps, history['T_R1'], label='R1 (FR)', alpha=0.8, linewidth=1)
    ax.plot(steps, history['T_L2'], label='L2 (RL)', alpha=0.8, linewidth=1)
    ax.plot(steps, history['T_R2'], label='R2 (RR)', alpha=0.8, linewidth=1)
    ax.set_title('Wheel Torques (Nm)')
    ax.set_ylabel('Torque')
    ax.legend(loc='upper right', fontsize='small')
    ax.grid(True, alpha=0.3)

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
    ax.set_ylim(-0.05, 0.5)
    ax.legend(loc='upper right', fontsize='small')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(steps, history['velocity'], 'b-', label='Actual Speed', linewidth=2)
    ax.set_title('Vehicle Velocity (km/h)')
    ax.set_ylabel('Speed (km/h)')
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 10))
    ax.set_xlabel('Step')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(steps, history['r_total'], 'k-', label='Total Reward', linewidth=2, alpha=0.9)
    
    for key in history:
        if key.startswith('R_') and len(history[key]) == len(steps):
            ax.plot(steps, history[key], label=key[2:], linestyle='--', alpha=0.7) 
    
    ax.set_title('Reward Composition')
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward Value')
    ax.set_ylim(-0.05, 0.1)
    ax.legend(loc='lower left', fontsize='x-small', ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    
    if save_dir:
        figures_dir = os.path.join(save_dir, "episode_plots")
        os.makedirs(figures_dir, exist_ok=True)
        save_path = os.path.join(figures_dir, f'episode_{episode_num}_{reward}.png')
        plt.savefig(save_path)
        
    plt.close(fig)

def save_episode_data(episode_num, history, reward, info, save_dir):
    """保存episode的原始数据用于后续分析和奖励函数改进"""
    if not save_dir:
        return
        
    data_dir = os.path.join(save_dir, "episode_data")
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
    
    save_path = os.path.join(data_dir, f'episode_{episode_num}.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(episode_data, f, indent=2, ensure_ascii=False)
    
    return save_path

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"随机种子已锁定为: {seed}")

def train_ddpg_Live(
    max_episodes: int = 10000,
    max_torque: float = 500.0,
    target_slip_ratio: float = 0.08,
    log_dir: str = "logs_Live",
    pretrained_model_path: str = None 
):
    # --- 配置 ---
    reward_weights = {
        'w_speed': 0.00,
        'w_accel': 0.12,
        'w_energy': 0.01,
        'w_consistency': -0.02,
        'w_beta': -0.1,
        'w_slip': -0.05,
        'w_smooth': 0.0,
        'w_yaw': -5.0,
    }
    
    hyperparams = {
        'Action Bound': 1.0,   
        'Hidden Dim': 256,
        'gamma': 0.99995,
        'Actor LR': 5e-6,      # 降低学习率 
        'Critic LR': 5e-5,     # 降低学习率
        'Buffer Capacity': 1000000,
        'Batch Size': 256,
        'Elite Ratio': 0.4,    
        'Elite Capacity': 200000,
        'Noise Scale': 0.2,    
        'Min Noise': 0.02,     
        'Noise Decay': 0.998,  
    }
    
    # --- 日志初始化 ---
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"Live_Carsim_{current_time}")
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_dir=log_path)

    # --- 备份脚本 ---
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        train_file = os.path.join(current_dir, "train_ddpg_Live.py")
        env_file = os.path.join(current_dir, "env_live.py")
        
        shutil.copy2(train_file, os.path.join(log_path, "train_ddpg_Live.py"))
        shutil.copy2(env_file, os.path.join(log_path, "env_live.py"))
    except Exception as e:
        print(f"[Warning] 备份文件失败: {e}")

    # --- 环境初始化 ---
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 切换到脚本目录确保路径正确
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
        frame_skip=5
    )
    
    # --- DDPG初始化 ---
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
    
    start_episode = 0
    best_episode_reward = -float('inf') 
    min_noise = hyperparams['Min Noise']
    noise_decay = hyperparams['Noise Decay']
    
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"加载预训练模型: {pretrained_model_path}")
        agent.load_model(pretrained_model_path)
        
        # 尝试从文件名解析 episode 和 reward
        try:
            filename = os.path.basename(pretrained_model_path)
            parts = filename.replace('.pt', '').split('_')
            # 假设格式: Live_Carsim_YYYYMMDD_HHMMSS_EPISODE_REWARD
            # parts[-2] 是 episode
            # parts[-1] 是 reward
            parsed_episode = int(parts[-2])
            parsed_reward = float(parts[-1])
            
            start_episode = parsed_episode + 1
            best_episode_reward = parsed_reward
            print(f"恢复训练: Episode {start_episode}, Best Reward {best_episode_reward}")
            
            # 恢复噪声衰减
            noise_scale = max(min_noise, hyperparams['Noise Scale'] * (noise_decay ** start_episode))
        except Exception as e:
            print(f"无法从文件名解析信息，使用默认设置: {e}")
            noise_scale = 0.1 
    else:
        print("从零开始训练")
        noise_scale = hyperparams['Noise Scale']

    print("\n========== Start Live Training ==========")
    # live_display, dashboard = create_live_monitor()
    
    try:
        # with live_display:
            for episode in range(start_episode, max_episodes):
                # Reset
                state, info = env.reset()
                agent.reset_noise() 
                episode_reward = 0

                reward_stats = {} 
                current_episode_memory = []
                episode_visual_history = {
                    'T_L1': [], 'T_R1': [], 'T_L2': [], 'T_R2': [],
                    'S_L1': [], 'S_R1': [], 'S_L2': [], 'S_R2': [],
                    'velocity': [], 
                    'r_total': [], 
                    'target_slip': []
                }
                critic_grads = []
                actor_grads = []
                is_elite_display = False
                final_info = {}
                
                while True:
                    # Select Action
                    action = agent.select_action(state, noise_scale=noise_scale)

                    # Step
                    next_state, reward, done, info = env.step(action)
                    final_info = info
                    
                    # --- 收集数据 ---
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
                            if k not in reward_stats: reward_stats[k] = []
                            reward_stats[k].append(v)
                            if k not in episode_visual_history: episode_visual_history[k] = []
                            episode_visual_history[k].append(v)

                    if done:
                        final_v = info.get('vx', 0.0)
                        r_terminal = final_v * 0.2 
                        reward += r_terminal
                        if 'R_Terminal' not in reward_stats: reward_stats['R_Terminal'] = []
                        reward_stats['R_Terminal'].append(r_terminal)
                        episode_visual_history['r_total'][-1] = reward

                    # Push & Train
                    agent.push(state, action, reward, next_state, done)
                    current_episode_memory.append((state, action, reward, next_state, done))
                    
                    #c_loss, a_loss, c_grad, a_grad = agent.train_step()
                    if env.current_step % 50 == 0:
                        c_loss, a_loss, c_grad, a_grad = agent.train_step()
                    else:
                        c_loss, a_loss = 0, 0
                        
                    if c_loss != 0:
                        critic_grads.append(c_grad)
                        actor_grads.append(a_grad)
                    
                    state = next_state
                    episode_reward += reward
                    
                    # if env.current_step % 10 == 0 or done:
                    #     dashboard.update(
                    #         episode=episode+1,
                    #         step=env.current_step,
                    #         info=info,
                    #         reward=reward,
                    #         noise=noise_scale,
                    #         elite_flag=is_elite_display
                    #     )
                    
                    if done: break
                
                # --- Episode End ---
                if (episode) % 3 == 0:
                    log_episode_visuals(writer, episode, episode_visual_history, save_dir=log_path, reward=episode_reward)
                    save_episode_data(episode, episode_visual_history, episode_reward, final_info, log_path)
                
                sum_rewards = {k: np.sum(v) for k, v in reward_stats.items()}
                avg_c = np.mean(critic_grads) if critic_grads else 0
                avg_a = np.mean(actor_grads) if actor_grads else 0
                final_speed = final_info.get('vx', 0.0)
                
                writer.add_scalar('Loss/Critic', c_loss, episode)
                writer.add_scalar('Loss/Actor', a_loss, episode)
                writer.add_scalar('Train/Reward', episode_reward, episode)
                writer.add_scalar('Train/Noise', noise_scale, episode)
                writer.add_scalar('Train/Final_Speed_kmh', final_speed, episode)
                
                # dashboard.log(f"Ep {episode}| Rw: {episode_reward:.0f} | EndV: {final_speed:.1f}")
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"\033[93m{'='*60}\033[0m")
                print(f"\033[93m[{timestamp}] EPISODE {episode} | Reward: {episode_reward:.0f} | Final Speed: {final_speed:.1f} km/h\033[0m")
                print(f"\033[93m{'='*60}\033[0m")
                
                if episode_reward > best_episode_reward*0.9 and episode_reward >=0:
                    is_elite_display = True
                    writer.add_scalar('Train/Is_Elite', 1, episode)
                    # dashboard.log(f"[精英]! Reward: {episode_reward:.1f}")
                    log_episode_visuals(writer, episode, episode_visual_history, save_dir=log_path, reward=episode_reward)
                    save_episode_data(episode, episode_visual_history, episode_reward, final_info, log_path)
                    print(f"[{timestamp}] [精英]! Reward: {episode_reward:.1f}")
                    for trans in current_episode_memory:
                        agent.push_elite(*trans)
                    if episode_reward > best_episode_reward:
                        best_episode_reward = episode_reward
                        agent.save_model(os.path.join("best_model_save", f"Live_Carsim_{current_time}_{episode}_{best_episode_reward}.pt"))
                        log_episode_visuals(writer, episode, episode_visual_history, save_dir=log_path, reward=episode_reward)
                        save_episode_data(episode, episode_visual_history, episode_reward, final_info, log_path)
                        # dashboard.log(f"[新纪录] ! Reward: {episode_reward:.1f}")
                        print(f"[{timestamp}] [新纪录] ! Reward: {episode_reward:.1f}")
                else:
                    writer.add_scalar('Train/Is_Elite', 0, episode)
                    noise_scale = max(min_noise, noise_scale * noise_decay)

    except KeyboardInterrupt:
        print("=====停止训练=====")
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        agent.save_model(os.path.join(log_path, "final_model.pt"))
        print("资源已释放，训练结束。")

if __name__ == "__main__":
    setup_seed(42)
    train_ddpg_Live(
        pretrained_model_path=None,
    )
