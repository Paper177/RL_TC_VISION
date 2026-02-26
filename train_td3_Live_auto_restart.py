#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TD3-Live Training Script (Auto-Restart Version)
每500个episode自动保存并退出，需要手动重新运行继续训练
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
from td3_agent import TD3Agent
from env_live import LiveCarsimEnv
import gc
import shutil
import argparse
from tqdm import tqdm

#================= 绘图与日志 =================
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
    
    # 对滑移率数据进行平滑处理
    def moving_average(data, window_size=10):
        if len(data) < window_size:
            return data
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window_size // 2)
            end = min(len(data), i + window_size // 2 + 1)
            smoothed.append(np.mean(data[start:end]))
        return smoothed
    
    s_l1_smooth = moving_average(history['S_L1'])
    s_r1_smooth = moving_average(history['S_R1'])
    s_l2_smooth = moving_average(history['S_L2'])
    s_r2_smooth = moving_average(history['S_R2'])
    
    ax.plot(steps, s_l1_smooth, label='L1', alpha=0.8, linewidth=1.5)
    ax.plot(steps, s_r1_smooth, label='R1', alpha=0.8, linewidth=1.5)
    ax.plot(steps, s_l2_smooth, label='L2', alpha=0.8, linewidth=1.5)
    ax.plot(steps, s_r2_smooth, label='R2', alpha=0.8, linewidth=1.5)
    
    if 'target_slip' in history and len(history['target_slip']) > 0:
        target_val = history['target_slip'][0]
        ax.axhline(y=target_val, color='r', linestyle='--', alpha=0.5, label=f'Target ({target_val})')
        
    ax.set_title('Slip Ratios (Smoothed)')
    ax.set_ylabel('Slip Ratio')
    ax.set_ylim(-0.05, 0.2)
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

def train_td3_Live(
    max_episodes: int = 10000,
    max_torque: float = 500.0,
    target_slip_ratio: float = 0.08,
    log_dir: str = "logs_TD3",
    pretrained_model_path: str = None,
    checkpoint_interval: int = 300
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
    
    hyperparams = {
        'Action Bound': 1.0,   
        'Hidden Dim': 256,
        'gamma': 0.99,
        'Actor LR': 3e-4,
        'Critic LR': 3e-4,
        'Buffer Capacity': 1500000,
        'Batch Size': 512,
        'Elite Ratio': 0.3,    
        'Elite Capacity': 1000000,
        'Noise Scale': 0.1,
        'Min Noise': 0.01,
        'Noise Decay': 0.995,
        'Policy Noise': 0.1,
        'Noise Clip': 0.25,
        'Policy Freq': 2,
    }
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"TD3_Carsim_{current_time}")
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_dir=log_path)

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        train_file = os.path.join(current_dir, "train_td3_Live_auto_restart.py")
        env_file = os.path.join(current_dir, "env_live.py")
        
        shutil.copy2(train_file, os.path.join(log_path, "train_td3_Live_auto_restart.py"))
        shutil.copy2(env_file, os.path.join(log_path, "env_live.py"))
    except Exception as e:
        print(f"[Warning] 备份文件失败: {e}")

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
    
    # 优化：调整 OU 噪声参数，使其更适合车辆控制（更平滑）
    agent.noise.theta = 0.3
    agent.noise.sigma = 0.15
    
    start_episode = 0
    best_episode_reward = -float('inf') 
    min_noise = hyperparams['Min Noise']
    noise_decay = hyperparams['Noise Decay']
    
    # Warm-up 设置
    warmup_episodes = 5 # 前 5 个 Episode 进行随机探索，不训练
    
    # 学习率衰减设置
    lr_decay_patience = 5  # 多少个episode没有进步就衰减学习率
    lr_decay_factor = 0.8  # 衰减因子
    lr_min = 1e-4  # 最小学习率
    recent_rewards = []  # 记录最近的奖励
    no_improvement_count = 0  # 没有进步的计数器
    last_best_reward = -float('inf')  # 上次最佳奖励
    
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"加载预训练模型: {pretrained_model_path}")
        agent.load_model(pretrained_model_path)
        
        try:
            filename = os.path.basename(pretrained_model_path)
            parts = filename.replace('.pt', '').split('_')
            parsed_episode = int(parts[-2])
            parsed_reward = float(parts[-1])
            
            start_episode = parsed_episode + 1
            best_episode_reward = parsed_reward
            print(f"恢复训练: Episode {start_episode}, Best Reward {best_episode_reward}")
            
            noise_scale = max(min_noise, hyperparams['Noise Scale'] * (noise_decay ** start_episode))
        except Exception as e:
            print(f"无法从文件名解析信息，使用默认设置: {e}")
            noise_scale = 0.1 
    else:
        print("从零开始训练")
        noise_scale = hyperparams['Noise Scale']

    print(f"\n========== Start TD3 Training (Auto-Restart every {checkpoint_interval} episodes) ==========")
    
    checkpoint_save_dir = os.path.join(script_dir, "checkpoints")
    os.makedirs(checkpoint_save_dir, exist_ok=True)
    
    should_exit = False
    
    try:
        for episode in range(start_episode, max_episodes):
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
                # Warm-up 阶段使用随机动作
                if episode < start_episode + warmup_episodes:
                    #action = np.random.uniform(-1, 1, size=env.get_action_dim())
                    action = agent.select_action(state, noise_scale=noise_scale)
                else:
                    action = agent.select_action(state, noise_scale=noise_scale)

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

                agent.push(state, action, reward, next_state, done)
                current_episode_memory.append((state, action, reward, next_state, done))
                
                state = next_state
                episode_reward += reward
                
                if done: break
            
            final_speed = final_info.get('vx', 0.0)
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"\033[93m{'='*60}\033[0m")
            print(f"\033[93m[{timestamp}] EPISODE {episode} | Reward: {episode_reward:.0f} | Final Speed: {final_speed:.1f} km/h\033[0m")
            print(f"\033[93m{'='*60}\033[0m")

            # Episode 结束后进行训练
            if episode < start_episode + warmup_episodes:
                print(f"[Warm-up] Episode {episode} 完成，跳过训练 (Buffer Size: {len(agent.buffer)})")
                c_loss, a_loss = 0, 0
            else:
                # 训练次数可以是当前 Episode 步数的一定比例，例如 1/2 或 1/1
                base_train_steps = env.current_step
                # 1. 短 Episode 补偿：如果步数太少，强行多练
                if env.current_step < 10000:
                    base_train_steps = 20000 - env.current_step  # 哪怕只跑了 10 步，也要练 1000 次

                # 2. 低分补偿：如果表现太差，额外加练
                if episode_reward < -100:
                    base_train_steps = int(base_train_steps * 1.5)  # 加练 50%

                # 3. 设置上限（防止训练时间过长，例如不超过 20000 次）
                train_steps = min(base_train_steps, 20000)

                # 4. 设置下限（至少练 100 次）
                train_steps = max(100, train_steps) 
                print(f"Training for {train_steps} steps...")
                
                # 记录训练过程中的 Loss
                episode_visual_history['training_c_loss'] = []
                episode_visual_history['training_a_loss'] = []
                
                # 使用 tqdm 显示训练进度
                pbar = tqdm(range(train_steps), desc=f"Training Episode {episode}", unit="step", leave=False)
                for _ in pbar:
                    c_loss, a_loss, c_grad, a_grad = agent.train_step()
                    
                    # 记录每一步的 Loss
                    episode_visual_history['training_c_loss'].append(float(c_loss))
                    episode_visual_history['training_a_loss'].append(float(a_loss))
                    
                    if c_loss != 0:
                        critic_grads.append(float(c_grad))
                        actor_grads.append(float(a_grad))
                    
                    # 更新进度条后缀信息
                    pbar.set_postfix({'C_Loss': f'{c_loss:.4f}', 'A_Loss': f'{a_loss:.4f}'})

            if (episode) % 1 == 0:
                #log_episode_visuals(writer, episode, episode_visual_history, save_dir=log_path, reward=episode_reward)
                save_episode_data(episode, episode_visual_history, episode_reward, final_info, log_path)
            
            writer.add_scalar('Loss/Critic', c_loss, episode)
            writer.add_scalar('Loss/Actor', a_loss, episode)
            writer.add_scalar('Train/Reward', episode_reward, episode)
            writer.add_scalar('Train/Noise', noise_scale, episode)
            writer.add_scalar('Train/Final_Speed_kmh', final_speed, episode)
            
            if episode_reward > best_episode_reward*0.9 and episode_reward >=0:
                is_elite_display = True
                writer.add_scalar('Train/Is_Elite', 1, episode)
                #log_episode_visuals(writer, episode, episode_visual_history, save_dir=log_path, reward=episode_reward)
                save_episode_data(episode, episode_visual_history, episode_reward, final_info, log_path)
                print(f"[{timestamp}] [精英]! Reward: {episode_reward:.1f}")
                for trans in current_episode_memory:
                    agent.push_elite(*trans)
                if episode_reward > best_episode_reward:
                    best_episode_reward = episode_reward
                    agent.save_model(os.path.join("best_model_save", f"TD3_Carsim_{current_time}_{episode}_{best_episode_reward}.pt"))
                    #log_episode_visuals(writer, episode, episode_visual_history, save_dir=log_path, reward=episode_reward)
                    save_episode_data(episode, episode_visual_history, episode_reward, final_info, log_path)
                    print(f"[{timestamp}] [新纪录] ! Reward: {episode_reward:.1f}")
            else:
                writer.add_scalar('Train/Is_Elite', 0, episode)
                noise_scale = max(min_noise, noise_scale * noise_decay)
            
            # 学习率衰减逻辑
            recent_rewards.append(episode_reward)
            if len(recent_rewards) > 10:
                recent_rewards.pop(0)
            
            if episode_reward > last_best_reward:
                last_best_reward = episode_reward
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            if no_improvement_count >= lr_decay_patience:
                current_lrs = agent.decay_learning_rate(factor=lr_decay_factor, min_lr=lr_min)
                print(f"\033[91m[{timestamp}] [LR衰减] {lr_decay_patience}个episode无进步，学习率降低至 Actor:{current_lrs['actor_lr']:.2e}, Critic:{current_lrs['critic_lr']:.2e}\033[0m")
                no_improvement_count = 0
                last_best_reward = max(recent_rewards) if recent_rewards else episode_reward
            
            current_lrs = agent.get_learning_rates()
            writer.add_scalar('Train/Actor_LR', current_lrs['actor_lr'], episode)
            writer.add_scalar('Train/Critic_LR', current_lrs['critic_lr'], episode)
            
            # 检查是否需要保存checkpoint并退出
            episodes_completed = episode - start_episode + 1
            if episodes_completed >= checkpoint_interval:
                checkpoint_path = os.path.join(checkpoint_save_dir, f"checkpoint_{episode}_{episode_reward:.0f}.pt")
                agent.save_model(checkpoint_path)
                print(f"\n{'='*60}")
                print(f"[CHECKPOINT] 已完成 {episodes_completed} 个episode，保存模型并退出")
                print(f"[CHECKPOINT] 模型已保存到: {checkpoint_path}")
                print(f"[CHECKPOINT] 请重新运行程序继续训练，使用参数:")
                print(f"[CHECKPOINT]   python train_td3_Live_auto_restart.py --model \"{checkpoint_path}\"")
                print(f"{'='*60}\n")
                should_exit = True
                break

    except KeyboardInterrupt:
        print("=====停止训练=====")
        checkpoint_path = os.path.join(checkpoint_save_dir, f"checkpoint_KeyboardInterrupt_{episode}.pt")
        agent.save_model(checkpoint_path)
        print(f"[CHECKPOINT] 模型已保存到: {checkpoint_path}")
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        if not should_exit:
            agent.save_model(os.path.join(log_path, "final_model.pt"))
        print("资源已释放，训练结束。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TD3 Training with Auto-Restart')
    parser.add_argument('--model', type=str, default=None, help='Path to pretrained model')
    parser.add_argument('--checkpoint_interval', type=int, default=200, help='Episodes before auto-save and exit')
    args = parser.parse_args()
    
    setup_seed(42)
    train_td3_Live(
        pretrained_model_path=args.model,
        checkpoint_interval=args.checkpoint_interval
    )
