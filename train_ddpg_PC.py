#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DDPG-PC Training Script
Python-CarSim DLL Link
"""
import numpy as np
import torch
import os
import random
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from dashboard import create_live_monitor
from ddpg_agent import DDPGAgent
from env_pc import PythonCarsimEnv 
import gc

#================= 绘图与日志 =================
def log_episode_visuals(writer, episode_num, history, save_dir=None):
    """
    绘制本回合的详细曲线并上传到 TensorBoard，同时保存到本地
    """
    # 显式清理内存，防止长时间运行导致内存泄漏
    plt.close('all')
    gc.collect()
    # 创建一个 2x2 的画布
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Detailed Analysis', fontsize=16)
    
    steps = range(len(history['velocity']))
    
    # --- 图1: 四轮输出扭矩 (Action) ---
    ax = axes[0, 0]
    ax.plot(steps, history['T_L1'], label='L1 (FL)', alpha=0.8, linewidth=1)
    ax.plot(steps, history['T_R1'], label='R1 (FR)', alpha=0.8, linewidth=1)
    ax.plot(steps, history['T_L2'], label='L2 (RL)', alpha=0.8, linewidth=1)
    ax.plot(steps, history['T_R2'], label='R2 (RR)', alpha=0.8, linewidth=1)
    ax.set_title('Wheel Torques (Nm)')
    ax.set_ylabel('Torque')
    ax.legend(loc='upper right', fontsize='small')
    ax.grid(True, alpha=0.3)

    # --- 图2: 四轮滑移率 (State) ---
    ax = axes[0, 1]
    ax.plot(steps, history['S_L1'], label='L1', alpha=0.8)
    ax.plot(steps, history['S_R1'], label='R1', alpha=0.8)
    ax.plot(steps, history['S_L2'], label='L2', alpha=0.8)
    ax.plot(steps, history['S_R2'], label='R2', alpha=0.8)
    
    # 绘制目标滑移率参考线 (红线)
    if 'target_slip' in history and len(history['target_slip']) > 0:
        target_val = history['target_slip'][0]
        ax.axhline(y=target_val, color='r', linestyle='--', alpha=0.5, label=f'Target ({target_val})')
        
    ax.set_title('Slip Ratios')
    ax.set_ylabel('Slip Ratio')
    ax.set_ylim(-0.05, 0.2)
    ax.legend(loc='upper right', fontsize='small')
    ax.grid(True, alpha=0.3)

    # --- 图3: 速度变化 (Velocity) ---
    ax = axes[1, 0]
    ax.plot(steps, history['velocity'], 'b-', label='Actual Speed', linewidth=2)
    ax.set_title('Vehicle Velocity (km/h)')
    ax.set_ylabel('Speed (km/h)')
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 10))
    ax.set_xlabel('Step')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 图4: 奖励分析  ---
    ax = axes[1, 1]
    ax.plot(steps, history['r_total'], 'k-', label='Total Reward', linewidth=2, alpha=0.9)
    
    # 绘制 env_pc.py 中定义的 R_Spd, R_Slp, R_Eng 等
    for key in history:
        if key.startswith('R_') and len(history[key]) == len(steps):
            ax.plot(steps, history[key], label=key[2:], linestyle='--', alpha=0.7) 
    
    ax.set_title('Reward Composition')
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward Value')
    ax.legend(loc='lower left', fontsize='x-small', ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # 保存到本地文件
    if save_dir:
        # 确保保存目录存在
        figures_dir = os.path.join(save_dir, "episode_plots")
        os.makedirs(figures_dir, exist_ok=True)
        save_path = os.path.join(figures_dir, f'episode_{episode_num}_analysis.png')
        plt.savefig(save_path)
        
    plt.close(fig)

#================= 固定训练种子 =================
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"随机种子已锁定为: {seed}")

#================= 训练主函数 =================
def train_ddpg_PC(
    max_episodes: int = 10000,
    max_torque: float = 1500.0,
    target_slip_ratio: float = 0.04, # 与 env_pc 默认值保持一致
    log_dir: str = "logs_PC",
    pretrained_model_path: str = None 
):
    # --- 配置 ---
    reward_weights = {
        'w_speed': 0.00,
        'w_accel': 0.06,
        'w_energy': 0.015,
        'w_consistency': -0.025,
        'w_beta': -0.0,
        'w_slip': -0.35,
        'w_smooth': -0.0
    }
    
    hyperparams = {
        'Action Bound': 1.0,   
        'Hidden Dim': 256,
        'Actor LR': 1e-5,      
        'Critic LR': 1e-4,
        'Buffer Capacity': 1000000,
        'Batch Size': 1024,
        'Elite Ratio': 0.4,    
        'Elite Capacity': 200000,
        'Noise Scale': 0.5,    
        'Min Noise': 0.05,
        'Noise Decay': 0.998,  
    }
    
    # --- 日志初始化 ---
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"Python_Carsim_{current_time}")
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_dir=log_path)

    # --- 备份训练脚本和环境文件 ---
    import shutil
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        train_file = os.path.join(current_dir, "train_ddpg_PC.py")
        env_file = os.path.join(current_dir, "env_pc.py")
        
        shutil.copy2(train_file, os.path.join(log_path, "train_ddpg_PC.py"))
        shutil.copy2(env_file, os.path.join(log_path, "env_pc.py"))
        print(f"[Info] 已备份 train_ddpg_PC.py 和 env_pc.py 到 {log_path}")
    except Exception as e:
        print(f"[Warning] 备份文件失败: {e}")
    # --- Tensorboard存储参数 ---
    
    md_table = "### Reward Coefficients\n| Key | Value |\n|---|---|\n"
    for k, v in reward_weights.items():
        md_table += f"| {k} | {v} |\n"
    writer.add_text("Configuration", md_table, 0)

    md_table = "\n### Hyperparameters\n| Key | Value |\n|---|---|\n"
    for k, v in hyperparams.items():
        md_table += f"| {k} | {v} |\n"
    writer.add_text("Configuration/Parameters", md_table, 0)

    print(f"训练日志: {log_path}")

    # --- 环境初始化 ---
    CARSIM_DB_DIR = r"E:\CarSim2022\CarSim2022.1_Prog\RL" 
    
    if not os.path.exists(CARSIM_DB_DIR):
        print(f"找不到 CarSim 数据库路径: {CARSIM_DB_DIR}")
        return

    env = PythonCarsimEnv(
        carsim_db_dir=CARSIM_DB_DIR,
        sim_time_s=10.0,       
        delta_time_s=0.01,
        max_torque=max_torque,
        target_slip_ratio=target_slip_ratio,
        target_speed=100,
        vehicle_type="normal_vehicle",
        reward_weights=reward_weights
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
    # --- 加载预训练模型 ---
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"加载预训练模型: {pretrained_model_path}")
        agent.load_model(pretrained_model_path)
        noise_scale = 0.1 
    else:
        print("从零开始训练")
        noise_scale = hyperparams['Noise Scale']

    best_episode_reward = -float('inf') 
    min_noise = hyperparams['Min Noise']
    noise_decay = hyperparams['Noise Decay']

    print("\n========== Start Training ==========")
    live_display, dashboard = create_live_monitor()
    
    try:
        with live_display:
            for episode in range(max_episodes):
                # Reset
                state, info = env.reset()
                agent.reset_noise() 
                episode_reward = 0

                # 数据容器
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
                    
                    # --- 收集绘图数据 ---
                    
                    # 扭矩 (从 info 中获取 env 实际施加的扭矩)
                    episode_visual_history['T_L1'].append(info.get('trq_L1', 0))
                    episode_visual_history['T_R1'].append(info.get('trq_R1', 0))
                    episode_visual_history['T_L2'].append(info.get('trq_L2', 0))
                    episode_visual_history['T_R2'].append(info.get('trq_R2', 0))
                    
                    # 滑移率 (对应 env_pc.py 的 keys: slip_L1, slip_R1...)
                    episode_visual_history['S_L1'].append(info.get('slip_L1', 0))
                    episode_visual_history['S_R1'].append(info.get('slip_R1', 0))
                    episode_visual_history['S_L2'].append(info.get('slip_L2', 0))
                    episode_visual_history['S_R2'].append(info.get('slip_R2', 0))
                    episode_visual_history['target_slip'].append(target_slip_ratio)
                    
                    # 速度
                    episode_visual_history['velocity'].append(info.get('vx', 0))
                    
                    # 总奖励
                    episode_visual_history['r_total'].append(reward)
                    
                    # 动态记录奖励分项 (R_Spd, R_Slp, R_Eng)
                    for k, v in info.items():
                        if k.startswith("R_"):
                            if k not in reward_stats: reward_stats[k] = []
                            reward_stats[k].append(v)
                            if k not in episode_visual_history: episode_visual_history[k] = []
                            episode_visual_history[k].append(v)

                    # ============================== =========

                    # --- 末速度奖励 ---
                    if done:
                        final_v = info.get('vx', 0.0)
                        r_terminal = final_v * 0.2 
                        reward += r_terminal
                        if 'R_Terminal' not in reward_stats: reward_stats['R_Terminal'] = []
                        reward_stats['R_Terminal'].append(r_terminal)
                        episode_visual_history['r_total'][-1] = reward
                    # -------------------------

                    # Push & Train
                    agent.push(state, action, reward, next_state, done)
                    current_episode_memory.append((state, action, reward, next_state, done))
                    
                    # DDPG
                    c_loss, a_loss, c_grad, a_grad = agent.train_step()
                    
                    if c_loss != 0:
                        critic_grads.append(c_grad)
                        actor_grads.append(a_grad)
                    
                    state = next_state
                    episode_reward += reward
                    
                    # 实时更新看板
                    if env.current_step % 10 == 0 or done:
                        dashboard.update(
                            episode=episode+1,
                            step=env.current_step,
                            info=info,
                            reward=reward,
                            noise=noise_scale,
                            elite_flag=is_elite_display
                        )
                    
                    if done: break
                
                # --- Episode End ---
                
                # 绘图每10回合画一次
                if (episode) % 10 == 0:
                    log_episode_visuals(writer, episode, episode_visual_history, save_dir=log_path)
                
                # Tensorboard记录处理
                sum_rewards = {k: np.sum(v) for k, v in reward_stats.items()}
                avg_c = np.mean(critic_grads) if critic_grads else 0
                avg_a = np.mean(actor_grads) if actor_grads else 0
                final_speed = final_info.get('vx', 0.0)
                writer.add_scalar('Loss/Critic', c_loss, episode)
                writer.add_scalar('Loss/Actor', a_loss, episode)
                writer.add_scalar('Train/Reward', episode_reward, episode)
                writer.add_scalar('Train/Noise', noise_scale, episode)
                writer.add_scalar('Train/Final_Speed_kmh', final_speed, episode)
                for key, val in sum_rewards.items():
                    writer.add_scalar(f'Reward_Details/{key}', val, episode)
                
                writer.add_scalar('Grad/Critic', avg_c, episode)
                writer.add_scalar('Grad/Actor', avg_a, episode)

                details_str = " | ".join([f"{k[2:]}: {v:.0f}" for k, v in sum_rewards.items()])
                dashboard.log(f"Ep {episode}| Rw: {episode_reward:.0f} | EndV: {final_speed:.1f} | "
                  f"{details_str} | "
                  f"Grad: {avg_c:.3f}/{avg_a:.3f}")
                
                # Elite Logic
                if episode_reward > best_episode_reward*0.9 and episode_reward >=0:
                    is_elite_display = True
                    writer.add_scalar('Train/Is_Elite', 1, episode)
                    dashboard.log(f"[精英]! Reward: {episode_reward:.1f}")
                    for trans in current_episode_memory:
                        agent.push_elite(*trans)
                    if episode_reward > best_episode_reward:
                        best_episode_reward = episode_reward
                        agent.save_model(os.path.join("best_model_save", f"Python_Carsim_{current_time}_{episode}_{best_episode_reward}.pt"))
                        log_episode_visuals(writer, episode, episode_visual_history, save_dir=log_path)
                        dashboard.log(f"[新纪录] ! Reward: {episode_reward:.1f}")
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
    train_ddpg_PC(
        #pretrained_model_path="best_model_save/Python_Carsim_20251223_173827_482_252.94361713080013.pt"
        #pretrained_model_path="best_model_save/Python_Carsim_20251217_144021.pt"
    )