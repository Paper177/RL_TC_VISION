#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TD3-Live Training Script with End-to-End Vision

训练流程:
  1. 环境返回多模态观测 {physics, image}
  2. Agent 的 Actor/Critic 内部 CNN 端到端训练
  3. Replay Buffer 直接存储 uint8 图像
  4. 支持自动保存 checkpoint 并退出, 方便恢复训练
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
from torch.utils.tensorboard import SummaryWriter
from td3_agent import VisionTD3Agent
from env_live_vision import LiveCarsimEnv
import gc
import shutil
import argparse
from tqdm import tqdm


# ================= 日志与绘图 =================

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
                pass

    for key, value in info.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            episode_data[f'info_{key}'] = float(value) if hasattr(value, 'item') else value

    save_path = os.path.join(data_dir, f'episode_{episode_num}.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(episode_data, f, indent=2, ensure_ascii=False)


def log_episode_visuals(writer, episode_num, history, save_dir=None, reward=None):
    plt.close('all')
    gc.collect()
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Episode {episode_num} | Reward: {reward:.1f}', fontsize=16)

    steps = range(len(history['velocity']))

    ax = axes[0, 0]
    for key, label in [('T_L1', 'FL'), ('T_R1', 'FR'), ('T_L2', 'RL'), ('T_R2', 'RR')]:
        ax.plot(steps, history[key], label=label, alpha=0.8, linewidth=1)
    ax.set_title('Wheel Torques (Nm)')
    ax.legend(loc='upper right', fontsize='small')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    window = 10
    for key, label in [('S_L1', 'FL'), ('S_R1', 'FR'), ('S_L2', 'RL'), ('S_R2', 'RR')]:
        data = history[key]
        smoothed = [np.mean(data[max(0, i - window // 2):i + window // 2 + 1])
                    for i in range(len(data))]
        ax.plot(steps, smoothed, label=label, alpha=0.8, linewidth=1.5)
    if history.get('target_slip'):
        ax.axhline(y=history['target_slip'][0], color='r', linestyle='--',
                    alpha=0.5, label='Target')
    ax.set_title('Slip Ratios (Smoothed)')
    ax.set_ylim(-0.05, 0.2)
    ax.legend(loc='upper right', fontsize='small')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(steps, history['velocity'], 'b-', linewidth=2)
    ax.set_title('Vehicle Velocity (km/h)')
    ax.set_ylim(0, 100)
    ax.set_xlabel('Step')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(steps, history['r_total'], 'k-', label='Total', linewidth=2, alpha=0.9)
    for key in history:
        if key.startswith('R_') and len(history[key]) == len(list(steps)):
            ax.plot(steps, history[key], label=key[2:], linestyle='--', alpha=0.7)
    ax.set_title('Reward Composition')
    ax.set_xlabel('Step')
    ax.set_ylim(-0.05, 0.1)
    ax.legend(loc='lower left', fontsize='x-small', ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_dir:
        fig_dir = os.path.join(save_dir, "episode_plots")
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(os.path.join(fig_dir, f'ep_{episode_num}_{reward:.0f}.png'))
    plt.close(fig)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"[Seed] Random seed locked to: {seed}")


# ================= 训练主函数 =================

def train_td3_vision(
    max_episodes: int = 10000,
    max_torque: float = 500.0,
    target_slip_ratio: float = 0.08,
    log_dir: str = "logs_TD3_Vision",
    pretrained_model_path: str = None,
    checkpoint_interval: int = 300,
):
    reward_weights = {
        'w_speed': 0.00,
        'w_accel': 0.15,
        'w_energy': 0.015,
        'w_consistency': -0.05,
        'w_beta': -0.02,
        'w_slip': -0.2,
        'w_smooth': -0.00,
        'w_yaw': -2.0,
    }

    hyperparams = {
        'Action Bound': 1.0,
        'Hidden Dim': 256,
        'Vision Feat Dim': 64,
        'gamma': 0.99,
        'Actor LR': 3e-4,
        'Critic LR': 3e-4,
        'Buffer Capacity': 300000,
        'Batch Size': 128,
        'Elite Ratio': 0.3,
        'Elite Capacity': 100000,
        'Noise Scale': 0.15,
        'Min Noise': 0.02,
        'Noise Decay': 0.995,
        'Policy Noise': 0.1,
        'Noise Clip': 0.25,
        'Policy Freq': 2,
    }

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))

    log_path = os.path.abspath(os.path.join(log_dir, f"TD3_Vision_{current_time}"))
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_dir=log_path)

    # 备份训练代码
    try:
        for fname in ["train_td3_Live_vision.py", "env_live_vision.py",
                       "networks.py", "td3_agent.py", "replay_buffer.py"]:
            src = os.path.join(script_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(log_path, fname))
    except Exception as e:
        print(f"[Warning] 备份文件失败: {e}")

    # 保存超参数
    with open(os.path.join(log_path, "hyperparams.json"), 'w') as f:
        json.dump({**hyperparams, **reward_weights}, f, indent=2)

    os.chdir(script_dir)
    print(f"[Train] Working dir: {os.getcwd()}")

    SIMFILE_PATH = os.path.join(script_dir, "simfile.sim")
    VS_DLL_PATH = os.path.join(script_dir, "vs_lv_ds_x64.dll")

    if not os.path.exists(SIMFILE_PATH):
        print(f"[Error] simfile.sim not found: {SIMFILE_PATH}")
        return

    # ---- 创建环境 ----
    env = LiveCarsimEnv(
        simfile_path=SIMFILE_PATH,
        vs_dll_path=VS_DLL_PATH,
        sim_time_s=10.0,
        max_torque=max_torque,
        target_slip_ratio=target_slip_ratio,
        reward_weights=reward_weights,
        frame_skip=5,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Device: {device}")

    # ---- 创建智能体 ----
    agent = VisionTD3Agent(
        physics_dim=env.get_physics_dim(),
        action_dim=env.get_action_dim(),
        img_shape=env.get_img_shape(),
        action_bound=hyperparams['Action Bound'],
        vision_feat_dim=hyperparams['Vision Feat Dim'],
        hidden_dim=hyperparams['Hidden Dim'],
        actor_lr=hyperparams['Actor LR'],
        critic_lr=hyperparams['Critic LR'],
        buffer_capacity=hyperparams['Buffer Capacity'],
        elite_capacity=hyperparams['Elite Capacity'],
        batch_size=hyperparams['Batch Size'],
        gamma=hyperparams['gamma'],
        policy_noise=hyperparams['Policy Noise'],
        noise_clip=hyperparams['Noise Clip'],
        policy_freq=hyperparams['Policy Freq'],
        elite_ratio=hyperparams['Elite Ratio'],
        device=device,
    )

    agent.noise.theta = 0.3
    agent.noise.sigma = 0.15

    start_episode = 0
    best_episode_reward = -float('inf')
    noise_scale = hyperparams['Noise Scale']
    min_noise = hyperparams['Min Noise']
    noise_decay = hyperparams['Noise Decay']
    warmup_episodes = 5

    # 学习率衰减
    lr_decay_patience = 5
    lr_decay_factor = 0.5
    lr_min = 1e-6
    recent_rewards = []
    no_improvement_count = 0
    last_best_reward = -float('inf')

    # 加载预训练模型
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"[Train] Loading pretrained: {pretrained_model_path}")
        agent.load_model(pretrained_model_path)
        try:
            filename = os.path.basename(pretrained_model_path)
            parts = filename.replace('.pt', '').split('_')
            parsed_episode = int(parts[-2])
            parsed_reward = float(parts[-1])
            start_episode = parsed_episode + 1
            best_episode_reward = parsed_reward
            noise_scale = max(min_noise, hyperparams['Noise Scale'] * (noise_decay ** start_episode))
            print(f"[Train] Resuming: Episode {start_episode}, Best {best_episode_reward:.1f}")
        except Exception as e:
            print(f"[Train] Could not parse filename, using defaults: {e}")
            noise_scale = 0.1
    else:
        print("[Train] Training from scratch")

    print(f"\n{'='*60}")
    print(f"  TD3 Vision Training (auto-restart every {checkpoint_interval} eps)")
    print(f"  Buffer memory: ~{agent.buffer.get_memory_mb():.0f} MB (normal) "
          f"+ ~{agent.elite_buffer.get_memory_mb():.0f} MB (elite)")
    print(f"{'='*60}\n")

    checkpoint_dir = os.path.join(script_dir, "checkpoints")
    best_model_dir = os.path.join(script_dir, "best_model_save")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    should_exit = False

    try:
        for episode in range(start_episode, max_episodes):
            obs, _ = env.reset()
            agent.reset_noise()
            episode_reward = 0.0

            reward_stats = {}
            episode_memory = []
            hist = {
                'T_L1': [], 'T_R1': [], 'T_L2': [], 'T_R2': [],
                'S_L1': [], 'S_R1': [], 'S_L2': [], 'S_R2': [],
                'velocity': [], 'r_total': [], 'target_slip': [],
            }
            final_info = {}

            while True:
                phys = obs['physics']
                img = obs['image']

                # Warmup: 随机探索填充 buffer
                if episode < start_episode + warmup_episodes:
                    action = np.random.uniform(0.0, 1.0, size=env.get_action_dim())
                else:
                    action = agent.select_action(phys, img, noise_scale=noise_scale)

                next_obs, reward, done, info = env.step(action)
                final_info = info

                # 记录历史
                hist['T_L1'].append(info.get('trq_L1', 0))
                hist['T_R1'].append(info.get('trq_R1', 0))
                hist['T_L2'].append(info.get('trq_L2', 0))
                hist['T_R2'].append(info.get('trq_R2', 0))
                hist['S_L1'].append(info.get('slip_L1', 0))
                hist['S_R1'].append(info.get('slip_R1', 0))
                hist['S_L2'].append(info.get('slip_L2', 0))
                hist['S_R2'].append(info.get('slip_R2', 0))
                hist['velocity'].append(info.get('vx', 0))
                hist['target_slip'].append(target_slip_ratio)
                hist['r_total'].append(reward)

                for k, v in info.items():
                    if k.startswith("R_"):
                        if k not in reward_stats:
                            reward_stats[k] = []
                        reward_stats[k].append(v)
                        if k not in hist:
                            hist[k] = []
                        hist[k].append(v)

                # 终止奖励: 鼓励高终速
                if done:
                    final_v = info.get('vx', 0.0)
                    r_terminal = final_v * 0.2
                    reward += r_terminal
                    hist['r_total'][-1] = reward

                next_phys = next_obs['physics']
                next_img = next_obs['image']

                agent.push(phys, img, action, reward,
                           next_phys, next_img, done)
                episode_memory.append((phys, img, action, reward,
                                       next_phys, next_img, done))

                obs = next_obs
                episode_reward += reward
                if done:
                    break

            # ---- Episode 结束 ----
            final_speed = final_info.get('vx', 0.0)
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"\033[93m{'='*60}\033[0m")
            print(f"\033[93m[{ts}] EP {episode} | Reward: {episode_reward:.0f} "
                  f"| Speed: {final_speed:.1f} km/h | Buffer: {len(agent.buffer)}\033[0m")
            print(f"\033[93m{'='*60}\033[0m")

            # ---- 训练 ----
            c_loss, a_loss = 0.0, 0.0
            if episode < start_episode + warmup_episodes:
                print(f"[Warmup] Episode {episode}, skip training "
                      f"(Buffer: {len(agent.buffer)})")
            else:
                base_steps = env.current_step
                if env.current_step < 10000:
                    base_steps = 20000 - env.current_step
                if episode_reward < -100:
                    base_steps = int(base_steps * 1.5)
                train_steps = max(100, min(base_steps, 20000))

                print(f"[Train] {train_steps} steps...")
                pbar = tqdm(range(train_steps),
                            desc=f"Training EP {episode}", unit="step", leave=False)
                for _ in pbar:
                    c_loss, a_loss, c_grad, a_grad = agent.train_step()
                    pbar.set_postfix({'C': f'{c_loss:.4f}', 'A': f'{a_loss:.4f}'})

            # ---- Logging ----
            save_episode_data(episode, hist, episode_reward, final_info, log_path)

            writer.add_scalar('Loss/Critic', c_loss, episode)
            writer.add_scalar('Loss/Actor', a_loss, episode)
            writer.add_scalar('Train/Reward', episode_reward, episode)
            writer.add_scalar('Train/Noise', noise_scale, episode)
            writer.add_scalar('Train/Final_Speed_kmh', final_speed, episode)
            writer.add_scalar('Train/Buffer_Size', len(agent.buffer), episode)

            # ---- Elite 判定 ----
            if episode_reward > best_episode_reward * 0.9 and episode_reward >= 0:
                writer.add_scalar('Train/Is_Elite', 1, episode)
                print(f"[{ts}] [Elite] Reward: {episode_reward:.1f}")
                for trans in episode_memory:
                    agent.push_elite(*trans)

                if episode_reward > best_episode_reward:
                    best_episode_reward = episode_reward
                    model_name = (f"TD3_Vision_{current_time}"
                                  f"_{episode}_{best_episode_reward:.0f}.pt")
                    agent.save_model(os.path.join(best_model_dir, model_name))
                    print(f"[{ts}] [New Best] Reward: {episode_reward:.1f}")
            else:
                writer.add_scalar('Train/Is_Elite', 0, episode)
                noise_scale = max(min_noise, noise_scale * noise_decay)

            # ---- 学习率衰减 ----
            recent_rewards.append(episode_reward)
            if len(recent_rewards) > 10:
                recent_rewards.pop(0)

            if episode_reward > last_best_reward:
                last_best_reward = episode_reward
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= lr_decay_patience:
                lrs = agent.decay_learning_rate(factor=lr_decay_factor, min_lr=lr_min)
                print(f"\033[91m[{ts}] [LR Decay] Actor: {lrs['actor_lr']:.2e}, "
                      f"Critic: {lrs['critic_lr']:.2e}\033[0m")
                no_improvement_count = 0
                last_best_reward = max(recent_rewards) if recent_rewards else episode_reward

            lrs = agent.get_learning_rates()
            writer.add_scalar('Train/Actor_LR', lrs['actor_lr'], episode)
            writer.add_scalar('Train/Critic_LR', lrs['critic_lr'], episode)

            # ---- Checkpoint 自动退出 ----
            episodes_completed = episode - start_episode + 1
            if episodes_completed >= checkpoint_interval:
                ckpt_path = os.path.join(
                    checkpoint_dir,
                    f"checkpoint_{episode}_{episode_reward:.0f}.pt")
                agent.save_model(ckpt_path)

                buffer_path = ckpt_path.replace('.pt', '_buffer.pkl')
                print(f"[CHECKPOINT] 正在保存 Buffer 到: {buffer_path} ...")
                agent.save_buffer(buffer_path)

                print(f"\n{'='*60}")
                print(f"[CHECKPOINT] 已完成 {episodes_completed} 个 episode，保存模型并退出")
                print(f"[CHECKPOINT] 模型: {ckpt_path}")
                print(f"[CHECKPOINT] Buffer: {buffer_path}")
                print(f"[CHECKPOINT] 恢复训练:")
                print(f'  python train_td3_Live_vision.py --model "{ckpt_path}"')
                print(f"{'='*60}\n")
                should_exit = True
                break

    except KeyboardInterrupt:
        print("\n===== Training interrupted =====")
        ckpt_path = os.path.join(
            checkpoint_dir,
            f"checkpoint_KeyboardInterrupt_{episode}.pt")
        agent.save_model(ckpt_path)

        buffer_path = ckpt_path.replace('.pt', '_buffer.pkl')
        print(f"[CHECKPOINT] 正在保存 Buffer 到: {buffer_path} ...")
        agent.save_buffer(buffer_path)

        print(f"[CHECKPOINT] 模型: {ckpt_path}")
        print(f"[CHECKPOINT] Buffer: {buffer_path}")
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        if not should_exit:
            agent.save_model(os.path.join(log_path, "final_model.pt"))
        writer.close()
        print("[Train] Resources released, training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='TD3 Vision Training (End-to-End, Auto-Restart)')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to pretrained model checkpoint')
    parser.add_argument('--checkpoint_interval', type=int, default=200,
                        help='Episodes before auto-save and exit')
    args = parser.parse_args()

    setup_seed(42)
    train_td3_vision(
        pretrained_model_path=args.model,
        checkpoint_interval=args.checkpoint_interval,
    )
