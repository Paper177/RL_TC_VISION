#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TD3 Vision 测试脚本

加载训练好的多模态模型，在 CarSim 环境中进行确定性推理测试。
无噪声、不训练，纯评估模型表现。
每个 episode 的详细数据保存为 JSON 文件。

用法:
  python test_td3_vision.py --model "checkpoints/checkpoint_xxx.pt"
  python test_td3_vision.py --model "checkpoints/checkpoint_xxx.pt" --episodes 5
"""
import numpy as np
import torch
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from td3_agent import VisionTD3Agent
from env_live_vision import LiveCarsimEnv
import argparse


def save_test_data(episode_num, history, reward, info, save_dir):
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
                episode_data[key] = [
                    float(x.item()) if hasattr(x, 'item') else float(x)
                    for x in value
                ]
            except (ValueError, TypeError, AttributeError):
                pass

    for key, value in info.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            episode_data[f'info_{key}'] = (
                float(value.item()) if hasattr(value, 'item') else float(value)
            )

    save_path = os.path.join(data_dir, f'test_episode_{episode_num}.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(episode_data, f, indent=2, ensure_ascii=False)
    print(f"[Test] Data saved: {save_path}")


def plot_test_episode(episode_num, history, reward, save_dir):
    if not save_dir:
        return
    fig_dir = os.path.join(save_dir, "test_plots")
    os.makedirs(fig_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Test Episode {episode_num} | Reward: {reward:.1f}', fontsize=16)
    steps = range(len(history['velocity']))

    ax = axes[0, 0]
    for key, label in [('T_L1', 'FL'), ('T_R1', 'FR'), ('T_L2', 'RL'), ('T_R2', 'RR')]:
        ax.plot(steps, history[key], label=label, alpha=0.8, linewidth=1)
    ax.set_title('Wheel Torques (Nm)')
    ax.set_ylabel('Torque')
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
    ax.set_ylabel('Slip Ratio')
    ax.set_ylim(-0.05, 0.3)
    ax.legend(loc='upper right', fontsize='small')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(steps, history['velocity'], 'b-', linewidth=2)
    ax.set_title('Vehicle Velocity (km/h)')
    ax.set_ylabel('Speed (km/h)')
    ax.set_ylim(0, 120)
    ax.set_xlabel('Step')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(steps, history['r_total'], 'k-', label='Total', linewidth=2, alpha=0.9)
    for key in history:
        if key.startswith('R_') and len(history[key]) == len(list(steps)):
            ax.plot(steps, history[key], label=key[2:], linestyle='--', alpha=0.7)
    ax.set_title('Reward Composition')
    ax.set_xlabel('Step')
    ax.legend(loc='lower left', fontsize='x-small', ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'test_ep_{episode_num}_{reward:.0f}.png'), dpi=120)
    plt.close(fig)


def test_td3_vision(
    model_path: str,
    num_episodes: int = 1,
    max_torque: float = 500.0,
    target_slip_ratio: float = 0.08,
    log_dir: str = "logs_TD3_Vision_Test",
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

    script_dir = os.path.dirname(os.path.abspath(__file__))
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.abspath(os.path.join(log_dir, f"Test_{current_time}"))
    os.makedirs(log_path, exist_ok=True)

    os.chdir(script_dir)
    print(f"[Test] Working dir: {os.getcwd()}")

    SIMFILE_PATH = os.path.join(script_dir, "simfile.sim")
    VS_DLL_PATH = os.path.join(script_dir, "vs_lv_ds_x64.dll")

    if not os.path.exists(SIMFILE_PATH):
        print(f"[Error] simfile.sim not found: {SIMFILE_PATH}")
        return

    # 测试时 frame_skip=1，看到更细粒度的控制效果
    env = LiveCarsimEnv(
        simfile_path=SIMFILE_PATH,
        vs_dll_path=VS_DLL_PATH,
        sim_time_s=10.0,
        max_torque=max_torque,
        target_slip_ratio=target_slip_ratio,
        reward_weights=reward_weights,
        frame_skip=1,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Test] Device: {device}")

    # 测试不需要大 Buffer，最小化内存占用
    agent = VisionTD3Agent(
        physics_dim=env.get_physics_dim(),
        action_dim=env.get_action_dim(),
        img_shape=env.get_img_shape(),
        action_bound=1.0,
        vision_feat_dim=64,
        hidden_dim=256,
        buffer_capacity=1000,
        elite_capacity=100,
        batch_size=64,
        device=device,
    )

    if not os.path.exists(model_path):
        print(f"[Error] Model not found: {model_path}")
        return

    print(f"[Test] Loading model: {model_path}")
    agent.load_model(model_path)

    print(f"\n{'='*60}")
    print(f"  TD3 Vision Test ({num_episodes} episode(s))")
    print(f"  Model: {os.path.basename(model_path)}")
    print(f"  Max Torque: {max_torque} Nm | Target Slip: {target_slip_ratio}")
    print(f"{'='*60}\n")

    all_rewards = []
    all_speeds = []

    try:
        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0.0

            hist = {
                'T_L1': [], 'T_R1': [], 'T_L2': [], 'T_R2': [],
                'S_L1': [], 'S_R1': [], 'S_L2': [], 'S_R2': [],
                'velocity': [], 'r_total': [], 'target_slip': [],
            }
            final_info = {}

            while True:
                phys = obs['physics']
                img = obs['image']

                # 测试时无噪声，确定性策略
                action = agent.select_action(phys, img, noise_scale=0.0)

                next_obs, reward, done, info = env.step(action)
                final_info = info

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
                        if k not in hist:
                            hist[k] = []
                        hist[k].append(v)

                if done:
                    final_v = info.get('vx', 0.0)
                    r_terminal = final_v * 0.2
                    reward += r_terminal
                    hist['r_total'][-1] = reward

                obs = next_obs
                episode_reward += reward
                if done:
                    break

            final_speed = final_info.get('vx', 0.0)
            all_rewards.append(episode_reward)
            all_speeds.append(final_speed)

            ts = datetime.now().strftime("%H:%M:%S")
            print(f"\033[92m{'='*60}\033[0m")
            print(f"\033[92m[{ts}] TEST EP {episode + 1}/{num_episodes} | "
                  f"Reward: {episode_reward:.1f} | "
                  f"Speed: {final_speed:.1f} km/h\033[0m")
            print(f"\033[92m{'='*60}\033[0m")

            save_test_data(episode, hist, episode_reward, final_info, log_path)
            plot_test_episode(episode, hist, episode_reward, log_path)

    except KeyboardInterrupt:
        print("\n===== Test interrupted =====")
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        env.close()

    # 汇总统计
    if all_rewards:
        print(f"\n{'='*60}")
        print(f"  Test Summary ({len(all_rewards)} episode(s))")
        print(f"  Reward  → Mean: {np.mean(all_rewards):.1f}  "
              f"Min: {np.min(all_rewards):.1f}  Max: {np.max(all_rewards):.1f}")
        print(f"  Speed   → Mean: {np.mean(all_speeds):.1f}  "
              f"Min: {np.min(all_speeds):.1f}  Max: {np.max(all_speeds):.1f} km/h")
        print(f"  Results → {log_path}")
        print(f"{'='*60}")

        summary = {
            'model': model_path,
            'num_episodes': len(all_rewards),
            'rewards': [float(r) for r in all_rewards],
            'final_speeds': [float(s) for s in all_speeds],
            'reward_mean': float(np.mean(all_rewards)),
            'reward_std': float(np.std(all_rewards)),
            'speed_mean': float(np.mean(all_speeds)),
        }
        with open(os.path.join(log_path, "test_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TD3 Vision Model Testing")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model (.pt)")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of test episodes")
    args = parser.parse_args()

    test_td3_vision(
        model_path=args.model,
        num_episodes=args.episodes,
    )
