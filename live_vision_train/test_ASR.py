#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ASR扭矩表验证程序
从CSV文件读取预计算扭矩，直接控制CarSim仿真，保存JSON并绘制图表

使用方式:
  python test_ASR.py --torque_dir "ASR_TorqueMap" --max_torque 1000
"""
import numpy as np
import torch
import os
import json
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from env_live_vision import LiveCarsimEnv


def load_torque_maps(torque_dir, max_torque=1000.0):
    """
    从CSV文件加载四轮扭矩表
    
    期望文件:
      - FLmap.csv (左前轮)
      - FRmap.csv (右前轮)  
      - RLmap.csv (左后轮)
      - RRmap.csv (右后轮)
    
    CSV格式: 单列扭矩值 (Nm)，每行一个，共10000行 (10s @ 0.001s)
    """
    wheel_names = ['FL', 'FR', 'RL', 'RR']
    file_names = ['FLmap.csv', 'FRmap.csv', 'RLmap.csv', 'RRmap.csv']
    torque_maps = {}
    
    for wheel, fname in zip(wheel_names, file_names):
        fpath = os.path.join(torque_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"扭矩表文件不存在: {fpath}")
        
        # 读取CSV (单列数据)
        values = []
        with open(fpath, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:  # 跳过空行
                    values.append(float(row[0]))
        
        torque_array = np.array(values, dtype=np.float32)
        print(f"[Load] {wheel}: {len(torque_array)} points, "
              f"range [{torque_array.min():.1f}, {torque_array.max():.1f}] Nm")
        
        # 归一化到 [0, 1] 范围 (与RL agent输出一致)
        torque_normalized = np.clip(torque_array / max_torque, 0.0, 1.0)
        torque_maps[wheel] = torque_normalized
    
    return torque_maps


def save_episode_data(episode_num, history, reward, info, save_dir):
    """保存 episode 数据到 JSON (保持与训练脚本一致)"""
    if not save_dir:
        return
    data_dir = os.path.join(save_dir, "episode_data")
    os.makedirs(data_dir, exist_ok=True)

    episode_data = {
        'episode': episode_num,
        'total_reward': float(reward),
        'num_steps': len(history.get('velocity', [])),
        'control_source': 'ASR_torque_map',  # 标记控制来源
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
    
    print(f"[Save] JSON saved: {save_path}")


def plot_episode_data(history, save_dir, episode_num):
    """绘制 episode 数据图表"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f'ASR Torque Map Validation - Episode {episode_num}', fontsize=14)
    
    time = np.arange(len(history['velocity'])) * 0.001  # 时间轴 (s)
    
    # 1. 四轮扭矩
    ax = axes[0, 0]
    ax.plot(time, history['T_L1'], label='FL (T_L1)', linewidth=1)
    ax.plot(time, history['T_R1'], label='FR (T_R1)', linewidth=1)
    ax.plot(time, history['T_L2'], label='RL (T_L2)', linewidth=1)
    ax.plot(time, history['T_R2'], label='RR (T_R2)', linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Torque (Nm)')
    ax.set_title('Wheel Torque')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 10)
    
    # 2. 四轮滑移率
    ax = axes[0, 1]
    ax.plot(time, history['S_L1'], label='FL (S_L1)', linewidth=1)
    ax.plot(time, history['S_R1'], label='FR (S_R1)', linewidth=1)
    ax.plot(time, history['S_L2'], label='RL (S_L2)', linewidth=1)
    ax.plot(time, history['S_R2'], label='RR (S_R2)', linewidth=1)
    if 'target_slip' in history and len(history['target_slip']) > 0:
        ax.axhline(y=history['target_slip'][0], color='r', linestyle='--', 
                   label='Target', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Slip Ratio')
    ax.set_title('Wheel Slip Ratio')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 10)
    
    # 3. 车速与奖励
    ax = axes[1, 0]
    ax.plot(time, np.array(history['velocity']) * 3.6, 'g-', linewidth=1.5, label='Speed')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (km/h)', color='g')
    ax.tick_params(axis='y', labelcolor='g')
    ax.set_xlim(0, 10)
    ax.grid(True, alpha=0.3)
    
    ax2 = ax.twinx()
    ax2.plot(time, history['r_total'], 'b-', linewidth=1, alpha=0.6, label='Reward')
    ax2.set_ylabel('Reward', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    ax.set_title('Speed & Reward')
    
    # 4. 各奖励分量 (如果有)
    ax = axes[1, 1]
    reward_keys = [k for k in history.keys() if k.startswith('R_')]
    if reward_keys:
        for key in reward_keys[:4]:  # 最多显示4个
            short_name = key.replace('R_', '')
            ax.plot(time, history[key], label=short_name, linewidth=1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Reward Component')
        ax.set_title('Reward Components')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 10)
    else:
        ax.text(0.5, 0.5, 'No reward components recorded', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Reward Components')
    
    # 5. 转向角
    ax = axes[2, 0]
    if 'steering' in history and len(history['steering']) > 0:
        ax.plot(time, history['steering'], 'm-', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Steering Angle (deg)')
        ax.set_title('Steering Input')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 10)
    else:
        ax.text(0.5, 0.5, 'No steering data', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Steering Input')
    
    # 6. 横向误差
    ax = axes[2, 1]
    if 'lateral_error' in history and len(history['lateral_error']) > 0:
        ax.plot(time, history['lateral_error'], 'c-', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Lateral Error (m)')
        ax.set_title('Lateral Error')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 10)
    else:
        ax.text(0.5, 0.5, 'No lateral error data', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Lateral Error')
    
    plt.tight_layout()
    
    # 保存图表
    plot_dir = os.path.join(save_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f'episode_{episode_num}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"[Plot] Figure saved: {plot_path}")


def run_asr_validation(
    torque_dir: str = "ASR_TorqueMap",
    max_torque: float = 1000.0,
    target_slip_ratio: float = 0.08,
    log_dir: str = "logs_ASR_Validation",
):
    """
    主函数: 加载扭矩表并运行CarSim仿真验证
    """
    print(f"\n{'='*60}")
    print(f"  ASR Torque Map Validation")
    print(f"  Max Torque: {max_torque} Nm | Target Slip: {target_slip_ratio}")
    print(f"{'='*60}\n")
    
    # 奖励函数权重 (用于计算奖励)
    reward_weights = {
        'w_speed': 0.00,
        'w_accel': 0.15,
        'w_energy': 0.015,
        'w_consistency': -0.05,
        'w_beta': -0.02,
        'w_slip': -0.25,
        'w_smooth': -0.1,
        'w_yaw': -2.0,
    }
    
    # 日志路径
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.abspath(os.path.join(log_dir, f"ASR_Validation_{current_time}"))
    os.makedirs(log_path, exist_ok=True)
    print(f"[Log] Output directory: {log_path}")
    
    # 保存配置
    config = {
        'max_torque': max_torque,
        'target_slip_ratio': target_slip_ratio,
        'reward_weights': reward_weights,
        'torque_dir': torque_dir,
        'sim_time': 10.0,
        'time_step': 0.001,
    }
    with open(os.path.join(log_path, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    # 加载扭矩表
    print(f"\n[Load] Loading torque maps from: {torque_dir}")
    torque_dir_full = os.path.join(script_dir, torque_dir)
    torque_maps = load_torque_maps(torque_dir_full, max_torque)
    
    os.chdir(script_dir)
    print(f"\n[Sim] Working dir: {os.getcwd()}")
    
    SIMFILE_PATH = os.path.join(script_dir, "simfile.sim")
    VS_DLL_PATH = os.path.join(script_dir, "vs_lv_ds_x64.dll")
    
    if not os.path.exists(SIMFILE_PATH):
        print(f"[Error] simfile.sim not found: {SIMFILE_PATH}")
        return
    
    # 创建环境
    env = LiveCarsimEnv(
        simfile_path=SIMFILE_PATH,
        vs_dll_path=VS_DLL_PATH,
        sim_time_s=10.0,
        max_torque=max_torque,
        target_slip_ratio=target_slip_ratio,
        reward_weights=reward_weights,
        frame_skip=1,  # ASR需要每个时间步都控制
    )
    
    print(f"\n{'='*60}")
    print(f"  Starting ASR Validation Simulation")
    print(f"{'='*60}\n")
    
    try:
        obs, _ = env.reset()
        episode_reward = 0.0
        step = 0
        final_info = {}
        
        # 历史数据记录
        hist = {
            'T_L1': [], 'T_R1': [], 'T_L2': [], 'T_R2': [],
            'S_L1': [], 'S_R1': [], 'S_L2': [], 'S_R2': [],
            'velocity': [], 'r_total': [], 'target_slip': [],
            'steering': [], 'lateral_error': [],
        }
        reward_stats = {}
        
        while True:
            # 从扭矩表获取当前步的控制信号
            # 动作格式: [ steering (保持0), FL, FR, RL, RR ]
            if step < len(torque_maps['FL']):
                action = np.array([
                    0.5,  # steering (0.5表示直行，范围[0,1])
                    torque_maps['FL'][step],
                    torque_maps['FR'][step],
                    torque_maps['RL'][step],
                    torque_maps['RR'][step],
                ], dtype=np.float32)
            else:
                # 超出扭矩表范围，保持最后一个值
                action = np.array([0.5, 0, 0, 0, 0], dtype=np.float32)
            
            # 执行动作
            next_obs, reward, done, info = env.step(action)
            final_info = info
            
            # 记录历史数据
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
            hist['steering'].append(info.get('steering', 0))
            hist['lateral_error'].append(info.get('lateral_error', 0))
            
            # 记录奖励分量
            for k, v in info.items():
                if k.startswith("R_"):
                    if k not in reward_stats:
                        reward_stats[k] = []
                        hist[k] = []
                    reward_stats[k].append(v)
                    hist[k].append(v)
            
            obs = next_obs
            episode_reward += reward
            step += 1
            
            # 每1000步打印进度
            if step % 1000 == 0:
                speed_kmh = info.get('vx', 0) * 3.6
                print(f"[Progress] Step {step}/10000 | Speed: {speed_kmh:.1f} km/h | "
                      f"Reward: {episode_reward:.2f}")
            
            if done:
                break
        
        # Episode 结束
        final_speed = final_info.get('vx', 0.0) * 3.6
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n{'='*60}")
        print(f"  [{ts}] Validation Complete")
        print(f"  Total Steps: {step}")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Final Speed: {final_speed:.1f} km/h")
        print(f"{'='*60}\n")
        
        # 保存数据
        save_episode_data(1, hist, episode_reward, final_info, log_path)
        
        # 绘制图表
        plot_episode_data(hist, log_path, 1)
        
        # 保存汇总结果
        summary = {
            'validation_time': current_time,
            'total_steps': step,
            'total_reward': float(episode_reward),
            'final_speed_kmh': float(final_speed),
            'max_torque': max_torque,
            'target_slip_ratio': target_slip_ratio,
            'torque_source': torque_dir,
        }
        with open(os.path.join(log_path, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[Done] All results saved to: {log_path}")
        
    except KeyboardInterrupt:
        print("\n[Interrupt] Validation interrupted by user")
    except Exception as e:
        import traceback
        print(f"\n[Error] {e}")
        traceback.print_exc()
    finally:
        env.close()
        print("[Done] Resources released")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='ASR Torque Map Validation - Run CarSim with preset torque tables')
    parser.add_argument('--torque_dir', type=str, default="ASR_TorqueMap",
                        help='Directory containing FLmap.csv, FRmap.csv, RLmap.csv, RRmap.csv')
    parser.add_argument('--max_torque', type=float, default=3000.0,
                        help='Maximum torque for normalization (Nm)')
    parser.add_argument('--target_slip', type=float, default=0.08,
                        help='Target slip ratio for reward calculation')
    parser.add_argument('--log_dir', type=str, default="logs_ASR_Validation",
                        help='Directory for saving logs and plots')
    
    args = parser.parse_args()
    
    run_asr_validation(
        torque_dir=args.torque_dir,
        max_torque=args.max_torque,
        target_slip_ratio=args.target_slip,
        log_dir=args.log_dir,
    )
