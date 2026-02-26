import json
import numpy as np

data = json.load(open(r'e:\CarSim2024.1_Prog\Database\RL_VISION\RL_TC_VISION\logs_Live\Live_Carsim_20260213_102440\episode_data\episode_0.json'))

print("=" * 60)
print("Episode 数据分析")
print("=" * 60)

print(f"\n【基本信息】")
print(f"  Episode: {data['episode']}")
print(f"  Total Reward: {data['total_reward']:.4f}")
print(f"  步数: {data['num_steps']}")

print(f"\n【速度】")
velocities = data.get('velocity', [])
if velocities:
    print(f"  初始速度: {velocities[0]:.2f} km/h")
    print(f"  最终速度: {velocities[-1]:.2f} km/h")
    print(f"  最大速度: {max(velocities):.2f} km/h")
    print(f"  平均速度: {np.mean(velocities):.2f} km/h")

print(f"\n【各奖励分量统计】")
for key in ['R_Spd', 'R_Acc', 'R_Slp', 'R_Eng', 'R_Cns', 'R_Yaw', 'R_Beta']:
    if key in data:
        values = data[key]
        print(f"  {key}:")
        print(f"    Sum: {np.sum(values):.6f}")
        print(f"    Mean: {np.mean(values):.6f}")
        print(f"    Std:  {np.std(values):.6f}")
        print(f"    Min:  {np.min(values):.6f}")
        print(f"    Max:  {np.max(values):.6f}")

print(f"\n【终止时info】")
for k, v in data.items():
    if k.startswith('info_'):
        print(f"  {k}: {v}")

print(f"\n【当前奖励权重】")
print(f"  w_speed: 0.00")
print(f"  w_accel: 0.12")
print(f"  w_energy: -0.02")
print(f"  w_consistency: 0")
print(f"  w_beta: -0.1")
print(f"  w_slip: -0.35")
print(f"  w_smooth: 0.0")
print(f"  w_yaw: -5.0")

print(f"\n【滑移率分析】")
for key in ['S_L1', 'S_R1', 'S_L2', 'S_R2']:
    if key in data:
        values = data[key]
        print(f"  {key}: mean={np.mean(values):.4f}, std={np.std(values):.4f}, final={values[-1]:.4f}")
