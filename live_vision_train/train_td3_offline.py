#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TD3 Offline Training Script

利用现有的 Replay Buffer 进行离线训练 (Offline RL)，无需启动 CarSim 环境。
适用于:
  1. 利用已有的高质量数据进一步优化策略
  2. 在没有仿真器环境的机器上训练
  3. 快速验证超参数
"""

import numpy as np
import torch
import os
import argparse
import pickle
import json
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import shutil

from td3_agent import VisionTD3Agent

def get_buffer_metadata(buffer_path):
    """从 Buffer 文件中读取元数据 (physics_dim, action_dim, etc.)"""
    if not os.path.exists(buffer_path):
        raise FileNotFoundError(f"Buffer file not found: {buffer_path}")
    
    print(f"[Offline] Reading metadata from {buffer_path}...")
    with open(buffer_path, 'rb') as f:
        data = pickle.load(f)
    
    metadata = {
        'physics_dim': data.get('physics_dim'),
        'action_dim': data.get('action_dim'),
        'img_shape': data.get('img_shape', (3, 48, 64)),
        'size': data.get('size', 0),
        'capacity': data.get('capacity', 0)
    }
    return metadata

def train_offline(
    buffer_path: str,
    total_steps: int = 100000,
    batch_size: int = 256,
    log_dir: str = "logs_TD3_Offline",
    pretrained_model_path: str = None,
    save_interval: int = 5000,
):
    # 1. 获取元数据并加载一次数据
    try:
        if not os.path.exists(buffer_path):
             raise FileNotFoundError(f"Buffer file not found: {buffer_path}")
        
        print(f"[Offline] Loading buffer data from {buffer_path}...")
        with open(buffer_path, 'rb') as f:
            data = pickle.load(f)
            
        meta = {
            'physics_dim': data.get('physics_dim'),
            'action_dim': data.get('action_dim'),
            'img_shape': data.get('img_shape', (3, 48, 64)),
            'size': data.get('size', 0),
            'capacity': data.get('capacity', 0)
        }
    except Exception as e:
        print(f"[Error] Failed to load buffer: {e}")
        return

    print(f"[Offline] Metadata: Physics={meta['physics_dim']}, Action={meta['action_dim']}, "
          f"Img={meta['img_shape']}, Size={meta['size']}")

    if meta['size'] < batch_size:
        print(f"[Error] Buffer size ({meta['size']}) is smaller than batch size ({batch_size})!")
        return

    # 2. 初始化环境配置 (无需真实环境)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Offline] Device: {device}")

    # 3. 初始化 Agent
    # 注意: Buffer capacity 设置为 meta['capacity'] 或更大，以容纳加载的数据
    agent = VisionTD3Agent(
        physics_dim=meta['physics_dim'],
        action_dim=meta['action_dim'],
        img_shape=meta['img_shape'],
        action_bound=1.0,  # 假设动作范围归一化为 [-1, 1] 或 [0, 1]，TD3通常处理连续动作
        vision_feat_dim=64,
        hidden_dim=256,
        actor_lr=3e-4,
        critic_lr=3e-4,
        buffer_capacity=meta['capacity'], 
        elite_capacity=30000, # 默认值
        batch_size=batch_size,
        gamma=0.99,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        elite_ratio=0.3, # 如果 buffer 中没有 elite 数据，这个比例可能需要调整，或者确保 elite buffer 也被加载
        device=device,
        use_amp=True
    )

    # 4. 加载 Buffer 数据
    print(f"[Offline] Populating agent buffer...")
    # 手动填充 buffer，避免重复加载文件
    n = min(meta['size'], agent.buffer.capacity)
    agent.buffer.physics_states[:n] = data['physics_states'][:n]
    agent.buffer.images[:n] = data['images'][:n]
    agent.buffer.actions[:n] = data['actions'][:n]
    agent.buffer.rewards[:n] = data['rewards'][:n]
    agent.buffer.next_physics_states[:n] = data['next_physics_states'][:n]
    agent.buffer.next_images[:n] = data['next_images'][:n]
    agent.buffer.dones[:n] = data['dones'][:n]
    agent.buffer.size = n
    agent.buffer.ptr = n % agent.buffer.capacity
    print(f"[Offline] Loaded {n} transitions into normal buffer.")

    # 释放内存
    del data
    import gc
    gc.collect()
    
    # 检查 Elite Buffer 是否也存在并加载
    elite_path = buffer_path.replace('.pkl', '_elite.pkl')
    if os.path.exists(elite_path):
        print(f"[Offline] Elite buffer found: {elite_path}")
        agent.elite_buffer.load(elite_path)
    else:
        print("[Offline] No elite buffer found. Training will use only normal buffer.")
        agent.elite_ratio = 0.0 # 如果没有 elite 数据，不仅用普通 buffer

    # 5. 加载预训练模型 (可选)
    start_step = 0
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"[Offline] Loading pretrained model: {pretrained_model_path}")
        agent.load_model(pretrained_model_path)
        # 尝试从文件名解析步数
        try:
            filename = os.path.basename(pretrained_model_path)
            # 假设文件名格式: checkpoint_STEP_REWARD.pt 或 similar
            # 这里简单处理，如果解析失败就从 0 开始
            if "checkpoint_" in filename:
                parts = filename.split('_')
                for i, p in enumerate(parts):
                    if p == "checkpoint" and i + 1 < len(parts):
                        start_step = int(parts[i+1])
                        break
        except:
            pass
    
    # 6. 设置日志
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"Offline_{current_time}")
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_path)
    
    # 备份代码
    script_dir = os.path.dirname(os.path.abspath(__file__))
    shutil.copy2(os.path.abspath(__file__), os.path.join(log_path, "train_offline_script.py"))

    print(f"\n{'='*60}")
    print(f"  TD3 Offline Training Start")
    print(f"  Steps: {total_steps}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Logs: {log_path}")
    print(f"{'='*60}\n")

    # 7. 训练循环
    pbar = tqdm(range(start_step, start_step + total_steps), unit="step")
    
    sum_c_loss = 0.0
    sum_a_loss = 0.0
    actor_update_count = 0

    checkpoint_dir = os.path.join(script_dir, "offline_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    try:
        for step in pbar:
            c_loss, a_loss, c_grad, a_grad = agent.train_step()
            
            sum_c_loss += c_loss
            if a_loss != 0:
                sum_a_loss += a_loss
                actor_update_count += 1
            
            # TensorBoard logging (每 100 步)
            if step % 100 == 0:
                writer.add_scalar('Loss/Critic', c_loss, step)
                if a_loss != 0:
                    writer.add_scalar('Loss/Actor', a_loss, step)
                writer.add_scalar('Grad/Critic', c_grad, step)
                writer.add_scalar('Grad/Actor', a_grad, step)
                
                # 更新进度条描述
                avg_a = sum_a_loss / max(1, actor_update_count)
                pbar.set_postfix({'C_Loss': f'{c_loss:.4f}', 'A_Loss': f'{avg_a:.4f}'})
                sum_c_loss = 0.0
                sum_a_loss = 0.0
                actor_update_count = 0

            # 保存模型
            if (step + 1) % save_interval == 0:
                save_name = f"offline_checkpoint_{step+1}.pt"
                save_path = os.path.join(checkpoint_dir, save_name)
                agent.save_model(save_path)
                # print(f"\n[Save] Model saved to {save_path}")

    except KeyboardInterrupt:
        print("\n[Offline] Training interrupted.")
    finally:
        final_save_path = os.path.join(log_path, "final_offline_model.pt")
        agent.save_model(final_save_path)
        writer.close()
        print(f"[Offline] Training finished. Final model: {final_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TD3 Offline Training')
    parser.add_argument('--buffer', type=str, required=True, help='Path to replay buffer (.pkl)')
    parser.add_argument('--model', type=str, default=None, help='Path to pretrained model (optional)')
    parser.add_argument('--steps', type=int, default=100000, help='Total training steps')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--log_dir', type=str, default="logs_TD3_Offline", help='Directory for TensorBoard logs')
    parser.add_argument('--save_interval', type=int, default=5000, help='Steps between model saves')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.buffer):
        print(f"Error: Buffer file '{args.buffer}' does not exist.")
        exit(1)
        
    train_offline(
        buffer_path=args.buffer,
        total_steps=args.steps,
        batch_size=args.batch_size,
        log_dir=args.log_dir,
        pretrained_model_path=args.model,
        save_interval=args.save_interval
    )
