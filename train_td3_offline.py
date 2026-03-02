#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TD3 Offline Training Script
不运行 CarSim 仿真，直接加载保存的 Buffer 进行训练
"""
import numpy as np
import torch
import os
import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import shutil

from td3_agent import TD3Agent

def train_td3_offline(
    model_path: str = None,
    buffer_path: str = None,
    train_steps: int = 100000,
    batch_size: int = 512,
    log_dir: str = "logs_TD3_Offline",
    save_interval: int = 5000,
    state_dim: int = 8,
    action_dim: int = 4,
    max_torque: float = 500.0
):
    # ================= 超参数设置 (保持与在线训练一致) =================
    hyperparams = {
        'Action Bound': 1.0,   
        'Hidden Dim': 256,
        'gamma': 0.99,
        'Actor LR': 2e-4,
        'Critic LR': 2e-4,
        'Buffer Capacity': 3000000,
        'Batch Size': batch_size,
        'Elite Ratio': 0.3,    
        'Elite Capacity': 1500000,
        'Noise Scale': 0.1,
        'Min Noise': 0.01,
        'Noise Decay': 0.995,
        'Policy Noise': 0.1,
        'Noise Clip': 0.25,
        'Policy Freq': 2,
    }
    
    # ================= 路径与日志 =================
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"TD3_Offline_{current_time}")
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_dir=log_path)
    
    checkpoint_save_dir = os.path.join(log_path, "checkpoints")
    os.makedirs(checkpoint_save_dir, exist_ok=True)

    print(f"\n========== Start TD3 Offline Training ==========")
    print(f"Log Directory: {log_path}")
    
    # ================= 初始化 Agent =================
    agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
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
    
    # ================= 加载模型（可选） =================
    if model_path and os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        agent.load_model(model_path)
    else:
        print("No pretrained model provided, training from scratch (random weights).")

    # ================= 加载 Buffer =================
    if not buffer_path and model_path:
        buffer_path = model_path.replace('.pt', '_buffer.pkl')
    
    if not buffer_path:
        print("Error: --buffer is required when no --model is provided.")
        return

    if os.path.exists(buffer_path):
        print(f"Loading buffer from: {buffer_path}")
        agent.load_buffer(buffer_path)
    else:
        print(f"Error: Buffer file not found: {buffer_path}")
        return
        
    if len(agent.buffer) < batch_size:
        print(f"Error: Buffer size ({len(agent.buffer)}) is smaller than batch size ({batch_size}).")
        return

    print(f"Buffer loaded successfully. Size: {len(agent.buffer)}")
    if hasattr(agent, 'elite_buffer'):
        print(f"Elite Buffer Size: {len(agent.elite_buffer)}")

    # ================= 训练循环 =================
    print(f"Starting training for {train_steps} steps...")
    
    try:
        pbar = tqdm(range(train_steps), desc="Offline Training", unit="step")
        for step in pbar:
            c_loss, a_loss, c_grad, a_grad = agent.train_step()
            
            # 记录日志
            writer.add_scalar('Loss/Critic', c_loss, step)
            writer.add_scalar('Loss/Actor', a_loss, step)
            writer.add_scalar('Train/Critic_Grad', c_grad, step)
            writer.add_scalar('Train/Actor_Grad', a_grad, step)
            
            pbar.set_postfix({'C_Loss': f'{c_loss:.4f}', 'A_Loss': f'{a_loss:.4f}'})
            
            # 定期保存
            if (step + 1) % save_interval == 0:
                save_path = os.path.join(checkpoint_save_dir, f"offline_checkpoint_{step+1}.pt")
                agent.save_model(save_path)
                
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    
    # 保存最终模型
    final_save_path = os.path.join(checkpoint_save_dir, f"offline_final_{train_steps}.pt")
    agent.save_model(final_save_path)
    print(f"Training finished. Model saved to: {final_save_path}")
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TD3 Offline Training")
    parser.add_argument("--model", type=str, default=None, help="Path to the pretrained model (.pt). If not provided, trains from scratch.")
    parser.add_argument("--buffer", type=str, default=None, help="Path to the buffer file (.pkl). If not provided, infers from model path.")
    parser.add_argument("--steps", type=int, default=300000, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    
    args = parser.parse_args()
    
    train_td3_offline(    
        model_path=args.model,
        buffer_path=args.buffer,
        train_steps=args.steps,
        batch_size=args.batch_size
    )
