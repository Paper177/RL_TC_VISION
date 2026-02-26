#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TD3 (Twin Delayed Deep Deterministic Policy Gradient) Agent
改进自DDPG：双Critic、延迟策略更新、目标策略平滑
"""
import numpy as np
import torch
import torch.nn as nn
import copy
from typing import Tuple
from collections import deque

from networks import ActorNetwork, CriticNetwork
from replay_buffer import ReplayBuffer


class OUNoise:
    """Ornstein-Uhlenbeck噪声"""
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def __call__(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state


class TD3Agent:
    """
    TD3 Agent
    
    主要改进：
    1. 双Critic网络 (Clipped Double Q-learning)
    2. 延迟策略更新 (Delayed Policy Updates)
    3. 目标策略平滑 (Target Policy Smoothing)
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_bound: float = 1.0,
        hidden_dim: int = 256,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        buffer_capacity: int = 1000000,
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,  # Actor更新频率
        elite_ratio: float = 0.3,
        elite_capacity: int = 100000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.device = device
        
        # 训练计数器
        self.total_it = 0
        
        # Actor网络
        self.actor = ActorNetwork(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
        
        # 双Critic网络 (TD3核心改进)
        self.critic_1 = CriticNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_1_opt = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr, eps=1e-5)
        
        self.critic_2 = CriticNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic_2_target = copy.deepcopy(self.critic_2)
        self.critic_2_opt = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr, eps=1e-5)
        
        # 经验回放缓冲区
        self.buffer = ReplayBuffer(buffer_capacity)
        self.elite_buffer = ReplayBuffer(elite_capacity)
        self.elite_ratio = elite_ratio
        
        # 噪声
        self.noise = OUNoise(action_dim)
        
    def reset_noise(self):
        self.noise.reset()
    
    def select_action(self, state: np.ndarray, noise_scale: float = 0.1) -> np.ndarray:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        
        if noise_scale > 0.0:
            ou_noise = self.noise() * noise_scale
            action = np.clip(action + ou_noise, 0.0, self.action_bound)
        
        return action
    
    def push(self, state, action, reward, next_state, done):
        """存入普通缓冲区"""
        self.buffer.push(state, action, reward, next_state, done)

    def push_elite(self, state, action, reward, next_state, done):
        """存入精英缓冲区"""
        self.elite_buffer.push(state, action, reward, next_state, done)
    
    def soft_update(self, net, target_net):
        with torch.no_grad():
            for param, target_param in zip(net.parameters(), target_net.parameters()):
                target_param.data.mul_(1 - self.tau)
                target_param.data.add_(self.tau * param.data)
    
    def train_step(self) -> Tuple[float, float, float, float]:
        """
        执行一步TD3训练
        返回: (critic_loss, actor_loss, critic_grad_norm, actor_grad_norm)
        """
        self.total_it += 1
        
        # 1. 混合采样
        if len(self.elite_buffer) < self.batch_size or len(self.buffer) < self.batch_size:
            if len(self.buffer) < self.batch_size:
                return 0.0, 0.0, 0.0, 0.0
            states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        else:
            n_elite = int(self.batch_size * self.elite_ratio)
            n_normal = self.batch_size - n_elite
            
            s_e, a_e, r_e, ns_e, d_e = self.elite_buffer.sample(n_elite)
            s_n, a_n, r_n, ns_n, d_n = self.buffer.sample(n_normal)
            
            states = s_e + s_n
            actions = a_e + a_n
            rewards = r_e + r_n
            next_states = ns_e + ns_n
            dones = d_e + d_n
        
        # 转换为Tensor
        states_tensor = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_tensor = torch.as_tensor(np.array(actions), dtype=torch.float32, device=self.device)
        rewards_tensor = torch.as_tensor(np.array(rewards), dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_states_tensor = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones_tensor = torch.as_tensor(np.array(dones), dtype=torch.float32, device=self.device).unsqueeze(-1)
        
        # ----------------------------
        # 2. 更新Critic (每步都更新)
        # ----------------------------
        with torch.no_grad():
            # 目标策略平滑：给目标动作添加噪声
            noise = (torch.randn_like(actions_tensor) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.actor_target(next_states_tensor) + noise).clamp(0.0, self.action_bound)
            
            # 计算目标Q值 (取两个Critic的最小值)
            target_q1 = self.critic_1_target(next_states_tensor, next_actions)
            target_q2 = self.critic_2_target(next_states_tensor, next_actions)
            target_q = torch.min(target_q1, target_q2)
            td_target = rewards_tensor + self.gamma * (1 - dones_tensor) * target_q
        
        # 更新Critic 1
        current_q1 = self.critic_1(states_tensor, actions_tensor)
        critic_1_loss = nn.MSELoss()(current_q1, td_target)
        
        self.critic_1_opt.zero_grad()
        critic_1_loss.backward()
        critic_1_grad = torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), 1.0)
        self.critic_1_opt.step()
        
        # 更新Critic 2
        current_q2 = self.critic_2(states_tensor, actions_tensor)
        critic_2_loss = nn.MSELoss()(current_q2, td_target)
        
        self.critic_2_opt.zero_grad()
        critic_2_loss.backward()
        critic_2_grad = torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), 1.0)
        self.critic_2_opt.step()
        
        critic_loss = (critic_1_loss.item() + critic_2_loss.item()) / 2
        critic_grad = (critic_1_grad.item() + critic_2_grad.item()) / 2
        
        # ----------------------------
        # 3. 延迟策略更新 (每隔policy_freq步更新一次Actor)
        # ----------------------------
        actor_loss = 0.0
        actor_grad = 0.0
        
        if self.total_it % self.policy_freq == 0:
            # Actor目标：最大化Critic_1的Q值
            actor_actions = self.actor(states_tensor)
            actor_loss_val = -self.critic_1(states_tensor, actor_actions).mean()
            
            self.actor_opt.zero_grad()
            actor_loss_val.backward()
            actor_grad = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_opt.step()
            
            actor_loss = actor_loss_val.item()
            
            # 软更新目标网络
            self.soft_update(self.actor, self.actor_target)
            self.soft_update(self.critic_1, self.critic_1_target)
            self.soft_update(self.critic_2, self.critic_2_target)
        
        return critic_loss, actor_loss, critic_grad, actor_grad
    
    def decay_learning_rate(self, factor: float = 0.5, min_lr: float = 1e-6):
        """
        衰减学习率
        
        参数:
            factor: 衰减因子，新学习率 = 当前学习率 * factor
            min_lr: 最小学习率
        """
        for param_group in self.actor_opt.param_groups:
            new_lr = max(param_group['lr'] * factor, min_lr)
            param_group['lr'] = new_lr
        
        for param_group in self.critic_1_opt.param_groups:
            new_lr = max(param_group['lr'] * factor, min_lr)
            param_group['lr'] = new_lr
        
        for param_group in self.critic_2_opt.param_groups:
            new_lr = max(param_group['lr'] * factor, min_lr)
            param_group['lr'] = new_lr
        
        return self.get_learning_rates()
    
    def get_learning_rates(self) -> dict:
        """获取当前学习率"""
        return {
            'actor_lr': self.actor_opt.param_groups[0]['lr'],
            'critic_lr': self.critic_1_opt.param_groups[0]['lr']
        }
    
    def save_model(self, filepath: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic_1': self.critic_1.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_1_opt': self.critic_1_opt.state_dict(),
            'critic_2_opt': self.critic_2_opt.state_dict(),
            'total_it': self.total_it
        }, filepath)
    
    def load_model(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic_1.load_state_dict(checkpoint['critic_1'])
        self.critic_2.load_state_dict(checkpoint['critic_2'])
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_2_target = copy.deepcopy(self.critic_2)
        self.actor_opt.load_state_dict(checkpoint['actor_opt'])
        self.critic_1_opt.load_state_dict(checkpoint['critic_1_opt'])
        self.critic_2_opt.load_state_dict(checkpoint['critic_2_opt'])
        self.total_it = checkpoint.get('total_it', 0)
