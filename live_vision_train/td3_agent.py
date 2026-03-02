#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Vision TD3 Agent - 端到端多模态强化学习智能体

核心特点:
  1. 双分支架构: CNN处理道路图像 + MLP处理车辆动力学
  2. CNN集成在Actor/Critic内部，通过RL梯度端到端训练
  3. 双Critic (Clipped Double Q-learning)
  4. 延迟策略更新 + 目标策略平滑
"""
import numpy as np
import torch
import torch.nn as nn
import copy
from typing import Tuple

from networks import VisionActorNetwork, VisionCriticNetwork
from replay_buffer import VisionReplayBuffer


class OUNoise:
    """Ornstein-Uhlenbeck 过程噪声"""
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


class VisionTD3Agent:
    """
    多模态 Vision TD3 智能体

    状态由两部分组成:
      - physics_state: 车辆动力学特征向量 (float32)
      - image: 道路图像 (uint8, CHW格式)

    CNN 作为 Actor/Critic 网络的一部分，通过 RL 的 policy/value loss 端到端训练，
    使其学会提取对控制决策有用的路面特征。
    """
    def __init__(
        self,
        physics_dim: int,
        action_dim: int,
        img_shape: tuple = (3, 48, 64),
        action_bound: float = 1.0,
        vision_feat_dim: int = 64,
        hidden_dim: int = 256,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        buffer_capacity: int = 100000,
        elite_capacity: int = 20000,
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        elite_ratio: float = 0.3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.physics_dim = physics_dim
        self.action_dim = action_dim
        self.img_shape = img_shape
        self.action_bound = action_bound
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.device = device
        self.elite_ratio = elite_ratio
        self.total_it = 0

        # --- Actor ---
        self.actor = VisionActorNetwork(
            physics_dim, action_dim, action_bound,
            vision_feat_dim, hidden_dim
        ).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)

        # --- Critic 1 ---
        self.critic_1 = VisionCriticNetwork(
            physics_dim, action_dim, vision_feat_dim, hidden_dim
        ).to(device)
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_1_opt = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr, eps=1e-5)

        # --- Critic 2 ---
        self.critic_2 = VisionCriticNetwork(
            physics_dim, action_dim, vision_feat_dim, hidden_dim
        ).to(device)
        self.critic_2_target = copy.deepcopy(self.critic_2)
        self.critic_2_opt = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr, eps=1e-5)

        # --- Replay Buffers ---
        self.buffer = VisionReplayBuffer(buffer_capacity, physics_dim, action_dim, img_shape)
        self.elite_buffer = VisionReplayBuffer(elite_capacity, physics_dim, action_dim, img_shape)

        print(f"[Agent] Normal buffer: {buffer_capacity} ({self.buffer.get_memory_mb():.0f} MB)")
        print(f"[Agent] Elite  buffer: {elite_capacity} ({self.elite_buffer.get_memory_mb():.0f} MB)")

        # --- Noise ---
        self.noise = OUNoise(action_dim)

    def _prepare_image_tensor(self, image_uint8):
        """uint8 图像 → float32 GPU tensor, [0,1] 归一化"""
        img = torch.as_tensor(image_uint8, dtype=torch.float32, device=self.device) / 255.0
        if img.dim() == 3:
            img = img.unsqueeze(0)
        return img

    def reset_noise(self):
        self.noise.reset()

    def select_action(self, physics_state: np.ndarray, image: np.ndarray,
                      noise_scale: float = 0.1) -> np.ndarray:
        phys_t = torch.as_tensor(physics_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        img_t = self._prepare_image_tensor(image)

        with torch.no_grad():
            action = self.actor(phys_t, img_t).cpu().numpy()[0]

        if noise_scale > 0.0:
            ou_noise = self.noise() * noise_scale
            action = np.clip(action + ou_noise, 0.0, self.action_bound)

        return action

    def push(self, physics_state, image, action, reward,
             next_physics_state, next_image, done):
        self.buffer.push(physics_state, image, action, reward,
                         next_physics_state, next_image, done)

    def push_elite(self, physics_state, image, action, reward,
                   next_physics_state, next_image, done):
        self.elite_buffer.push(physics_state, image, action, reward,
                               next_physics_state, next_image, done)

    def soft_update(self, net, target_net):
        with torch.no_grad():
            for p, tp in zip(net.parameters(), target_net.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

    def _sample_batch(self):
        """混合采样: 从普通 buffer 和 elite buffer 按比例混合"""
        if not self.buffer.is_ready(self.batch_size):
            return None

        if self.elite_buffer.is_ready(self.batch_size):
            n_elite = int(self.batch_size * self.elite_ratio)
            n_normal = self.batch_size - n_elite

            pe, ie, ae, re, npe, nie, de = self.elite_buffer.sample(n_elite)
            pn, in_, an, rn, npn, nin, dn = self.buffer.sample(n_normal)

            return (
                np.concatenate([pe, pn]),
                np.concatenate([ie, in_]),
                np.concatenate([ae, an]),
                np.concatenate([re, rn]),
                np.concatenate([npe, npn]),
                np.concatenate([nie, nin]),
                np.concatenate([de, dn]),
            )
        else:
            return self.buffer.sample(self.batch_size)

    def train_step(self) -> Tuple[float, float, float, float]:
        """
        执行一步 TD3 训练 (端到端, 包含 CNN)
        返回: (critic_loss, actor_loss, critic_grad_norm, actor_grad_norm)
        """
        self.total_it += 1

        batch = self._sample_batch()
        if batch is None:
            return 0.0, 0.0, 0.0, 0.0

        phys, imgs, acts, rews, next_phys, next_imgs, dones = batch

        phys_t = torch.as_tensor(phys, dtype=torch.float32, device=self.device)
        imgs_t = torch.as_tensor(imgs, dtype=torch.float32, device=self.device) / 255.0
        acts_t = torch.as_tensor(acts, dtype=torch.float32, device=self.device)
        rews_t = torch.as_tensor(rews, dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_phys_t = torch.as_tensor(next_phys, dtype=torch.float32, device=self.device)
        next_imgs_t = torch.as_tensor(next_imgs, dtype=torch.float32, device=self.device) / 255.0
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)

        # ---- Critic 更新 (每步) ----
        with torch.no_grad():
            noise = (torch.randn_like(acts_t) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_phys_t, next_imgs_t) + noise).clamp(
                0.0, self.action_bound)

            target_q1 = self.critic_1_target(next_phys_t, next_actions, next_imgs_t)
            target_q2 = self.critic_2_target(next_phys_t, next_actions, next_imgs_t)
            target_q = torch.min(target_q1, target_q2)
            td_target = rews_t + self.gamma * (1 - dones_t) * target_q

        current_q1 = self.critic_1(phys_t, acts_t, imgs_t)
        critic_1_loss = nn.MSELoss()(current_q1, td_target)
        self.critic_1_opt.zero_grad()
        critic_1_loss.backward()
        c1_grad = torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), 1.0)
        self.critic_1_opt.step()

        current_q2 = self.critic_2(phys_t, acts_t, imgs_t)
        critic_2_loss = nn.MSELoss()(current_q2, td_target)
        self.critic_2_opt.zero_grad()
        critic_2_loss.backward()
        c2_grad = torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), 1.0)
        self.critic_2_opt.step()

        critic_loss = (critic_1_loss.item() + critic_2_loss.item()) / 2
        critic_grad = (c1_grad.item() + c2_grad.item()) / 2

        self.soft_update(self.critic_1, self.critic_1_target)
        self.soft_update(self.critic_2, self.critic_2_target)

        # ---- Actor 延迟更新 ----
        actor_loss = 0.0
        actor_grad = 0.0

        if self.total_it % self.policy_freq == 0:
            actor_actions = self.actor(phys_t, imgs_t)
            q1 = self.critic_1(phys_t, actor_actions, imgs_t)
            q2 = self.critic_2(phys_t, actor_actions, imgs_t)
            actor_loss_val = -torch.min(q1, q2).mean()

            self.actor_opt.zero_grad()
            actor_loss_val.backward()
            actor_grad = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_opt.step()

            actor_loss = actor_loss_val.item()
            self.soft_update(self.actor, self.actor_target)

        return critic_loss, actor_loss, critic_grad, actor_grad

    # ---- 学习率管理 ----
    def decay_learning_rate(self, factor: float = 0.5, min_lr: float = 1e-6):
        for opt in [self.actor_opt, self.critic_1_opt, self.critic_2_opt]:
            for pg in opt.param_groups:
                pg['lr'] = max(pg['lr'] * factor, min_lr)
        return self.get_learning_rates()

    def get_learning_rates(self) -> dict:
        return {
            'actor_lr': self.actor_opt.param_groups[0]['lr'],
            'critic_lr': self.critic_1_opt.param_groups[0]['lr']
        }

    # ---- 模型持久化 ----
    def save_model(self, filepath: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_1': self.critic_1.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'critic_1_target': self.critic_1_target.state_dict(),
            'critic_2_target': self.critic_2_target.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_1_opt': self.critic_1_opt.state_dict(),
            'critic_2_opt': self.critic_2_opt.state_dict(),
            'total_it': self.total_it,
            'physics_dim': self.physics_dim,
            'action_dim': self.action_dim,
            'img_shape': self.img_shape,
        }, filepath)

    def load_model(self, filepath: str):
        ckpt = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic_1.load_state_dict(ckpt['critic_1'])
        self.critic_2.load_state_dict(ckpt['critic_2'])

        if 'actor_target' in ckpt:
            self.actor_target.load_state_dict(ckpt['actor_target'])
        else:
            self.actor_target = copy.deepcopy(self.actor)

        if 'critic_1_target' in ckpt:
            self.critic_1_target.load_state_dict(ckpt['critic_1_target'])
        else:
            self.critic_1_target = copy.deepcopy(self.critic_1)

        if 'critic_2_target' in ckpt:
            self.critic_2_target.load_state_dict(ckpt['critic_2_target'])
        else:
            self.critic_2_target = copy.deepcopy(self.critic_2)

        self.actor_opt.load_state_dict(ckpt['actor_opt'])
        self.critic_1_opt.load_state_dict(ckpt['critic_1_opt'])
        self.critic_2_opt.load_state_dict(ckpt['critic_2_opt'])
        self.total_it = ckpt.get('total_it', 0)

    def save_buffer(self, filepath: str):
        self.buffer.save(filepath)
        elite_path = filepath.replace('.pkl', '_elite.pkl')
        self.elite_buffer.save(elite_path)

    def load_buffer(self, filepath: str):
        self.buffer.load(filepath)
        import os
        elite_path = filepath.replace('.pkl', '_elite.pkl')
        if os.path.exists(elite_path):
            self.elite_buffer.load(elite_path)
