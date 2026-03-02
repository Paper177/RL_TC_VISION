"""
支持图像数据的经验回放缓冲区

使用预分配 numpy 数组实现高效存储和采样。
图像以 uint8 格式存储以节省内存（约为 float32 的 1/4）。

内存估算 (img_shape=(3,48,64), physics_dim=8, action_dim=4):
  - 100K capacity ≈ 1.85 GB
  - 50K  capacity ≈ 0.93 GB
"""
import numpy as np
import pickle
import os


class VisionReplayBuffer:
    """
    支持多模态数据（物理状态 + 图像）的经验回放缓冲区。
    使用环形缓冲区 + 预分配 numpy 数组，避免 Python 对象开销。
    """

    def __init__(self, capacity, physics_dim, action_dim, img_shape=(3, 48, 64)):
        self.capacity = capacity
        self.physics_dim = physics_dim
        self.action_dim = action_dim
        self.img_shape = img_shape
        self.ptr = 0
        self.size = 0

        self.physics_states = np.zeros((capacity, physics_dim), dtype=np.float32)
        self.images = np.zeros((capacity, *img_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_physics_states = np.zeros((capacity, physics_dim), dtype=np.float32)
        self.next_images = np.zeros((capacity, *img_shape), dtype=np.uint8)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def push(self, physics_state, image, action, reward,
             next_physics_state, next_image, done):
        idx = self.ptr
        self.physics_states[idx] = physics_state
        self.images[idx] = image
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_physics_states[idx] = next_physics_state
        self.next_images[idx] = next_image
        self.dones[idx] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.physics_states[indices],
            self.images[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_physics_states[indices],
            self.next_images[indices],
            self.dones[indices],
        )

    def __len__(self):
        return self.size

    def is_ready(self, batch_size):
        return self.size >= batch_size

    def get_memory_mb(self):
        total = (self.physics_states.nbytes + self.images.nbytes +
                 self.actions.nbytes + self.rewards.nbytes +
                 self.next_physics_states.nbytes + self.next_images.nbytes +
                 self.dones.nbytes)
        return total / (1024 * 1024)

    def save(self, filepath):
        data = {
            'capacity': self.capacity,
            'physics_dim': self.physics_dim,
            'action_dim': self.action_dim,
            'img_shape': self.img_shape,
            'ptr': self.ptr,
            'size': self.size,
            'physics_states': self.physics_states[:self.size],
            'images': self.images[:self.size],
            'actions': self.actions[:self.size],
            'rewards': self.rewards[:self.size],
            'next_physics_states': self.next_physics_states[:self.size],
            'next_images': self.next_images[:self.size],
            'dones': self.dones[:self.size],
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        loaded_size = data['size']
        n = min(loaded_size, self.capacity)

        self.physics_states[:n] = data['physics_states'][:n]
        self.images[:n] = data['images'][:n]
        self.actions[:n] = data['actions'][:n]
        self.rewards[:n] = data['rewards'][:n]
        self.next_physics_states[:n] = data['next_physics_states'][:n]
        self.next_images[:n] = data['next_images'][:n]
        self.dones[:n] = data['dones'][:n]

        self.size = n
        self.ptr = n % self.capacity
        print(f"[Buffer] Loaded {n} transitions from {filepath}")

    def clear(self):
        self.ptr = 0
        self.size = 0
