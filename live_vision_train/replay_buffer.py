"""
经验回放缓冲区实现

此模块提供了用于强化学习的经验回放缓冲区，
可以存储智能体与环境交互产生的经验，并支持随机采样。
经验回放技术能够打破数据的时间相关性，提高训练稳定性。
"""
import random
from collections import deque
from typing import Tuple, List, Any, Optional


class ReplayBuffer:
    """
    经验回放缓冲区类
    
    使用双端队列(deque)实现固定容量的缓冲区，用于存储智能体与环境交互的经验。
    当缓冲区满时，会自动移除最旧的经验。
    """
    
    def __init__(self, capacity: int):
        """
        初始化经验回放缓冲区
        
        参数:
            capacity: 缓冲区最大容量
        """
        # 设置缓冲区容量
        self.capacity = capacity
        
        # 创建固定大小的双端队列作为缓冲区存储结构
        # maxlen参数确保队列自动维护固定长度，当满时移除最旧元素
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self, 
        state: Any, 
        action: Any, 
        reward: float, 
        next_state: Any, 
        done: bool
    ) -> None:
        """
        将一条经验存储到缓冲区
        
        参数:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 执行动作后的下一个状态
            done: 回合是否结束的标志
        """
        # 将经验元组添加到缓冲区
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[List, List, List, List, List]:
        """
        从缓冲区中随机采样一批经验
        
        参数:
            batch_size: 采样的批次大小
            
        返回:
            元组(states, actions, rewards, next_states, dones)，每个元素是一个列表
        """
        # 从缓冲区中随机采样batch_size条经验
        batch = random.sample(self.buffer, batch_size)
        
        # 使用zip(*batch)解包并将同类数据聚合
        # 例如，将所有状态放在一起，所有动作放在一起等
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 返回聚合后的经验数据
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """
        获取当前缓冲区中的经验数量
        
        返回:
            缓冲区中的经验数量
        """
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """
        检查缓冲区是否有足够的经验可供采样
        
        参数:
            batch_size: 需要的最小经验数量
            
        返回:
            如果缓冲区中的经验数量大于等于batch_size，则返回True，否则返回False
        """
        return len(self.buffer) >= batch_size
    
    def clear(self) -> None:
        """
        清空缓冲区中的所有经验
        """
        self.buffer.clear()