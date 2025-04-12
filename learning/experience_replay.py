import numpy as np
from collections import deque
import random

class PrioritizedExperienceReplay:
    def __init__(self, max_size=100000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.max_size = max_size
        self.alpha = alpha  # 优先级指数
        self.beta = beta    # 重要性采样指数
        self.beta_increment = beta_increment
        self.memory = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        
    def add(self, experience):
        """添加经验到回放池"""
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.memory.append(experience)
        self.priorities.append(max_priority)
        
    def sample(self, batch_size):
        """采样一批经验"""
        # 计算采样概率
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # 采样索引
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        
        # 计算重要性采样权重
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # 获取经验
        experiences = [self.memory[idx] for idx in indices]
        
        return experiences, indices, weights
        
    def update_priorities(self, indices, priorities):
        """更新经验的优先级"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            
    def __len__(self):
        return len(self.memory)

class OnlineLearning:
    def __init__(self, replay_memory, batch_size=32, update_frequency=1000):
        self.replay_memory = replay_memory
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.step_counter = 0
        
    def learn(self, agent):
        """在线学习过程"""
        if len(self.replay_memory) < self.batch_size:
            return
            
        # 采样经验
        experiences, indices, weights = self.replay_memory.sample(self.batch_size)
        
        # 更新智能体
        agent.replay(experiences, weights)
        
        # 更新目标网络
        if self.step_counter % self.update_frequency == 0:
            agent.update_target_model()
            
        self.step_counter += 1
        
    def add_experience(self, state, action, reward, next_state, done):
        """添加新的经验"""
        experience = (state, action, reward, next_state, done)
        self.replay_memory.add(experience) 