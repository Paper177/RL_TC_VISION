import torch
import torch.nn as nn
import torch.nn.functional as F

def weight_init(m):
    if isinstance(m, nn.Linear):
        # 使用更保守的初始化
        nn.init.orthogonal_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        
        self.out = nn.Linear(128, action_dim)    
        self.action_bound = action_bound
        
        # 输出层使用更小的初始化
        nn.init.uniform_(self.out.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.out.bias, -3e-3, 3e-3)
        
        self.apply(weight_init)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        return torch.sigmoid(self.out(x)) * self.action_bound

class QValueNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        
        self.input_dim = state_dim + action_dim
        
        self.fc1 = nn.Linear(self.input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        
        self.out = nn.Linear(128, 1)
        
        # 输出层使用更小的初始化
        nn.init.uniform_(self.out.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.out.bias, -3e-3, 3e-3)
        
        self.apply(weight_init)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        return self.out(x)

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.q_network = QValueNet(state_dim, action_dim, hidden_dim)
    
    def forward(self, state, action):
        return self.q_network(state, action)

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super().__init__()
        self.policy_network = PolicyNet(state_dim, hidden_dim, action_dim, action_bound)
    
    def forward(self, state):
        return self.policy_network(state)
