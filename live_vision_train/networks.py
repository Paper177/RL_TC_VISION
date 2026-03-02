"""
多模态视觉 TD3 网络结构 (live_vision_train)

┌───────────────────────────────────────────────────────────────────────────────┐
│ VisionEncoder                                                                 │
│   Input: image (B, 3, H, W)                                                   │
│   Conv2d 3→32 → Conv2d 32→64 → Conv2d 64→128 → Conv2d 128→64 (stride=2, ReLU) │
│   → AdaptiveAvgPool2d(2,2) → Flatten → Linear(256→64) → LayerNorm → tanh      │
│   Output: (B, 64)                                                             │
└───────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│ VisionActorNetwork                                                            │
│   physics (B, 8)  ──→ Linear(8→64) + LayerNorm + ReLU ──┐                     │
│   image (B,3,H,W) ──→ VisionEncoder ──→ (B, 64) ────────┼→ Concat (B, 128)    │
│                                                                               │
│   → Linear(128→256) + LN + ReLU → Linear(256→128) + LN + ReLU                 │
│   → Linear(128→4) → sigmoid × action_bound  →  action (B, 4)                  │
└───────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ VisionCriticNetwork                                                                     │
│   physics||action (B, 12) ──→ Linear(12→64) + LayerNorm + ReLU ──┐                      │
│   image (B,3,H,W) ──→ VisionEncoder ──→ (B, 64) ─────────────────┼→ Concat(B, 128)      │
│                                                                                         │
│   → Linear(128→256) + LN + ReLU → Linear(256→128) + LN + ReLU → Linear→1 → Q(B, 1)      │
└─────────────────────────────────────────────────────────────────────────────────────────┘
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionEncoder(nn.Module):
    """
    轻量级CNN视觉编码器，从道路图像中提取路面特征。
    使用 AdaptiveAvgPool2d 适配任意输入分辨率。
    使用 LayerNorm 替代 BatchNorm，兼容 RL 中 batch_size=1 的推理场景。
    """
    def __init__(self, img_channels=3, output_dim=64):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.fc = nn.Linear(64 * 2 * 2, output_dim)
        self.ln = nn.LayerNorm(output_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.ln(self.fc(x)))
        return x


class VisionActorNetwork(nn.Module):
    """
    多模态 Actor 网络 (双分支架构)
      - 视觉分支: VisionEncoder 处理道路图像
      - 物理分支: MLP 处理车辆动力学状态
      - 融合层: 合并两分支特征输出控制动作
    """
    def __init__(self, physics_dim, action_dim, action_bound=1.0,
                 vision_feat_dim=64, hidden_dim=256, img_channels=3):
        super().__init__()
        self.action_bound = action_bound

        self.vision_encoder = VisionEncoder(img_channels, vision_feat_dim)

        self.phys_fc = nn.Linear(physics_dim, 64)
        self.phys_ln = nn.LayerNorm(64)

        fusion_dim = vision_feat_dim + 64
        self.fc1 = nn.Linear(fusion_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln2 = nn.LayerNorm(hidden_dim // 2)
        self.out = nn.Linear(hidden_dim // 2, action_dim)

        nn.init.uniform_(self.out.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.out.bias, -3e-3, 3e-3)

    def forward(self, physics_state, image):
        vis_feat = self.vision_encoder(image)
        phys_feat = F.relu(self.phys_ln(self.phys_fc(physics_state)))

        x = torch.cat([vis_feat, phys_feat], dim=1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return torch.sigmoid(self.out(x)) * self.action_bound


class VisionCriticNetwork(nn.Module):
    """
    多模态 Critic 网络 (双分支架构)
      - 视觉分支: VisionEncoder 处理道路图像
      - 物理+动作分支: MLP 处理车辆状态和控制动作
      - 融合层: 合并后输出 Q 值
    """
    def __init__(self, physics_dim, action_dim, vision_feat_dim=64,
                 hidden_dim=256, img_channels=3):
        super().__init__()

        self.vision_encoder = VisionEncoder(img_channels, vision_feat_dim)

        self.phys_act_fc = nn.Linear(physics_dim + action_dim, 64)
        self.phys_act_ln = nn.LayerNorm(64)

        fusion_dim = vision_feat_dim + 64
        self.fc1 = nn.Linear(fusion_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln2 = nn.LayerNorm(hidden_dim // 2)
        self.out = nn.Linear(hidden_dim // 2, 1)

        nn.init.uniform_(self.out.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.out.bias, -3e-3, 3e-3)

    def forward(self, physics_state, action, image):
        vis_feat = self.vision_encoder(image)
        phys_act = torch.cat([physics_state, action], dim=1)
        phys_act_feat = F.relu(self.phys_act_ln(self.phys_act_fc(phys_act)))

        x = torch.cat([vis_feat, phys_act_feat], dim=1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.out(x)
