import torch
from torchviz import make_dot
from networks import VisionActorNetwork, VisionCriticNetwork

# 1. 模拟输入数据 (Batch size = 2 方便看维度)
batch_size = 2
physics_dim = 10  # 假设车辆状态有10个维度 (速度, 角度等)
action_dim = 2    # 假设输出 2 个动作 (方向盘, 油门)
img_channels = 3
img_height = 128
img_width = 128

# 随机生成输入
dummy_physics = torch.randn(batch_size, physics_dim)
dummy_image = torch.randn(batch_size, img_channels, img_height, img_width)
dummy_action = torch.randn(batch_size, action_dim)

# 2. 实例化模型
actor = VisionActorNetwork(physics_dim, action_dim)
critic = VisionCriticNetwork(physics_dim, action_dim)

# 3. 前向传播
actor_out = actor(dummy_physics, dummy_image)
critic_out = critic(dummy_physics, dummy_action, dummy_image)

# 4. 生成 Actor 可视化图
print("正在生成 Actor 网络结构图...")
dot_actor = make_dot(actor_out, params=dict(actor.named_parameters()))

# 尝试渲染为PNG，如果失败则保存为dot文件
try:
    dot_actor.render("vision_actor_structure", format="png", cleanup=True)
    print("✅ 已保存: vision_actor_structure.png")
except Exception as e:
    print(f"⚠️  无法生成PNG（需要安装Graphviz软件）: {e}")
    # 保存为dot文件，可以用在线工具打开: http://viz-js.com/
    with open("vision_actor_structure.dot", "w") as f:
        f.write(dot_actor.source)
    print("✅ 已保存: vision_actor_structure.dot (可用 http://viz-js.com/ 在线查看)")

# 5. 生成 Critic 可视化图
print("正在生成 Critic 网络结构图...")
dot_critic = make_dot(critic_out, params=dict(critic.named_parameters()))

try:
    dot_critic.render("vision_critic_structure", format="png", cleanup=True)
    print("✅ 已保存: vision_critic_structure.png")
except Exception as e:
    print(f"⚠️  无法生成PNG（需要安装Graphviz软件）: {e}")
    with open("vision_critic_structure.dot", "w") as f:
        f.write(dot_critic.source)
    print("✅ 已保存: vision_critic_structure.dot (可用 http://viz-js.com/ 在线查看)")

print("\n💡 提示: 要生成PNG图片，请运行: conda install -c conda-forge graphviz")
