# RL_TC_VISION — 基于强化学习的车辆限滑控制

基于 **TD3** 的车辆驱动防滑（TC）算法，在 **CarSim** 仿真环境中训练。通过封装 CarSim VS/Live 接口实现 Python 控制仿真，支持**纯物理状态**与**物理+视觉**两种观测模式。

---

## 项目结构

| 类型 | 说明 | 入口脚本 |
|------|------|----------|
| **无视觉** | 状态为 8 维物理量（速度、加速度、四轮滑移率、横摆角速度、质心侧偏角等） | `train_td3_Live_auto_restart.py` |
| **带视觉** | 状态 = 物理状态 + CNN 视觉特征，端到端训练 | `live_vision_train/train_td3_Live_vision.py` |
| **离线训练** | 不启动 CarSim，仅用已保存的 Buffer 训练 | `train_td3_offline.py` |

奖励函数在以**纵向控制为主**的前提下，同时考虑滑移率、横摆、侧偏角等**横向/稳定性**约束。

---

## 环境与依赖

- CarSim 2024.1（含 Live Animation / VS 接口）
- Python 3.x，PyTorch，NumPy，OpenCV（视觉版本），TensorBoard
- 需配置 `simfile.sim`、`vs_lv_ds_x64.dll`（或对应 VS DLL）

---

## 运行方式

**在线训练（无视觉）**

```bash
# 从零开始
python train_td3_Live_checkpoint.py

# 断点续训（每 N 个 episode 自动保存并退出，需手动重新运行）
python train_td3_Live_checkpoint.py --model "checkpoints/checkpoint_xxx.pt" --checkpoint_interval 200
```

**在线训练（带视觉）**

```bash
cd live_vision_train
python train_td3_Live_vision.py
```

**离线训练（仅 Buffer，不连 CarSim）**

```bash
# 从 Buffer 从零训练
python train_td3_offline.py --buffer "path/to/xxx_buffer.pkl" --steps 100000

# 加载已有模型 + Buffer 继续训
python train_td3_offline.py --model "path/to/model.pt" --steps 100000
```

因 CarSim 实时画面下长时间连续仿真可能出现内存问题，在线脚本支持**按 episode 数自动保存 checkpoint 并退出**，通过 `--model` 传入最新 checkpoint 即可继续训练。

---

## 可视化

- **TensorBoard**：在线/离线训练日志（Loss、Reward、速度等）  
  - 无视觉：`logs_TD3`  
  - 带视觉：`logs_TD3_Vision`  
  - 离线：`logs_TD3_Offline`
- **Web 可视化**：`python visualizer_app.py` 后访问页面，可查看各次训练的 episode 曲线与数据（需在 `visualizer_app.py` 中配置 `LOG_ROOT` 指向对应日志目录，如 `logs_TD3` 或 `logs_TD3_Vision`）。

---

## 主要环境设计

- `env_live.py` — 无视觉的 CarSim 实时环境（8 维状态）
- `live_vision_train/env_live_vision.py` — 带视觉的 CarSim 环境（物理 + CNN 特征）
