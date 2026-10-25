# 造梦航海

游戏环境的观测和动作说明见 [环境文档](docs/env.md)，专家数据格式见
[专家数据文档](docs/expert-data.md)。

使用专家数据预训练 PPO 的策略网络（当前仅行为克隆，不使用奖励函数）：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u -m zaomeng_hanghai.train.train_bc
```

训练参数、checkpoint 与游戏推理命令见 [模仿学习说明](docs/imitation-learning.md)。

在线 PPO 训练、断点续训和仅在奖励变化时打印日志见 [PPO 训练说明](docs/ppo-training.md)：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u -m zaomeng_hanghai.train.train_ppo --init runs/bc/你的运行目录/best.pt --steps 100000
```
