# 造梦航海

游戏环境的观测和动作说明见 [环境文档](docs/env.md)，专家数据格式见
[专家数据文档](docs/expert-data.md)。

使用专家数据预训练 PPO 的策略网络（当前仅行为克隆，不使用奖励函数）：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u -m zaomeng_hanghai.train.train_bc
```

训练参数、checkpoint 与游戏推理命令见 [模仿学习说明](docs/imitation-learning.md)。
