# PPO 策略的专家模仿学习

当前阶段使用行为克隆（BC）：以专家观测为输入，以专家移动动作为标签，
最小化 `CrossEntropyLoss(actor(obs), acts)`。没有奖励、优势估计或 PPO clipped loss。
专家记录不包含连续转移，因此不将相邻记录拼成 PPO rollout。

`ActorCritic` 提供独立的 actor 和 critic，默认均为 `96 → 256 → 256 → 256` 的 Tanh MLP；
actor 输出 9 个动作 logits，critic 输出标量价值。BC 只优化 actor。
当前使用 [v4 状态编码](state-encoding.md)，只接受新版 96 维专家数据。
旧 60 维数据缺少边界框，不能直接用于新版模型训练。
`act()` 返回动作、log probability、value，`evaluate_actions()` 返回 log probability、
entropy、value，供后续添加 PPO 更新时复用。critic 目前是随机初始化，不能用来评估收益。

## 离线训练

在项目根目录运行，不需要打开游戏：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u -m zaomeng_hanghai.train.train_bc
```

默认递归读取 `data/expert` 中所有运行目录的清单，仅加载 `kept=true` 的完整 NPZ。
也可以限定某个运行目录：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u -m zaomeng_hanghai.train.train_bc --data data/expert/20261024_201616_c5f7a78d --epochs 50 --device auto
```

参数：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--data` | `data/expert` | 专家根目录或单次采集目录 |
| `--output` | `runs/bc/<时间_随机后缀>` | 新目录，拒绝覆盖已有目录 |
| `--epochs` | `50` | 最大训练轮数 |
| `--batch-size` | `512` | 每个 batch 的样本数 |
| `--learning-rate` | `0.0003` | Adam 学习率 |
| `--hidden-sizes` | `256 256 256` | 隐藏层宽度 |
| `--validation-fraction` | `0.2` | 按整局划分的验证比例；`0` 禁用验证 |
| `--patience` | `10` | 验证损失连续不改善时早停；`0` 禁用 |
| `--seed` | `42` | 模型初始化、整局划分和 batch 打乱种子 |
| `--device` | `auto` | 有 CUDA 时用 CUDA，否则 CPU；可显式指定 `cpu` 或 `cuda:0` |
| `--threads` | `4` | PyTorch CPU 线程数 |

保留原始归一化状态，不对 one-hot、实体存在标志和补零再做标准化。
同一局只属于训练集或验证集。仅有一局时需禁用验证或补充数据。
数据集在启动时读取一次；训练前应结束采集，避免读取正在追加的清单。

## 训练产物

- `best.pt`：验证交叉熵最低的模型；禁用验证时按训练损失选择。
- `last.pt`：最后完成一轮的模型。
- `config.json`：参数和实际设备。
- `split.json`：训练集和验证集的完整局文件路径，便于复查划分。
- `history.jsonl`：每轮损失、动作准确率、各动作召回率、动作计数和混淆矩阵。

checkpoint 包含模型结构、动作映射、actor/critic 权重、BC 优化器状态和指标。
`critic_trained=false` 明确说明价值网络未训练。后续 PPO 训练应加载模型权重，
重新创建包括 actor 和 critic 的 PPO 优化器。当前入口不提供断点续训。

动作准确率衡量与专家的一致程度，不等于游戏通关率。
样本中的动作类别可能不均衡，可结合每个动作的召回率和计数判断效果。
BC 默认使用普通交叉熵，保留专家动作的实际分布。

## 加载与游戏推理

```python
from zaomeng_hanghai.policy import load_policy

policy = load_policy("runs/bc/实际运行目录/best.pt", device="cpu")
action = policy.predict(env.state_encode())  # 默认 argmax；返回 0..8
status = env.step(action)
```

完整游戏入口（运行后会控制游戏，请先停在航海起始界面并确保窗口无遮挡）：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u -m zaomeng_hanghai.run_policy runs/bc/实际运行目录/best.pt --episodes 3
```

可加 `--render` 查看调试窗口，`--stochastic` 按策略概率采样，默认确定性动作。
开火与清场后的导航仍由 `Env` 的公共控制器执行，开局和战斗移动由 actor 选择。
`--max-episode-seconds` 默认 300 秒，在每次决策前检查；阻塞中的环境调用可能超出预算。
`--combat-timeout` 默认 30 秒，限制同一关的累计战斗时间；切关重置，加载和自动导航期间暂停。
单关战斗超时为 `-3`，加载、自动过关及外部整局预算截断为 `-4`，详见环境文档。
`Ctrl+C` 或异常退出时释放移动键和开火键。命令中的 `--episodes` 是尝试次数。

## 离线测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_imitation.py -v
```

测试不启动游戏，覆盖清单过滤、整局划分、坏数据检查、BC 学习、critic 保持不变、
PPO 接口一致性和 checkpoint 加载。
