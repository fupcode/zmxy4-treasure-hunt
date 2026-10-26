# 在线 PPO 训练和续训

训练入口为 `zaomeng_hanghai.train.train_ppo`，算法、采样和 checkpoint 代码均在
`src/zaomeng_hanghai/train/`。推理继续使用 `policy.py` 和 `run_policy.py`。
当前观测为 [v4 的 96 维编码](state-encoding.md)，旧 60 维模型不能直接初始化或续训。

运行在线训练会操作游戏窗口。先将游戏停在航海起始界面，确保窗口无遮挡。
当前文档中的 `你的运行目录` 需要替换为实际目录名。

## 从专家模仿学习模型开始

```powershell
.\.venv\Scripts\python.exe -X utf8 -u -m zaomeng_hanghai.train.train_ppo --init runs/bc/你的运行目录/best.pt --steps 100000
```

`--init` 加载模型结构和权重，新建包含 actor 和 critic 的 PPO 优化器。
因此不会误用 BC 中只包含 actor 参数的优化器。兼容的 v4 模型结构以 checkpoint 为准；
隐藏层不会自动变为当前默认的 `256 × 3`，60 维旧模型会被拒绝。

也可从随机初始化开始，默认隐藏层为三层、每层 256 个单元：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u -m zaomeng_hanghai.train.train_ppo --steps 100000
```

## 继续训练

```powershell
.\.venv\Scripts\python.exe -X utf8 -u -m zaomeng_hanghai.train.train_ppo --resume runs/ppo/你的运行目录/last.pt --steps 50000
```

`--steps 50000` 表示本次**额外**采集至少 50,000 个有效步骤。
恢复模型、Adam 状态、PPO 超参数、有效/已训练步数、更新次数、局数、成功数、
近期回报、奖励日志状态，以及 Python/NumPy/PyTorch 的随机数状态。
恢复时从新一局开始，无法恢复真实游戏的中途画面和实体状态。

默认继续写入 checkpoint 所在目录，日志追加。也可用 `--output` 指定一个不存在的新目录。
新训练拒绝覆盖已有目录；同目录续训会替换 `last.pt`，并按需替换 `best.pt`。
`--resume` 只接受 PPO 存档，BC 模型应使用 `--init`。

续训默认沿用已保存的 PPO 参数，但每关战斗时限默认使用 60 秒，覆盖旧存档的时限；命令行可以显式指定其他时限：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u -m zaomeng_hanghai.train.train_ppo --resume runs/ppo/你的运行目录/last.pt --steps 50000 --learning-rate 0.00005 --combat-timeout 60
```

新训练设置默认保存到 `config.json`，每次启动的有效配置追加到 `sessions.jsonl`。
checkpoint 中始终保存当前生效的参数。

## 奖励日志

只比较**相邻有效步骤的总奖励数值**，相同则不打印、不写入 `rewards.jsonl`：

```text
实际奖励： 0.98  0.98  -0.12  -0.12  0.98
打印奖励： 0.98        -0.12         0.98
```

每条 `[Reward]` 日志包含步数、局数、动作、总奖励和非零奖励分项。
即使动作或分项变化，只要总奖励相等也不重复打印。
首条有效奖励会打印；跨局不清空上一奖励；同目录续训与日志末条奖励继续比较。
引擎故障 `-4` 不作为一条零奖励打印，不改变上一条有效奖励。
局结束的 `[Episode]` 汇总和更新时的 `[PPO]` 指标独立打印。

## 采样、更新和超时

采用 PPO clipped policy loss、价值 MSE、熵奖励和 GAE，同时训练独立的 actor/critic。
采样时按策略分布随机选择动作；同一批采样期间保持权重固定。

累计到 `--rollout-steps` 后仍完成当前局，释放按键，再进行 PPO 更新。
达到本次 `--steps` 目标后也完成当前局，因此实际有效步数可能超过目标。
对于 `-3/-4` 截断，先由引擎重开到起始区域再优化，避免优化期间继续处于战斗。

- 成功 `1` 和死亡 `-2`：终止状态价值为 0。
- 单关战斗超时 `-3`：保留超时惩罚，使用截断前有效观测的 critic 值 bootstrap；GAE 不跨局递推。
- 引擎问题 `-4`：丢弃故障步骤；已有有效样本可保留，并在最后一个有效观测处切断轨迹，
  不把故障产生的零奖励、未知观测或下一局初始观测当作转移。
- 加载和自动导航期间暂停战斗计时，切关重置，默认每关 60 秒。
- 连续 3 次引擎/开局失败时停止，避免对异常窗口持续重试；可用 `--max-engine-failures` 调整。

折扣按每次 `Env.step_with_reward()` 决策计算，包含自动导航的一步可能持续更久，
当前没有按实际秒数调整 gamma。没有给已有专家数据补造奖励或转移。

## 保存和中断

默认输出为 `runs/ppo/<时间_随机后缀>/`：

| 文件 | 内容 |
| --- | --- |
| `last.pt` | 最近完整完成的一次 PPO 更新；开始采样前也保存初始状态 |
| `best.pt` | 按最近最多 20 个有效完整局的平均训练回报选择的模型，不是独立评估结果 |
| `rewards.jsonl` | 仅奖励数值变化时的记录 |
| `episodes.jsonl` | 每局结果、有效样本数和有效步骤回报；引擎中断局单独标记 |
| `updates.jsonl` | 策略损失、价值损失、熵、KL、裁剪比例、采样数和更新次数 |
| `config.json` / `sessions.jsonl` | 初始配置 / 各次启动配置 |

`Ctrl+C` 释放按键并保留最后稳定 checkpoint，未完成采样或半次优化不会覆盖它。
续训从该完整更新恢复，尚未写入 checkpoint 的采样进度会丢弃；日志可能保留这些采样记录。
硬退出时 `last.pt` 也通过临时文件替换保护；不要用 `.tmp` 文件续训。
恢复训练前需将游戏回到航海起始界面。

PPO checkpoint 同样可交给推理入口：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u -m zaomeng_hanghai.run_policy runs/ppo/你的运行目录/last.pt
```

## 默认参数

| 参数 | 默认值 |
| --- | ---: |
| `--steps` | 100000 个额外有效步骤 |
| `--rollout-steps` | 2048 |
| `--update-epochs` | 10 |
| `--batch-size` | 256 |
| `--learning-rate` | 0.0001 |
| `--gamma` | 0.99 |
| `--gae-lambda` | 0.95 |
| `--clip-range` | 0.2 |
| `--entropy-coef` | 0.01 |
| `--value-coef` | 0.5，价值损失为 `0.5 × MSE` |
| `--max-grad-norm` | 0.5 |
| `--target-kl` | 0.03；超过其 1.5 倍时停止继续更新；0 禁用 |
| `--combat-timeout` | 60 秒 |
| `--device` | auto：有 CUDA 时使用 CUDA，否则 CPU |
| `--threads` | 4 |
| `--seed` | 新训练为 42，续训恢复存档随机状态 |

`--render` 显示环境调试窗口；`--hidden-sizes` 仅用于没有 `--init/--resume` 的新模型。

项目当前以规则专家引擎为主，PPO 相关源码保留，对应学习测试已移除。
现有引擎测试与手动诊断入口见 [测试说明](../tests/README.md)。
