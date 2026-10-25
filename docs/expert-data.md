# 专家数据采集与格式

每次调用 `Env.auto_run()` 都会从 `reset()` 开始运行一局专家，并记录该局的开局和战斗动作。
`auto_loop()`、`main.py` 和环境模块的自动模式均使用此入口，无需额外采集脚本。
只读 `--debug`、策略推理和 PPO 的 `step()` 不会采集专家数据。

## 启动与保存位置

在项目根目录运行，游戏停在航海起始界面且保持无遮挡：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u -m zaomeng_hanghai.env 3
```

`3` 表示目标成功次数，不是尝试局数；死亡或超时后恢复并继续，直到成功次数达到目标。
默认开启调试窗口。也可以运行 `main.py`，通过交互选择游戏窗口和目标成功次数。

默认目录为项目根目录下的 `data/expert/<日期_时间_随机后缀>/`，启动时输出
`Expert data directory:`。同一次 `auto_loop()` 的所有尝试写入一个运行目录。
`data/expert/` 已被 Git 忽略。

指定一个尚不存在的目录：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u -m zaomeng_hanghai.env 3 --expert-output data/expert/my_run
```

已有目录不会覆盖，也不支持追加续跑。环境模块启动时会切换至项目根目录，相对输出路径以项目根目录为基准。
Python API 中的相对输出路径以当前工作目录为基准：

```python
from zaomeng_hanghai.env import Env, auto_loop

env = Env(render_enabled=False, combat_timeout=120)
env.init_window()
status = env.auto_run()  # 单局；每次独立调用默认创建一个新目录
print(env.expert_data_path)
# 或：auto_loop(env, target=3, expert_output="data/expert/my_run")
```

`auto_run()` 返回终局状态码；单局 API 不负责恢复游戏，下一局前需自行调用
`env.game_window.recover(status)`。`auto_loop()` 会负责恢复。
默认单关战斗时限沿用 `Env` 的设置（当前默认 60 秒），可在构造环境时自定义。
旧文档中的 `zaomeng_hanghai.collect_expert`、`--episodes`、`--max-episode-seconds`
不是当前入口；当前没有额外的整局时间预算。

## 采集范围和过滤

- 只保存 `level=1..4` 的 `start` 和 `combat` 动作。
- 在专家动作入口、开火和修改移动按键之前复制 `state_encode()`，保存目标移动方向。
- `try_next_level()` 的拾取、寻找出口等导航动作与 `level=0` 动作不保存。
- 动作编号不包含公共控制器的开火或临时转向，也不是寻路策略编号 `move_index`。
- 第一关死亡的整局丢弃；第 2–4 关死亡、单关战斗超时的局，只要有有效样本就保留。
- 引擎故障、异常、Ctrl+C 中断或没有有效样本的局不生成 NPZ，但写入清单。
- Ctrl+C 后已完成并保留的局仍可训练，当前未完成局丢弃。

## 文件结构

```text
<run>/
├── config.json
├── progress.json
├── manifest.jsonl
├── episode_0001.npz
└── episode_0003.npz
```

局编号从 1 开始；被丢弃的局无 NPZ，因此文件编号可能不连续。
配置、进度和 NPZ 先写 `.tmp` 再替换；NPZ 完成后才追加清单。
读取时只使用 `manifest.jsonl` 中 `kept=true` 的文件；忽略临时文件和未被清单引用的文件。
进程被强制终止时，当前局可能只有进度记录，没有清单条目。

`config.json` 声明 `schema_version=4`、`observation_schema_version=4`、
`observation_dim=96`、动作顺序、阶段映射、战斗时限和过滤规则。
当前观测布局详见 [状态编码 v4](state-encoding.md)。旧 60/81 维数据不兼容，
不能通过修改元数据转换；建议为新数据使用单独的训练根目录，避免混入旧目录。

## NPZ 字段

使用 `numpy.load(path, allow_pickle=False)`。设本局样本数为 `N`：

| 字段 | shape | dtype | 含义 |
| --- | --- | --- | --- |
| `obs` | `(N, 96)` | `float32` | 动作执行前的观测快照 |
| `acts` | `(N,)` | `int64` | 目标移动动作，范围 0..8 |
| `phases` | `(N,)` | `int8` | 0=start，1=combat |
| `levels` | `(N,)` | `int8` | 关卡编号，1..4 |
| `tasks` | `(N,)` | `int8` | 关卡类型，0..8 |
| `map_indices` | `(N,)` | `int8` | 大地图编号，0..6 |
| `segments` | `(N,)` | `int32` | 局内片段编号，从 0 开始 |
| `timestamps` | `(N,)` | `float64` | 从本局 reset 前到动作入口的单调时钟秒数 |
| `dt` | `(N,)` | `float32` | 到下一次专家动作入口或本局结束的秒数 |
| `episode_id` | `()` | `int32` | 运行内的局编号 |
| `status` | `()` | `int8` | 终局状态码 |
| `terminal_level` | `()` | `int8` | 结束时关卡编号 |

`dt` 的下一次动作可以是不保存的导航动作，不是到下一条保留样本的时间差。
开火、转向等待、观测更新与显式停留均可能影响 `dt`。
阶段、关卡、大地图或任务发生变化，或中间出现不保存的动作后，会开始新片段。
片段编号只在同一局内有意义。

这些是行为克隆 `(obs, acts)` 数据，不包含 `reward`、`next_obs` 或逐步 `done`。
不能直接将相邻样本作为连续强化学习转移。

### 动作编号

| acts | 目标方向 | 按键 |
| --- | --- | --- |
| 0 | 停止移动 | 无 |
| 1 | 右 | D |
| 2 | 下 | S |
| 3 | 左 | A |
| 4 | 上 | W |
| 5 | 右下 | SD |
| 6 | 左下 | SA |
| 7 | 右上 | WD |
| 8 | 左上 | WA |

## 状态与清单

| 状态码 | 含义 | 是否保留 |
| --- | --- | --- |
| 1 | 通关成功 | 有样本则保留 |
| -2 | 死亡 | 第一关丢弃；其余有样本则保留 |
| -3 | 单关累计战斗超时 | 有样本则保留 |
| -4 | 加载、导航等引擎故障，或采集异常、中断 | 丢弃 |

战斗计时在空场暂停，加载不计时，进入下一关重置。

`manifest.jsonl` 每行对应一个已结束的尝试，包含：

- `episode_id、status、success、terminal_level、max_recorded_level`：编号和结果。
- `kept、discard_reason、file`：是否保留、丢弃原因和相对 NPZ 文件名。
- `sample_count、start_samples、combat_samples、duration_seconds`：采样数量和整局耗时。
- `error`：异常信息，无异常为 null。

`discard_reason` 为 `died_at_level_1 / no_samples / collection_error`；保留局为 null。
被丢弃局的样本数量表示丢弃前的数量，不代表已保存的训练样本。

`progress.json` 包含 `state`（created/running/completed/failed/interrupted）、
`finished_episodes、active_episode、kept_episodes、discarded_episodes、successful_episodes、saved_samples、error`。
`completed` 表示本次调用正常结束，不表示所有尝试均通关；运行目标由 `auto_loop(target=...)` 控制。

## 读取与训练

```python
from zaomeng_hanghai.train.expert_data import load_episodes, concatenate

episodes = load_episodes("data/expert/实际运行目录")
obs, acts = concatenate(episodes)
print(obs.shape, acts.shape)  # (总样本数, 96), (总样本数,)
```

加载器会校验 v4 配置、有限观测值、动作范围及清单样本数，并跳过被过滤的局。
训练集和验证集按整局划分，避免相邻帧泄漏：

```powershell
.\.venv\Scripts\python.exe -m zaomeng_hanghai.train.train_bc --data data/expert/实际运行目录
```

默认验证集需要至少两局保留数据；只有一局时可显式传入 `--validation-fraction 0`。
