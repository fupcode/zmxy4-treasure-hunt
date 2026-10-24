# 专家数据位置与格式

## 采集命令与保存位置

在项目根目录运行，游戏需停在航海起始界面且保持无遮挡：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u -m zaomeng_hanghai.collect_expert --episodes 100
```

默认保存到项目根目录下的 `data/expert/<日期_时间_随机后缀>/`，例如：

```text
D:\project\personal\zaomeng\zaomeng-hanghai\data\expert\20261024_160000_ab12cd34\
```

上面的运行目录名仅为示例。实际路径会在启动后以 `Expert data directory:` 输出。
`--output data/expert/my_run` 可以指定新的保存目录；已有目录不会被覆盖，也不支持从该目录续跑。
`data/expert/` 已加入 `.gitignore`，数据不会默认提交到 Git。

常用选项：

| 选项 | 默认值 | 说明 |
| --- | --- | --- |
| `--episodes` | `100` | 尝试局数，包含死亡、超时和被过滤的局，并非保留局数或成功局数 |
| `--output` | 自动创建运行目录 | 保存位置，相对路径以当前工作目录为基准 |
| `--render` | 不开启 | 加上此选项显示调试窗口 |
| `--max-episode-seconds` | `300` | 在主循环检查的整局超时预算；阻塞中的单次调用可能使实际结束时间超过预算 |

## 采集范围与过滤

- 只采集 `level=1..4` 的开局动作（`start`）和战斗动作（`combat`）。
- 不保存 `try_next_level()` 内的拾取、寻找出口等导航动作，也不保存 `level=0` 的起始区域动作。
- 在 `take_action_direction(direction)` 入口、开火和修改移动按键之前复制观测，并将传入的目标方向转换为动作编号。
- 若整局以死亡 `status=-2` 结束且 `terminal_level=1`，丢弃该局所有样本。
- 采集异常或没有有效样本的局也不会生成 NPZ。
- 第 2–4 关死亡或超时的局，只要有样本且没有采集异常，仍保留并标记结果；不要求整局通关。
- `Ctrl+C` 会保留已完成且符合规则的局；当前未完成局按采集异常丢弃。

## 运行目录结构

```text
<run>/
├── config.json           # 格式版本、动作/阶段映射、目标局数、字段语义
├── progress.json         # 当前运行状态和累计统计
├── manifest.jsonl        # 每个已结束的局一行，包括被丢弃的局
├── episode_0002.npz      # 每个保留局一个压缩 NumPy 文件
└── episode_0005.npz
```

局编号从 `1` 开始，文件编号可能不连续，因为被丢弃的局没有 NPZ 文件。
写入期间可能短暂出现 `.tmp` 文件，读取数据时只使用清单中 `kept=true` 的文件。
采集进程异常退出时，临时文件不应作为训练数据。

## NPZ 字段

所有字段均可使用 `numpy.load(path, allow_pickle=False)` 读取。设该局保留的样本数为 `N`：

| 字段 | shape | dtype | 含义 |
| --- | --- | --- | --- |
| `obs` | `(N, 60)` | `float32` | 动作入口处的观测快照 |
| `acts` | `(N,)` | `int64` | 专家目标移动动作，范围 `0..8` |
| `phases` | `(N,)` | `int8` | `0=start`，`1=combat` |
| `levels` | `(N,)` | `int8` | 当前第几关，范围 `1..4`，未归一化 |
| `tasks` | `(N,)` | `int8` | 关卡类型，范围 `0..8`，未做 one-hot |
| `map_indices` | `(N,)` | `int8` | 大地图编号，范围 `0..6` |
| `segments` | `(N,)` | `int32` | 局内片段编号，从 `0` 开始 |
| `timestamps` | `(N,)` | `float64` | 从本局开始到动作入口的秒数，使用单调时钟，不是日期时间 |
| `dt` | `(N,)` | `float32` | 从当前动作入口到下一次动作入口或本局结束的秒数 |
| `episode_id` | `()` | `int32` | 本次运行内的局编号 |
| `status` | `()` | `int8` | 本局结束状态码 |
| `terminal_level` | `()` | `int8` | 本局结束时的关卡编号 |

`dt` 的下一次动作可以是不保存的导航动作；它不是到“下一条保留样本”的时间差，
也不等于 `fire()` 或按键函数自身的耗时。转向等待、观测更新、显式停留等会影响这个值。

阶段、大地图、关卡发生变化，或中间出现不保存的动作后，下一条保存样本会使用新 `segment`。
片段编号只在同一局内有意义，跨文件使用时应同时保留运行目录和 `episode_id`。

这些数据是行为克隆的 `(obs, acts)` 样本，**不包含 `reward`、`next_obs` 或逐步 `done`**。
导航阶段被省略，不能直接把相邻两条观测视为连续强化学习转移；即便在同一片段，仍需注意实际 `dt`。

### 60 维观测布局

以下切片采用 Python 的左闭右开规则：

| 索引 | 含义 |
| --- | --- |
| `0:2` | 角色归一化位置 `[x, y]` |
| `2` | 剩余开火冷却除以 `0.6`，裁剪到 `[0, 1]` |
| `3` | HP 除以 `4`，裁剪到 `[0, 1]` |
| `4:7` | `pao、dan、fan` 三种道具的拥有标志，取 `0/1` |
| `7` | 当前关卡 `level / 4` |
| `8:17` | 9 种关卡类型 one-hot |
| `17:32` | 最近 5 个敌人，每个 `[存在标志, x, y]`，共 15 维 |
| `32:60` | 最近 4 颗危险弹幕，每颗 `[存在标志, 右, 下, 左, 上, x, y]`，共 28 维 |

敌人和弹幕的位置为绝对归一化坐标，不是相对角色的坐标。不足固定数量时补零，超出时保留最近的实体。
弹幕方向使用四维 one-hot；敌人不编码方向和速度。
关卡类型 one-hot 表示地图类型编号，观测中没有直接存储障碍网格。

### 动作编号

| `acts` | 目标方向 | 按键 |
| --- | --- | --- |
| `0` | 停止移动 | 无 |
| `1` | 右 | D |
| `2` | 下 | S |
| `3` | 左 | A |
| `4` | 上 | W |
| `5` | 右下 | SD |
| `6` | 左下 | SA |
| `7` | 右上 | WD |
| `8` | 左上 | WA |

动作标签是目标移动方向，不是寻路策略编号 `move_index`，也不包含开火动作。
开火由公共控制器执行，包括开火所需的临时转向。

## 状态码与 JSON 文件

| 状态码 | 含义 |
| --- | --- |
| `1` | 通关成功 |
| `-2` | 死亡 |
| `-3` | 超时，包括过关/开局接管超时或整局时间预算耗尽 |
| `-4` | 采集异常或中断，仅记入清单，不保存当前局 NPZ |

`config.json` 当前 `schema_version=1`，声明 60 维观测、动作和阶段映射，以及采集过滤规则。

`progress.json` 包含：

- `state`：`created / running / completed / failed / interrupted`。
- `target_episodes、finished_episodes、active_episode`：目标局数、已结束局数和进行中的局编号。
- `kept_episodes、discarded_episodes、successful_episodes、saved_samples`：累计保留、丢弃、成功局数及保存样本数。
- `error`：错误信息，无错误时为 `null`。

`manifest.jsonl` 每行是一个独立 JSON 对象，字段包括：

- `episode_id、status、success、terminal_level、max_recorded_level`：局编号和结果。
- `kept、discard_reason、file`：是否保留、丢弃原因、相对运行目录的 NPZ 文件名。
- `sample_count、start_samples、combat_samples、duration_seconds`：该局采集到的数量和总耗时。
- `error`：采集异常说明。

`discard_reason` 为 `died_at_level_1 / no_samples / collection_error`，保留局为 `null`。
被丢弃局的 `sample_count` 仍表示丢弃前采集到的样本数，不代表已落盘的训练样本数。

## 读取示例

在项目根目录运行以下 Python 代码，将目录替换为实际运行目录：

```python
import json
from pathlib import Path

import numpy as np

run_dir = Path("data/expert/实际运行目录")
episodes = [
    json.loads(line)
    for line in (run_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]

observations, actions = [], []
for episode in episodes:
    if not episode["kept"]:
        continue
    with np.load(run_dir / episode["file"], allow_pickle=False) as data:
        observations.append(data["obs"].copy())
        actions.append(data["acts"].copy())
        print(episode["episode_id"], data["obs"].shape, int(data["status"].item()))

if observations:
    obs = np.concatenate(observations)  # (总样本数, 60)
    acts = np.concatenate(actions)     # (总样本数,)
```

训练集/验证集建议按整局划分，避免把同一局中高度相似的相邻帧随机分到两边。
暂不使用当前采集进程正在写入的局，也不根据文件编号是否连续来判断数据损坏。
