# 造梦航海

游戏环境的观测和动作说明见 [环境文档](docs/env.md)，专家数据格式见
[专家数据文档](docs/expert-data.md)。
当前使用 [v2 的 86 维状态编码](docs/state-encoding.md)，旧 60 维数据和模型不能直接用于新版训练/推理。

寻路的 A* 核心通过 PyO3 使用 Rust 加速。安装项目（如 `uv sync`）时会自动编译，
需要 Rust 工具链；Windows 还需要 Visual Studio C++ Build Tools。
已存在虚拟环境时，可仅编译扩展，不更新其他项目依赖：

```powershell
uv pip install --python .venv/Scripts/python.exe "maturin>=1.15,<2"
$env:VIRTUAL_ENV = Join-Path $PWD '.venv'
.venv/Scripts/python.exe -m maturin develop --release --skip-install
```

修改 `rust/lib.rs` 后需重新执行编译命令。直接运行未编译的源码时自动使用 Python 实现。
验证扩展是否启用：

```powershell
.venv/Scripts/python.exe -c "from zaomeng_hanghai.core.maze_utils import _native_region_path; print(_native_region_path is not None)"
.venv/Scripts/python.exe -m unittest discover -s tests -p 'test_maze*.py'
.venv/Scripts/python.exe tests/benchmark_maze.py
```

Rust 搜索持有 GIL，直接借用 NumPy 数组，不复制输入；Python 仅在 dtype、连续布局不匹配
或需要屏蔽墙内目标时转换或复制。保留跨步逐格检查、危险代价及部分脱险规则。
搜索无时间限制：找到评分大于 0.8 的目标即返回，否则遍历可达节点选择最高分目标；
脱险时找到安全目标即返回。

所有路径搜索使用 `last_direction`（上次实际执行的移动方向）计算首步转向成本，
替代旧终点偏好。成本按 0°、45°、90°、135°、180° 依次为
`0`、`0.25 * turn_cost`、`0.5 * turn_cost`、`0.75 * turn_cost`、`turn_cost`，
只计入第一步，不随搜索步长放大。默认脱险 `turn_cost=1`，射击位置和出口导航为 `2`。
停留、重置或切关后清空方向记录；调试预览不改变记录。

使用专家数据预训练 PPO 的策略网络（当前仅行为克隆，不使用奖励函数）：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u -m zaomeng_hanghai.train.train_bc
```

训练参数、checkpoint 与游戏推理命令见 [模仿学习说明](docs/imitation-learning.md)。

在线 PPO 训练、断点续训和仅在奖励变化时打印日志见 [PPO 训练说明](docs/ppo-training.md)：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u -m zaomeng_hanghai.train.train_ppo --init runs/bc/你的运行目录/best.pt --steps 100000
```
