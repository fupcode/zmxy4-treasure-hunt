# 造梦西游4寻宝助手

项目发行包名为 `zmxy4-treasure-hunt`，Python 导入包为 `zmxy4_treasure_hunt`。
航海寻宝位于 `voyage` 子包，后续迷宫寻宝使用 `maze` 子包。
共用 Rust 原生扩展为 `zmxy4_treasure_hunt._native`。

基于 YOLO 目标识别和规则寻路实现自动航海。运行方式、状态码和调试说明见
[环境文档](docs/env.md)，目标识别模型训练见 [YOLO 文档](yolo/README.md)。

`uv sync` 仅安装主程序依赖；训练或导出模型时使用 `uv sync --group train`，
运行训练和导出脚本时使用 `uv run --group train python ...`。

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
.venv/Scripts/python.exe -c "from zmxy4_treasure_hunt.voyage.core.maze_utils import _native_region_path; print(_native_region_path is not None)"
.venv/Scripts/python.exe -m unittest discover -s tests/voyage -p 'test_maze*.py'
.venv/Scripts/python.exe tests/benchmark_maze.py
```

Rust 搜索持有 GIL，直接借用 NumPy 数组，不复制输入；Python 仅在 dtype、连续布局不匹配
或需要屏蔽墙内目标时转换或复制。保留跨步逐格检查、危险代价及部分脱险规则。
搜索无时间限制：找到评分大于 0.8 的目标即返回，否则遍历可达节点选择最高分目标；
脱险时找到安全目标即返回。

所有路径搜索使用 `last_direction`（上次实际执行的移动方向）计算首步转向成本，
替代旧终点偏好。成本按 0°、45°、90°、135°、180° 依次为
`0`、`0.25 * turn_cost`、`0.5 * turn_cost`、`0.75 * turn_cost`、`turn_cost`，
只计入第一步，不随搜索步长放大。默认脱险 `turn_cost=4`，射击位置和出口导航为 `2`。
停留、重置或切关后清空方向记录；调试预览不改变记录。
