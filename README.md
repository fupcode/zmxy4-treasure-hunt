# 造梦西游4寻宝助手

支持 Windows 上的迷宫寻宝和航海寻宝：迷宫寻宝使用内置的每周地图规划路线，航海寻宝基于 YOLO 目标识别和规则寻路自动操作。

## 下载与使用

1. 打开 [Releases](https://github.com/fupcode/zmxy4-treasure-hunt/releases)，下载对应版本附件中的 Windows 程序压缩包。
2. 将压缩包完整解压到本地目录，保留随包附带的文件和目录。
3. 进入游戏，将画质设为最高，并保持游戏画面完整可见、无遮挡。迷宫寻宝需将游戏窗口放在主显示器上；航海寻宝需先进入出现船与海的航海界面。
4. 双击程序，在窗口中选择“航海寻宝”或“迷宫寻宝”。航海胜利次数默认为 3 次，点击“开始”运行。

“寻宝时最小化窗口”默认勾选，开始后助手会自动最小化，避免遮挡游戏；取消勾选可保持窗口显示。运行日志显示在助手窗口中。需要停止时，从任务栏恢复助手并点击“停止”；关闭助手也会停止任务。找不到游戏窗口时会弹出提示，可调整游戏画面后重试。

发布版无需单独安装 Python、Rust 或源码依赖。

首次打开默认选择迷宫寻宝。点击“开始”时，会将模式、航海胜利次数和最小化选项保存到 exe 同目录的 `settings.json`，下次打开自动恢复；源码运行时保存在 `main.py` 同目录。配置缺失或损坏时使用默认值。

### 命令行使用

在解压目录打开 PowerShell。以下以 `造梦西游4寻宝助手.exe` 为例；若下载版本的文件名不同，请替换为实际名称。

```powershell
# 打开图形界面
.\造梦西游4寻宝助手.exe

# 直接开始迷宫寻宝
.\造梦西游4寻宝助手.exe maze

# 航海寻宝，直到胜利 5 次（省略次数时默认 3 次）
.\造梦西游4寻宝助手.exe voyage 5

# 查看入口及子命令帮助
.\造梦西游4寻宝助手.exe --help
.\造梦西游4寻宝助手.exe maze --help
.\造梦西游4寻宝助手.exe voyage --help
```

`maze`、`voyage` 后的参数由对应子命令解析。命令行模式在终端按 `Ctrl+C` 退出。

| 模式 | 参数 | 说明 |
| --- | --- | --- |
| 迷宫寻宝 | `--interval 0.05` | 最小循环间隔，单位秒，默认 0.05，必须为正数 |
| 迷宫寻宝 | `--debug` | 显示监控图，识别或定位失败时保存诊断现场；仍会自动操作游戏 |
| 航海寻宝 | `次数` | 目标胜利次数，必须为正整数，默认 3 |
| 航海寻宝 | `--debug` | 显示监控窗口，仍会自动操作游戏 |
| 航海寻宝 | `--read-only` | 只更新感知和建议路径，由用户手动操作游戏；可搭配 `--debug` 显示窗口 |
| 航海寻宝 | `--map-index 0` | 只读模式下指定地图编号，范围 0–6，默认从画面识别 |
| 航海寻宝 | `--level 1` | 只读模式下的起始关卡，范围 0–4，默认 0 |

```powershell
# 迷宫寻宝：显示监控图并启用失败诊断
.\造梦西游4寻宝助手.exe maze --debug

# 航海寻宝：只读调试指定地图和关卡
.\造梦西游4寻宝助手.exe voyage --read-only --debug --map-index 0 --level 1
```

迷宫诊断现场保存在当前运行目录的 `runs/maze-debug/` 下。识别、定位和诊断说明见
[迷宫文档](docs/maze.md)；航海流程、状态码与调试说明见 [航海文档](docs/env.md)。

## 进阶：从源码配置与运行

### 环境准备

源码运行需要 Windows、Python 3.12 或更新版本，以及 `uv`。
项目通过 PyO3 编译 Rust 寻路扩展，安装时还需要 Rust 工具链和 Visual Studio C++ Build Tools。

获取源码后，在项目根目录安装依赖：

```powershell
git clone https://github.com/fupcode/zmxy4-treasure-hunt.git
cd zmxy4-treasure-hunt

# 仅安装运行依赖，并编译 Rust 扩展
uv sync
```

航海使用项目内的 `weights/best.onnx` 模型；请保留该文件，并在项目根目录运行。
迷宫的七张地图和遮罩素材随 Python 包提供，无需另行配置。

### 运行入口

源码与发布版使用相同的子命令和参数，将 exe 替换为 `python main.py` 即可：

```powershell
.\.venv\Scripts\python.exe main.py
.\.venv\Scripts\python.exe main.py maze
.\.venv\Scripts\python.exe main.py voyage 3
.\.venv\Scripts\python.exe main.py maze --debug --interval 0.05
.\.venv\Scripts\python.exe main.py voyage --read-only --debug --map-index 0 --level 1
```

两个子命令也提供独立的模块入口：

```powershell
.\.venv\Scripts\python.exe -m zmxy4_treasure_hunt.maze.cli --help
.\.venv\Scripts\python.exe -m zmxy4_treasure_hunt.voyage.cli --help
```

### 开发与测试

`uv sync` 默认只安装运行依赖。诊断绘图使用的 `matplotlib` 放在 `debug` 组中，需要时通过 `uv sync --group debug` 安装。
自动化测试和手动检查工具说明见 [测试文档](tests/README.md)。

```powershell
uv sync
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py'
```

修改 `rust/lib.rs` 后，需要重新编译扩展。已有虚拟环境时，可仅编译扩展，不更新其他项目依赖：

```powershell
uv pip install --python .venv/Scripts/python.exe "maturin>=1.15,<2"
$env:VIRTUAL_ENV = Join-Path $PWD '.venv'
.venv/Scripts/python.exe -m maturin develop --release --skip-install
```

直接运行未编译扩展的源码时，寻路自动使用 Python 实现。可检查 Rust 扩展是否启用，并运行性能基准：

```powershell
.venv/Scripts/python.exe -c "from zmxy4_treasure_hunt.voyage.core.maze_utils import _native_region_path; print(_native_region_path is not None)"
.venv/Scripts/python.exe tools/voyage_benchmark_maze.py
```

训练或导出模型时，使用 `uv sync --group train` 安装额外依赖，再通过
`uv run --group train python ...` 运行脚本。数据与模型导出说明见 [YOLO 文档](yolo/README.md)。

### 代码结构

| 路径 | 职责 |
| --- | --- |
| `main.py` | 图形界面与统一命令行入口 |
| `src/zmxy4_treasure_hunt/maze/` | 迷宫寻宝、每周地图、识别与诊断 |
| `src/zmxy4_treasure_hunt/voyage/` | 航海寻宝、目标识别与自动操作 |
| `rust/` | Rust 寻路扩展 |
| `tests/` | 自动化测试 |
| `tools/` | 以 `maze_`、`voyage_` 为前缀的诊断与手动检查工具 |

项目发行包名为 `zmxy4-treasure-hunt`，Python 导入包为 `zmxy4_treasure_hunt`，
共用 Rust 原生扩展为 `zmxy4_treasure_hunt._native`。
