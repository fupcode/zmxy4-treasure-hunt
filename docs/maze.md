# 迷宫寻宝

在项目根目录运行（启动后立即开始）：

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m zmxy4_treasure_hunt.maze.ft
```

监控窗口按 Esc、终端按 Ctrl+C 退出，退出时松开 WASD 并释放截图设备。

截图使用 DXcam 的 BGR 输出，DXGI 初始化失败时尝试 WinRT。目前使用主显示器；
游戏完整画面应位于主显示器内，区域坐标为该显示器内的物理像素坐标。
启动时根据金色外框、矩形四角和暗色直边自动定位游戏区域。可调整循环间隔：

```powershell
.\.venv\Scripts\python.exe -m zmxy4_treasure_hunt.maze.ft --interval 0.05
```

`--interval` 是包含处理时间的最小循环间隔，默认 0.05 秒；截图单次等待上限为 5 秒。

启动时按北京时间的星期加载包内 `maze/maps/0.json` 至 `maze/maps/6.json`，星期日为 0、星期一为 1。
程序直接使用完整地图，沿最短路径前往最近的剩余宝箱，不再探索或拼接未知区域。
地图缺失、格式错误时直接报错；格式与地图来源见 [地图说明](../src/zmxy4_treasure_hunt/maze/maps/README.md)。

## 模块职责

| 文件 | 职责 |
| --- | --- |
| `ft.py` | 命令行入口、截图 → 识别 → 规划 → 执行、过关与监控窗口 |
| `capture.py` | BGR 截图、超时、后端回退和释放 |
| `controls.py` | 鼠标点击、方向键变化和统一松键 |
| `vision.py` | 人物、宝箱、墙壁线条及窗口区域识别 |
| `viewport.py` | 金色外框颜色、四角和暗色直边联合定位视区 |
| `mask.py` | 原始黑色 Alpha 遮罩反算、缓存增益与可信像素区域 |
| `grid.py` | 格距标定、补线、局部格网和人物偏移 |
| `map_templates.py` | 按北京时间选择星期、读取并校验完整地图 |
| `navigation.py` | 在完整地图中定位、BFS 宝箱最短路径和移动建议 |

识别和导航不创建窗口或发送按键。`FindTreasure.step()` 执行一帧；
`NavigationResult` 包含用于显示的地图、路径、移动建议和接近宝箱标记。
调用者应在完成使用后调用 `FindTreasure.close()`；`main()` 会自动清理。
`FindTreasure` 从 `ft` 导入，`Graph` 和 `Maze` 从 `navigation` 导入。
`Maze` 为纯规划接口，不隐式发送按键或以 `None` 表示接近宝箱。

人物起点在程序中初始化为 `(1, 1)`，不从地图文件读取。人物黑色特征通过凸包定位，墙线用于标定格距。
首帧根据起点和画面中的人物格确定视图原点，并用完整地图校验；后续帧结合近期实际移动方向跟踪视图。
宝箱是否可见不会改变完整地图中的候选位置，只有触发假宝箱后才移除本局目标。
监控窗口从启动时就显示完整地图和实时人物坐标。

定位结果有歧义、当前画面与地图不匹配或剩余宝箱不可达时松键报错，
不会猜测起点、覆盖地图墙壁或切回探索。墙线检测过滤宝箱纹理造成的短直线干扰。
格距校准、人物掩膜和颜色阈值仍需用实际游戏画面校验。

定位视区后，按实际宽高缩放原始 `620×344` 遮罩，缓存透光率 `T=1-alpha` 和增益。
每帧先反算 `恢复图像=截图/T`，再识别人物和墙线。素材位于 `maze/assets/edge_mask.png`，
来自原 SWF，只读取 Alpha，不依赖外部反编译目录。
显示反算阈值为 `T>=0.03`；识别使用更严格的 `T>=0.15`，极暗边缘不参与人物、宝箱或墙线识别。
格网连接格的可信像素不足 60% 时只标为本帧不可观测，不修改完整地图，也不当作墙壁。
完全不透明区域无法恢复；反算后的极暗边缘仍可能有量化和对齐噪声。
这些处理作用于程序收到的截图，不修改游戏窗口本身的渲染。

顶部暗场表示触发假宝箱：立即松键，等待 5 秒（覆盖约 3 秒黑屏）后继续导航。
暗场检测继续使用未提亮的原始截图，额外黑屏不作为边缘遮罩处理。
假宝箱触发后消失，对应格子恢复为普通道路；保留完整地图、人物位置、格距和移动历史，
不点击旧版过关按钮，也不回到游戏起始点。仅修改内存中的本局状态，JSON 模板保持原样。
所有候选宝箱都处理完毕后停止。

窗口定位不再匹配精确的 `(1, 1, 1)` 像素，也不依赖 WASD 模板。
先用多档暗色阈值寻找接近 `620:344` 的矩形轮廓，再检查轴对齐四角、四侧金色区域、
直边连续性和 Shi–Tomasi 角点支持。多个不同位置得分接近时拒绝定位。
该方法要求完整边框可见；严重遮挡、极小缩放或 UI 边框改版仍可能失败。
定位评分表示规则满足程度，不是成功率估计。

可使用真实截图离线复核，不启动游戏或发送按键：

```powershell
.\.venv\Scripts\python.exe -X utf8 tools/check_maze_viewport.py images/maze/full.png
```

输出位于 `images/maze/viewport_debug/`：`marked.png` 为矩形/四角标注，
`viewport.png` 为原始裁剪视区，`viewport_restored.png` 为反算图，
`recognition_mask.png` 为可信识别区域，`comparison.png` 为左右对照图，
`screenshot_restored.png` 为恢复后贴回的完整截图，`report.json` 为位置、评分和处理区域比例。

## 离线回归

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m unittest discover -s tests/maze -v
```

测试使用合成图像和模拟截图、输入设备，覆盖人物/墙壁/宝箱/窗口定位、
横纵独立格距、全局定位与视图跟踪、七张地图的宝箱路线、按键变化和错误清理。
本地存在 `images/maze/ft*.png` 时，还会验证七张原图的墙线识别与定位（使用预设人物坐标）。
不连接游戏，不发送真实鼠标或键盘操作。

## PyInstaller 打包

七张地图位于 `src/zmxy4_treasure_hunt/maze/maps/`，遮罩位于 `maze/assets/`。
地图和遮罩都通过 `importlib.resources` 读取，不依赖启动目录，不需要在 exe 旁另放 `maps/`。
数据是只读模板，运行中只修改内存。

独立入口为项目根目录 `maze_main.py`。在项目根目录执行：

```powershell
uv pip install --python .venv/Scripts/python.exe "pyinstaller>=6"
.\.venv\Scripts\python.exe -m PyInstaller --noconfirm --onefile --name maze-hunt --paths src --additional-hooks-dir src/zmxy4_treasure_hunt/_pyinstaller/hooks --distpath dist/maze --workpath build/maze --specpath build/maze maze_main.py
```

生成 `dist/maze/maze-hunt.exe`。将 `--onefile` 换成 `--onedir` 可生成目录版。
项目安装后的包元数据也会注册 hook；上述命令显式指定 hook 目录，方便直接从源码打包。
该 hook 只收集七张地图 JSON 和遮罩 PNG。机制参考 [PyInstaller 官方 hook 文档](https://pyinstaller.org/en/stable/hooks.html#providing-pyinstaller-hooks-with-your-package)。

纯资源验收入口为 `tools/check_maze_resources.py`，不会截图或发送游戏操作。
可以按相同参数打包这个入口，在没有项目 `maps/` 的工作目录运行 exe，验证资源确实来自包内。
