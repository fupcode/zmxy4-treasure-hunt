# 迷宫寻宝

在项目根目录运行（启动后立即开始）：

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe main.py maze
```

默认不显示监控窗口、不保存诊断图片。终端按 Ctrl+C 退出，退出时松开 WASD 并释放截图设备。
调试时使用 `--debug`，显示左上角监控图，识别或定位失败时保存现场；监控窗口支持 Esc 退出。
调试模式仍会执行正常的游戏操作。

```powershell
.\.venv\Scripts\python.exe main.py maze --debug
# exe：.\dist\maze\maze-hunt.exe maze --debug
```

截图使用 DXcam 的 BGR 输出，DXGI 初始化失败时尝试 WinRT。目前使用主显示器；
游戏完整画面应位于主显示器内，区域坐标为该显示器内的物理像素坐标。
启动时根据金色外框、矩形四角和暗色直边自动定位游戏区域。可调整循环间隔：

```powershell
.\.venv\Scripts\python.exe main.py maze --interval 0.05
```

`--interval` 是包含处理时间的最小循环间隔，默认 0.05 秒；截图单次等待上限为 5 秒。

启动时按北京时间的星期加载包内 `maze/maps/0.json` 至 `maze/maps/6.json`，星期日为 0、星期一为 1。
程序直接使用完整地图，沿最短路径前往最近的剩余宝箱，不再探索或拼接未知区域。
地图缺失、格式错误时直接报错；格式与地图来源见 [地图说明](../src/zmxy4_treasure_hunt/maze/maps/README.md)。

## 模块职责

| 目录/入口 | 职责 |
| --- | --- |
| `cli.py` | 命令行参数解析与启动 |
| `ft.py` | `FindTreasure` 自动运行流程 |
| `runtime/` | `capture.py` 负责截图，`controls.py` 负责 Windows 输入 |
| `vision/` | `scene.py` 识别人物和墙壁，`ring.py` 拟合黑圈，`viewport.py` 定位视区，`mask.py` 反算遮罩 |
| `navigation/` | `maze.py` 定位与寻路，`grid.py` 重建局部格网，`templates.py` 加载每周地图 |
| `debug/` | `diagnostics.py` 导出失败现场、比较七张地图 |
| `maps/`、`assets/` | 内置地图 JSON 和原始 Alpha 遮罩素材 |

`maze.vision` 和 `maze.navigation` 的 `__init__.py` 提供公开接口；具体实现放在子模块中。
统一入口为 `python main.py maze`，也可使用 `python -m zmxy4_treasure_hunt.maze.cli`。

识别和导航不创建窗口或发送按键。`FindTreasure.step()` 执行一帧；
`NavigationResult` 包含用于显示的地图、路径、移动建议和接近宝箱标记。
调用者应在完成使用后调用 `FindTreasure.close()`；CLI 启动的主循环会自动清理。
`FindTreasure` 从 `ft` 导入，`Graph` 和 `Maze` 从 `navigation` 导入。
`Maze` 为纯规划接口，不隐式发送按键或以 `None` 表示接近宝箱。

人物起点在程序中初始化为 `(1, 1)`，不从地图文件读取。人物通过周围可见的黑色圆弧定位，墙线用于标定格距。
首帧根据起点和画面中的人物格确定视图原点，并用完整地图校验；后续帧在上一视图原点附近横纵各 ±2 格内匹配，兼容摄像机缓动和首条可见格线变化。
宝箱是否可见不会改变完整地图中的候选位置，只有触发假宝箱后才移除本局目标。
仅 `--debug` 模式显示完整地图和实时人物坐标。

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
.\.venv\Scripts\python.exe -X utf8 tools/maze_check_viewport.py images/maze/full.png
```

输出位于 `images/maze/viewport_debug/`：`marked.png` 为矩形/四角标注，
`viewport.png` 为原始裁剪视区，`viewport_restored.png` 为反算图，
`recognition_mask.png` 为可信识别区域，`comparison.png` 为左右对照图，
`screenshot_restored.png` 为恢复后贴回的完整截图，`report.json` 为位置、评分和处理区域比例。

## 离线回归

四个人物样例的黑圈识别鲁棒性可单独测试：

```powershell
.\.venv\Scripts\python.exe -X utf8 tools/maze_check_character_robustness.py
```

输入为已裁剪的 `images/maze/ch2.png`、`ch5.png`、`ch6.png`、`ch7.png`，每张测试 40 组：
重点覆盖 0.25～2 倍等比例缩放及多种缩放插值；另覆盖亮度、黑电平、Gamma、JPEG、噪声和遮罩错位，不添加额外模糊。依据人工人物标注，检查中心误差、
人物掩膜覆盖率和半径；不是仅检查识别函数是否返回坐标。
输出在 `runs/maze-character-check/`，包括 JSON/Markdown 报告及原图识别标注。
返回码 1 表示存在未通过的鲁棒性样例；不修改识别算法、不控制游戏、不构建 exe。

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m unittest discover -s tests/maze -v
```

测试使用合成图像和模拟截图、输入设备，覆盖人物/墙壁/宝箱/窗口定位、
横纵独立格距、全局定位与视图跟踪、七张地图的宝箱路线、按键变化和错误清理。
本地存在 `images/maze/ft*.png` 时，还会验证七张原图的墙线识别与定位（使用预设人物坐标）。
不连接游戏，不发送真实鼠标或键盘操作。

## 定位失败诊断

开启 `--debug` 后，识别或地图定位失败时，程序先松键，再自动保存当前迷宫视区到
`runs/maze-debug/时间戳/`，终端与异常备注会显示绝对路径。
文件只包含游戏视区，没有额外抓取全屏。正常模式仅报错，不创建诊断目录或写入图片。
以下离线诊断工具由用户主动运行，始终输出诊断结果，不受自动程序的 `--debug` 开关影响。

- `report.json`：当前星期、人物像素坐标、历史全局位置、移动方向、格距、候选拒绝原因及七张地图评分。
- `grid_overlay.png`：恢复图上的格网线、黄色黑圈拟合和红色人物掩膜，检查墙边、圆心和掩膜覆盖是否正确。
- `raw.png`、`restored.png`、`valid_pixels.png`、`walls.png`：分辨裁剪、遮罩和墙线识别问题。
- `comparison.png`：左侧为当前观测，右侧为模板对应区域；红框是墙/通路不一致或额外宝箱的位置。
- `observed.png`：灰色格表示本帧不可观测，不是地图墙壁。

失败现场可以离线重放，不启动游戏、不发送按键：

```powershell
.\.venv\Scripts\python.exe -X utf8 tools/maze_debug_alignment.py runs/maze-debug/目录/raw.png --cropped --state runs/maze-debug/目录/report.json
```

也能从完整截图重新诊断（用 `--weekday 0..6` 指定截图拍摄时的地图，不一定是今天）：

```powershell
.\.venv\Scripts\python.exe -X utf8 tools/maze_debug_alignment.py images/maze/full.png --weekday 2
```

判读顺序：

1. 看 `alignment.candidates` 的 `reason`。`out_of_bounds` 通常是首帧不在 `(1,1)` 或人物格识别错误；
   `grid_parity` 是视图原点与大小格奇偶不一致；`player_in_wall` 是人物被定位到墙格。
2. `low_score` 时看 `compared_cells`、`mismatch_cells` 和 `unexpected_treasures`。
   `wall_score` 很高但 `score=0`，通常是额外识别出的宝箱不在模板位置。
3. 查看 `weekdays` 中各图的 `fixed_start.score`。另一张图明显更高，可能是星期对应关系或截图日期不对。
4. 当日图 `best_global` 高、运行允许范围内的候选低，检查初始人物位置、摄像机滚动和首条可见格线变化；
   只有部分小视图区相似不能证明全局位置正确。
5. 所有图都低，优先检查格网叠加图。格距或墙线错位时不应通过降低 `0.88` 阈值绕过。

七图全局搜索仅用于诊断，不会自动更换运行地图、跳过起点限制或恢复探索。
源码新增诊断后，旧 exe 需要重新打包才能自动导出现场。

已修复一种跟踪失败：只按最近的 WASD 方向枚举候选，会在摄像机仍沿另一轴缓动、
或边缘格线显隐导致视图索引改变时漏掉正确位置。
现在跟踪候选始终覆盖上一原点附近的双轴偏移，仍按边界、奇偶、人物可通行性和 `0.88` 阈值筛选，
不扩大为全图重定位，也不自动改变星期地图。

人物定位不再使用精确纯黑像素的合并凸包。`vision/ring.py` 使用灰度梯度和近黑掩膜提出圆环候选，
辅以连通圆弧候选，在遮挡时通过可见弧段拟合圆心和半径。
半径范围随视区缩放；候选需满足环外亮度反差、径向梯度方向和圆周半径一致性，
排除墙角及宝箱纹理。不同位置存在同样可信的圆环时拒绝猜测。
圆心用于导航，1.4 倍黑圈半径的掩膜用于排除冠饰、头发和头像对墙线识别的干扰。
只有少量黑点、圆弧过少或不可信区域遮挡过多时仍会停止识别。
