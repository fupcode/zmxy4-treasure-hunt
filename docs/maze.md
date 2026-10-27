# 迷宫寻宝

在项目根目录运行（默认等待 2 秒后开始发送游戏操作）：

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m zmxy4_treasure_hunt.maze
```

也可直接运行 `src/zmxy4_treasure_hunt/maze/ft.py`，或使用原来的 `-m zmxy4_treasure_hunt.maze.ft`。
监控窗口按 Esc、终端按 Ctrl+C 退出，退出时松开 WASD 并释放截图设备。

截图使用 DXcam 的 BGR 输出，DXGI 初始化失败时尝试 WinRT。目前使用主显示器；
游戏完整画面应位于主显示器内，区域坐标为该显示器内的物理像素坐标。
自动定位依赖画面中的颜色标记；定位失败时可以手动指定裁剪范围：

```powershell
.\.venv\Scripts\python.exe -m zmxy4_treasure_hunt.maze --region 100 100 1000 700 --interval 0.05 --delay 3
```

`--region` 依次为左、上、右、下边界，范围应对应自动定位后得到的迷宫游戏区域。
`--interval` 是包含处理时间的最小循环间隔，默认 0.05 秒；截图单次等待上限为 5 秒。

## 模块职责

| 文件 | 职责 |
| --- | --- |
| `ft.py`、`__main__.py` | 参数解析与启动入口 |
| `runner.py` | 截图 → 识别 → 规划 → 执行、过关与监控窗口 |
| `capture.py` | BGR 截图、超时、后端回退和释放 |
| `controls.py` | 鼠标点击、方向键变化和统一松键 |
| `vision.py` | 人物、宝箱、墙壁线条及窗口区域识别 |
| `grid.py` | 格距标定、补线、局部格网和人物偏移 |
| `navigation.py` | 29×29 地图拼接、BFS 探索和移动建议 |

识别和导航不创建窗口或发送按键。`FindTreasure.step()` 执行一帧；
`NavigationResult` 包含用于显示的地图、路径、移动建议和接近宝箱标记。
调用者应在完成使用后调用 `FindTreasure.close()`；`main()` 会自动清理。
直接从 `ft` 导入 `FindTreasure`、`Graph` 和 `Maze` 的入口保留，
其中 `Maze` 已改为纯规划接口，不再隐式发送按键或以 `None` 表示接近宝箱。

保留原有颜色阈值和探索偏好：优先最近未知区域，足够近的宝箱优先。
人物黑色特征通过凸包定位；原脚本未完成的坐标、半径赋值已补齐，
取消逐帧阻塞的调试弹窗。格距校准、人物掩膜和颜色阈值仍需用实际游戏画面校验。
识别、拼图或截图失败时停止移动并抛出明确错误，不沿用旧画面继续控制。
顶部暗场表示触发假宝箱：立即松键，等待 5 秒（覆盖约 3 秒黑屏）后继续探索。
假宝箱触发后消失，对应格子恢复为普通道路；保留现有地图、人物位置、格距和移动历史，
不点击旧版过关按钮，也不回到游戏起始点重新拼图。

## 离线回归

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m unittest discover -s tests/maze -v
```

测试使用合成图像和模拟截图、输入设备，覆盖人物/墙壁/宝箱/窗口定位、
横纵独立格距、首帧定位、视图拼接、探索策略、按键变化和错误清理。
不连接游戏，不发送真实鼠标或键盘操作。
