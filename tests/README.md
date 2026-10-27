# 寻宝测试

航海自动测试放在 `tests/voyage/`，覆盖规则自动战斗引擎：射击对齐、敌人冷却、子弹识别与追踪、
危险图与寻路、渲染开关、帧率统计、环境超时和 ONNX 感知接口。
不启动游戏窗口、不发送游戏操作。

迷宫自动测试放在 `tests/maze/`，覆盖图像识别、格网重建、完整地图定位与宝箱最短路径、截图和输入清理。

在项目根目录运行全部自动测试：

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p "test_*.py" -v
```

仅运行航海寻宝自动测试：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests/voyage -p "test_*.py" -v
```

`test_env_execution.py` 验证环境执行流程、感知与导航边界及动作状态。

手动图片诊断和性能比较脚本保留在 `tests/` 根目录，不纳入上述自动测试：

```powershell
.\.venv\Scripts\python.exe -X utf8 tests/collision_test.py images/dead1.png --map-index 1 --level 1 --force-detect --no-show
.\.venv\Scripts\python.exe -X utf8 tests/onnx_test.py images/dead1.png --no-show
.\.venv\Scripts\python.exe -X utf8 tests/benchmark_maze.py
```

碰撞诊断的地图编号和关卡需要与图片对应。
