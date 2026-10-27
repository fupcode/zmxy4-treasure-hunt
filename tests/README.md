# 引擎测试

当前测试覆盖规则自动战斗引擎：地图碰撞、射击对齐、敌人冷却、子弹识别与追踪、
危险图与寻路、渲染开关、帧率统计、环境超时和 ONNX 感知接口。
不启动游戏窗口、不发送游戏操作。

在项目根目录运行全部自动测试：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p "test_*.py" -v
```

手动图片诊断脚本继续保留，不纳入上述自动测试：

```powershell
.\.venv\Scripts\python.exe -X utf8 tests/collision_test.py images/dead1.png --map-index 1 --level 1 --force-detect --no-show
.\.venv\Scripts\python.exe -X utf8 tests/onnx_test.py images/dead1.png --no-show
```

碰撞诊断的地图编号和关卡需要与图片对应。
