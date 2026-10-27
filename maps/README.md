# 每周迷宫地图

从 `images/maze/ft0.png` 至 `ft6.png` 解析，按用户约定编号：

| 文件 | 星期 | 来源 |
| --- | --- | --- |
| `0.json` | 星期日 | `ft0.png` |
| `1.json` | 星期一 | `ft1.png` |
| `2.json` | 星期二 | `ft2.png` |
| `3.json` | 星期三 | `ft3.png` |
| `4.json` | 星期四 | `ft4.png` |
| `5.json` | 星期五 | `ft5.png` |
| `6.json` | 星期六 | `ft6.png` |

每张图包含 14×14 个房间，转换为现有导航使用的 29×29 格网。
坐标均为 `[行, 列]`，从 0 开始；奇数行、奇数列对应房间中心。

| 字符 | 数值 | 含义 |
| --- | --- | --- |
| `#` | 2 | 墙壁或格网交点 |
| `.` | 1 | 普通房间 |
| `+` | 3 | 两个相邻房间之间的可通行连接 |
| `T` | 4 | 宝箱候选位置 |

`grid` 保存字符串地图，`legend` 保存字符到现有数值编码的映射；
`treasures` 保存 8 个宝箱的格网坐标，与 `grid` 中的 `T` 一致。
人物起点由程序初始化为 `(1, 1)`，地图文件不保存 `start` 字段。宝箱真假无法从这些图片判断。
交点按现有导航约定统一记为墙格；这是房间连通图，不是像素级碰撞图。

## 读取

自动程序已接入这些模板：启动时按北京时间加载当日地图，不再探索。
局部画面只用于定位和移动纠偏，触发假宝箱后只更新本局内存状态。

在项目根目录运行：

```python
import json
from datetime import date
from pathlib import Path

import numpy as np

weekday = date.today().isoweekday() % 7  # 星期日为 0，星期一为 1
data = json.loads(Path(f"maps/{weekday}.json").read_text(encoding="utf-8"))
maze = np.array(
    [[data["legend"][cell] for cell in row] for row in data["grid"]],
    dtype=np.uint8,
)
treasures = [tuple(cell) for cell in data["treasures"]]
```

本局假宝箱消失时应修改内存中的 `maze`，保留 JSON 模板供下次运行加载。

## 重新解析与复核

```powershell
.\.venv\Scripts\python.exe -X utf8 tools/extract_maze_maps.py
```

提取脚本针对这批 1210×1098 图片：按格网连接位置采样区分沙路与石墙，
再用 `ft0.png` 的宝箱纹理识别其余宝箱。脚本检查分类余量、宝箱数量和房间连通性，
生成 `maps/0.json` 至 `6.json`，并输出 `images/maze/parsed/ft0.png` 至 `ft6.png` 供复核。
复核图中蓝点表示通路、紫点表示墙壁、红框表示宝箱，标签为 `[行, 列]`。
每张图的 196 个房间均连通；图片只用于离线解析，不操作游戏。
