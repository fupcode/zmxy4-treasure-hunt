"""移动方向及动作编号对应关系。"""

from enum import Enum


class Direction(Enum):
    Right = "D"
    Down = "S"
    Left = "A"
    Up = "W"
    DownRight = "SD"
    DownLeft = "SA"
    UpRight = "WD"
    UpLeft = "WA"


direction_list = [Direction.Right, Direction.Down, Direction.Left, Direction.Up,
                  Direction.DownRight, Direction.DownLeft, Direction.UpRight, Direction.UpLeft]


move_list = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]
