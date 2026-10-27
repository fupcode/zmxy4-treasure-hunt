"""迷宫程序独立入口，供 PyInstaller 使用。"""

from zmxy4_treasure_hunt.maze.ft import main


if __name__ == "__main__":
    raise SystemExit(main())
