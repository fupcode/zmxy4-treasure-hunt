"""航海寻宝与迷宫寻宝的统一命令行入口。"""

import argparse
import sys


def get_count_from_user():
    while True:
        value = input("请输入需要胜利的次数（默认 3）：").strip()
        if not value:
            return 3
        try:
            count = int(value)
        except ValueError:
            count = 0
        if count > 0:
            return count
        print("请输入正整数。")


def choose_command():
    print("请开启最高画质，进入寻宝游戏界面，确保出现迷宫或海，并保持游戏窗口无遮挡。")
    print("如果使用微端这种以管理员权限运行的游戏启动器，请以管理员身份运行本程序，否则无法输入鼠标和键盘指令。")
    print()
    while True:
        choice = input("请选择：1. 航海寻宝  2. 迷宫寻宝\n>> ").strip().lower()
        if choice in ("1", "voyage", "航海", "航海寻宝"):
            return "voyage", [str(get_count_from_user())]
        if choice in ("2", "maze", "迷宫寻宝", "寻宝"):
            return "maze", []
        print("请输入 1（航海寻宝）或 2（迷宫寻宝）。")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="航海寻宝与迷宫寻宝；不带参数时交互选择。",
        epilog="示例：main.py voyage 3；main.py maze --debug。子命令帮助：main.py maze --help。",
    )
    parser.add_argument("command", choices=("maze", "voyage"), help="maze：迷宫寻宝；voyage：航海寻宝")
    try:
        if argv:
            command = parser.parse_args(argv[:1]).command
            child_args = argv[1:]
        else:
            command, child_args = choose_command()

        if command == "maze":
            from zmxy4_treasure_hunt.maze.cli import main as run
        else:
            from zmxy4_treasure_hunt.voyage.cli import main as run
        return run(child_args)
    except (KeyboardInterrupt, EOFError):
        print("\n已退出。")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
