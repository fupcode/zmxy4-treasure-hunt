"""读取全部内置地图和遮罩；可作为 PyInstaller 资源验收入口，不操作游戏。"""

from zmxy4_treasure_hunt.maze.map_templates import load_weekday_map
from zmxy4_treasure_hunt.maze.mask import MaskRestorer


def main():
    for weekday in range(7):
        template = load_weekday_map(weekday)
        if template.cells.shape != (29, 29):
            raise ValueError(f"地图尺寸不正确：{weekday}")
        print(f"map {weekday}: 29x29, {(template.cells == 4).sum()} treasures")
    restorer = MaskRestorer(620, 344)
    if restorer.transmission.shape != (344, 620):
        raise ValueError("遮罩尺寸不正确")
    print("OK: 7 bundled maps and mask")


if __name__ == "__main__":
    main()
