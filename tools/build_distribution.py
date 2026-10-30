"""构建包含运行依赖、模型和说明素材的 Windows 发布 ZIP。"""

from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "造梦西游4寻宝助手"


def main():
    if sys.platform != "win32":
        raise SystemExit("请在 Windows 上构建 Windows 发布包")
    model = ROOT / "weights" / "best.onnx"
    assets = ROOT / "packaging" / "assets"
    if not model.is_file():
        raise FileNotFoundError(f"缺少航海模型：{model}")
    if not list(assets.glob("*.png")):
        raise FileNotFoundError(f"缺少发布说明图片：{assets}")

    # 使用独立输出目录，保留手动维护的 dist 目录。
    output = ROOT / "build" / "release"
    subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
         "--distpath", str(output), "main.spec"],
        cwd=ROOT, check=True,
    )
    app = output / APP_NAME
    (app / "weights").mkdir(exist_ok=True)
    shutil.copy2(model, app / "weights" / model.name)
    for image in assets.glob("*.png"):
        shutil.copy2(image, app / image.name)
    for name in ("LICENSE", "README.md"):
        shutil.copy2(ROOT / name, app / name)

    required = [app / f"{APP_NAME}.exe", app / "_internal",
                app / "weights" / "best.onnx"]
    if not all(path.exists() for path in required):
        raise RuntimeError("发布包缺少程序、运行依赖或模型")
    archive = shutil.make_archive(str(output / APP_NAME), "zip", output, APP_NAME)
    print(f"发布包：{archive}")


if __name__ == "__main__":
    main()
