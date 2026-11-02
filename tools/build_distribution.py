"""构建包含运行依赖、模型和说明素材的 Windows 发布 ZIP。"""

from pathlib import Path
import json
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

    # 在打包前给出原始导入错误，避免缺依赖的构建环境生成有问题的发布包。
    subprocess.run(
        [sys.executable, "-c",
         "from zmxy4_treasure_hunt.runtime_check import check_winrt_dependencies; check_winrt_dependencies()"],
        cwd=ROOT, check=True,
    )

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
    # 实际运行冻结程序；仅检查依赖，不打开 GUI 或初始化截图设备。
    report_path = output / "runtime-check.json"
    report_path.unlink(missing_ok=True)
    result = subprocess.run(
        [str(app / f"{APP_NAME}.exe"), "--check-runtime", str(report_path)],
        cwd=output, timeout=60, check=False,
    )
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    if result.returncode or not report.get("ok") or not report.get("frozen"):
        raise RuntimeError(f"冻结程序 WinRT 依赖检查失败：{report}")
    archive = shutil.make_archive(str(output / APP_NAME), "zip", output, APP_NAME)
    print(f"发布包：{archive}")


if __name__ == "__main__":
    main()
