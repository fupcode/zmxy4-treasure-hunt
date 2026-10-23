# YOLO 训练

Windows 下项目已将 `torch`、`torchvision` 绑定到 PyTorch 官方 CUDA 13.0 源，
用于 RTX 50 系显卡。运行 `uv sync` 安装，首次需要下载较大的 CUDA 版 wheel。
可用 `uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`
确认安装结果，最后一项应为 `True`。

在项目根目录运行（使用项目已有的 uv 环境）：

```powershell
uv run python yolo/train.py --dry-run
uv run python yolo/train.py
```

默认使用 `zaomeng.v2i.yolo26/data.yaml` 和 `yolo26n.pt`，训练 100 轮、
长边尺寸 800、矩形训练（`rect=True`）、batch 16，自动选择 CUDA 或 CPU。首次训练需要联网下载预训练权重；
也可通过 `--model` 指定已有的本地 `.pt` 文件。

指定 GPU 和训练参数：

```powershell
uv run python yolo/train.py --device 0 --model yolo26s.pt --epochs 200 --batch 8 --imgsz 960
```

显存不足时调小 `--batch` 或 `--imgsz`。Windows 默认 `--workers 0`，
可根据机器情况增大。`--patience 30` 表示连续 30 轮验证指标无提升时提前停止。
完整参数可通过 `--help` 查看。

训练结果写入 `yolo/runs/zaomeng/`，重名时自动递增目录名。
`weights/best.pt` 是最佳权重，`weights/last.pt` 用于中断后续训：

```powershell
uv run python yolo/train.py --resume yolo/runs/zaomeng/weights/last.pt
```

续训沿用断点中的原始训练计划；`--epochs`、`--model` 等新训练参数不会覆盖它。

## 导出免 NMS ONNX

```powershell
uv run python yolo/export.py
# 若训练结果目录自动递增，指定实际权重路径
uv run python yolo/export.py --weights yolo/runs/zaomeng2/weights/best.pt --max-det 300
```

导出脚本固定 `nms=False`，选择 YOLO26 的免 NMS 检测头，输出每行
`[x1, y1, x2, y2, confidence, class_id]`，只需置信度过滤，不再执行 NMS。
`max_det` 默认 300，密集子弹场景可在导出时增加；导出后修改此上限需要重新导出。
文件保存在权重旁的 `best.onnx`。部署时复制到 `model/best.onnx`，替换现有模型。

数据集应重新导出为 Stretch 到 **宽 800、高 512**。训练使用长边 800 和
矩形训练，两边均对齐 stride 32，无需额外补边。训练仍使用 Ultralytics 默认增强，
数据集导出时未增强不代表训练时禁用增强。
训练会同时学习两个检测头，`nms=False` 在导出时选择免 NMS 分支。

导出使用 `imgsz=(512, 800)`（顺序为高、宽）。`OnnxModel` 读取模型输入尺寸，
将 RGB 图像直接拉伸到宽 800、高 512，不再补边；输出坐标直接按模型宽高归一化。
请使用新尺寸的数据集重新训练并导出 ONNX，确保训练与推理的预处理一致。
`OnnxModel` 输入沿用 dxcam 的 RGB 顺序；使用 `cv2.imread` 时先执行
`cv2.cvtColor(image, cv2.COLOR_BGR2RGB)`。

推理代码仅支持 `nms=False` 导出的模型，已删除 NMS 方法和 IoU 相关参数。
使用 `model(rgb_image, conf_threshold=0.3)` 控制保留的检测结果，重叠框不会互相抑制。

脚本默认数据集路径相对脚本定位，可从其他工作目录启动。自定义 `--data`、
`--model`、`--resume` 和 `--project` 的相对路径按当前工作目录解析。
自定义数据集 YAML 中的分集路径相对 YAML 所在目录解析；如有 `path` 字段，使用绝对路径。

数据集配置已按实际目录设置 `train/images` 和 `valid/images`，未设置不存在的测试集。
`--dry-run` 只检查配置与分集路径，不检查每条标注，也不下载模型或启动训练。

训练接口参考：[Ultralytics 官方文档](https://docs.ultralytics.com/modes/train/)。
