import cv2
import time
import os
from pathlib import Path

from zaomeng_hanghai.core.onnx_model import OnnxModel

script_path = Path(__file__).resolve()
parent_dir = script_path.parent.parent

os.chdir(parent_dir)

keys = ['boss', 'bullet', 'dan', 'enemy', 'fan', 'jia', 'pao', 'player']
model = OnnxModel()
image = cv2.imread("images/item_92.png")
assert image is not None, "Image not found or unable to read."

start = time.time()
results = model(image)
print(f"Detection time: {time.time() - start:.4f}s")

for class_name, detection_results in results.items():
    print(f"Class: {class_name}")
    for i, result in enumerate(detection_results):
        print(f"  Box {i}:")

        height, width = image.shape[:2]
        x1, y1, x2, y2 = result.xyxyn
        cv2.rectangle(image, (int(x1 * width), int(y1 * height)), (int(x2 * width), int(y2 * height)), (0, 255, 0), 2)

cv2.imshow("Detection", image)
cv2.waitKey(0)
