import cv2
import numpy as np
import onnxruntime as ort
import time


class DetectionResult:
    def __init__(self, conf, xyxy, xyxyn, xywh, xywhn):
        self.conf = conf  # 置信度
        self.xyxy = xyxy  # (x_min, y_min, x_max, y_max) 绝对坐标
        self.xyxyn = xyxyn  # (x_min, y_min, x_max, y_max) 归一化坐标
        self.xywh = xywh  # (x_center, y_center, width, height) 绝对坐标
        self.xywhn = xywhn  # (x_center, y_center, width, height) 归一化坐标


class OnnxModel:
    def __init__(self, onnx_path=None):
        # 加载 ONNX 模型
        if onnx_path is None:
            self.session = ort.InferenceSession("model/best.onnx")
        else:
            self.session = ort.InferenceSession(onnx_path)

        # 获取模型输入的名称和形状
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

        metadata = self.session.get_modelmeta()
        names = metadata.custom_metadata_map["names"]
        self.class_names = eval(names)

        self.original_shape = None
        self.scale_factors = None

    def preprocess(self, RGB_image):
        # 获取图像原始大小
        self.original_shape = RGB_image.shape[:2]  # (height, width)

        # 调整图像大小以匹配模型输入
        image_resized = cv2.resize(RGB_image, (self.input_shape[3], self.input_shape[2]), interpolation=cv2.INTER_LINEAR)

        # 转换为 NCHW 格式并归一化
        input_image = image_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_image = np.expand_dims(input_image, axis=0)  # 添加批次维度

        # 记录缩放因子，用于后续还原检测框
        self.scale_factors = (
            self.original_shape[1] / self.input_shape[3],
            self.original_shape[0] / self.input_shape[2]
        )

        return input_image

    def forward(self, BGR_image, conf_threshold=0.4, iou_threshold=0.4, specific_class=()):
        # 预处理图像
        input_image = self.preprocess(BGR_image)

        # 获取模型的输出
        outputs = self.session.run(None, {self.input_name: input_image})
        detections = outputs[0][0]  # (count, boxes+classes)

        count = detections.shape[0]
        class_probs = detections[:, 4:]
        class_boxes = detections[:, :4]
        class_ids = np.argmax(class_probs, axis=1)
        class_scores = class_probs[np.arange(count), class_ids]

        # 过滤掉置信度低的检测结果
        mask = class_scores > conf_threshold
        class_boxes = class_boxes[mask]
        class_ids = class_ids[mask]
        class_scores = class_scores[mask]

        # 变换box为xyxy格式
        class_boxes[:, 0] -= class_boxes[:, 2] / 2
        class_boxes[:, 1] -= class_boxes[:, 3] / 2
        class_boxes[:, 2] += class_boxes[:, 0]
        class_boxes[:, 3] += class_boxes[:, 1]

        # 还原到原图尺度
        class_boxes[:, [0, 2]] *= self.scale_factors[0]
        class_boxes[:, [1, 3]] *= self.scale_factors[1]

        # 使用 NMS 处理每个类别
        results = {self.class_names[key]: [] for key in self.class_names}
        for class_id in np.unique(class_ids):
            # 获取当前类别的所有框和置信度
            indices = np.where(class_ids == class_id)[0]
            boxes = class_boxes[indices].tolist()
            scores = class_scores[indices].tolist()

            # 执行 NMS
            if class_id in specific_class:
                nms_indices = self.nms(boxes, scores, 0.1)
            else:
                nms_indices = self.nms(boxes, scores, iou_threshold)

            # 检查 nms_indices 格式并处理结果
            if len(nms_indices) > 0:
                for i in nms_indices:
                    x_min, y_min, x_max, y_max = boxes[i]
                    conf = scores[i]
                    xyxy = [x_min, y_min, x_max, y_max]

                    # 计算其他格式的框
                    x_center = (x_min + x_max) / 2
                    y_center = (y_min + y_max) / 2
                    width = x_max - x_min
                    height = y_max - y_min
                    xywh = [x_center, y_center, width, height]
                    xyxyn = [x_min / self.original_shape[1], y_min / self.original_shape[0],
                             x_max / self.original_shape[1], y_max / self.original_shape[0]]
                    xywhn = [x_center / self.original_shape[1], y_center / self.original_shape[0],
                             width / self.original_shape[1], height / self.original_shape[0]]

                    # 创建 DetectionResult 对象并添加到结果字典
                    class_name = self.class_names[class_id]
                    detection_result = DetectionResult(conf, xyxy, xyxyn, xywh, xywhn)
                    results[class_name].append(detection_result)

        return results

    def nms(self, boxes, scores, iou_threshold):
        # 如果没有框，直接返回空列表
        if len(boxes) == 0:
            return []

        # 将 boxes 转换为 numpy 数组，确保数据类型一致
        boxes = np.array(boxes)
        scores = np.array(scores)

        # 获取每个框的坐标
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # 计算每个框的面积
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        # 按得分排序（从大到小）
        order = scores.argsort()[::-1]

        keep = []  # 保存最终保留的框的索引

        while len(order) > 0:
            # 选择当前得分最高的框，并将其加入保留列表
            i = order[0]
            keep.append(i)

            # 计算当前框和剩余框的交集区域
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            # 计算交集宽度和高度
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # 计算交集面积和 IoU
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            # 保留 IoU 小于阈值的框
            remaining = np.where(iou <= iou_threshold)[0]
            order = order[remaining + 1]

        return keep

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
