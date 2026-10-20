"""One serialized YOLO runtime for image, video and camera inference."""

import torch
import cv2
from copy import deepcopy
from pathlib import Path
import threading

import ultralytics
from ultralytics import YOLO

from dataset_store import file_sha
from inference_result import FrameResult


class YoloDetector:
    def __init__(self, weights, device='cpu', *, imgsz=640, expected_sha256=None, model_metadata=None, require_fire_smoke=True):
        if type(imgsz) is not int or not 64 <= imgsz <= 1280 or imgsz % 32:
            raise ValueError('推理图像尺寸须为64～1280内32的倍数。')
        self.device, self.imgsz = device, imgsz
        weights = Path(weights).resolve()
        digest = file_sha(weights)
        if expected_sha256 and digest != expected_sha256:
            raise OSError('模型权重校验失败，未加载替代模型。')
        self.model = YOLO(str(weights))
        names = self.model.names
        if self.model.task != 'detect' or (require_fire_smoke and (len(names) != 2 or set(names.values()) != {'fire', 'smoke'})):
            raise ValueError('请选择仅包含fire和smoke两类的目标检测模型。')
        if model_metadata and model_metadata.get('origin') == 'trained' and names != {0: 'fire', 1: 'smoke'}:
            raise ValueError('自训模型类别映射与训练导出不一致。')
        self.model.to(device)
        if file_sha(weights) != digest:
            raise OSError('模型在加载时发生变化，请重新选择。')
        self._metadata = {**deepcopy(model_metadata or {}), 'path': str(weights), 'sha256': digest,
                          'device': device, 'imgsz': imgsz, 'classes': deepcopy(names),
                          'precision': 'FP32', 'torch': torch.__version__, 'ultralytics': ultralytics.__version__}
        self._lock = threading.Lock()

    @property
    def metadata(self):
        return deepcopy(self._metadata)

    def _infer(self, source, conf, iou):
        result = self.model.predict(source, conf=conf, iou=iou, device=self.device, imgsz=self.imgsz,
                                    half=False, max_det=300, verbose=False)[0]
        detections, counts = [], {}
        for x1, y1, x2, y2, confidence, class_id in result.boxes.data.cpu().tolist():
            label = result.names[int(class_id)]
            counts[label] = counts.get(label, 0) + 1
            detections.append({'class': label, 'confidence': confidence, 'xyxy': [x1, y1, x2, y2]})
        return result, detections, counts

    def predict(self, bgr, conf, iou):
        with self._lock:
            result, detections, counts = self._infer(bgr, conf, iou)
            annotated = result.plot()
            factor = min(1.0, 960 / annotated.shape[1], 720 / annotated.shape[0])
            if factor < 1:
                annotated = cv2.resize(annotated, (round(annotated.shape[1] * factor), round(annotated.shape[0] * factor)))
            return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), detections, {'fire': 0, 'smoke': 0, **counts}, float(result.speed.get('inference') or 0)

    def predict_image(self, image, conf, iou):
        with self._lock:
            result, detections, counts = self._infer(image.convert('RGB'), conf, iou)
            return cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB), detections, counts, sum(result.speed.values())

    def predict_frame(self, bgr, conf, iou):
        from PIL import Image
        rgb, boxes, counts, elapsed = self.predict_image(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), conf, iou)
        return FrameResult(rgb, boxes, {'fire': 0, 'smoke': 0, **counts}, bgr.shape[1], bgr.shape[0],
                           {'detection': elapsed}, self.metadata)
