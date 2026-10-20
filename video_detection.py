"""Frame-by-frame fire/smoke detection, independent of the Streamlit interface."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from alarm import AlarmConfig
from alarm_log import AlarmMonitor


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    fps_estimated: bool
    frame_count: int
    width: int
    height: int


@dataclass
class DetectedFrame:
    number: int
    media_seconds: float
    rgb: np.ndarray
    detections: list[dict]
    counts: dict[str, int]
    inference_ms: float
    time_estimated: bool


class FireSmokeDetector:
    """Serialize calls because YOLO keeps a mutable predictor on its model."""

    def __init__(self, weights: str | Path, device: str = 'cpu'):
        self.device = device
        self.model = YOLO(str(weights))
        if not {'fire', 'smoke'}.issubset(set(self.model.names.values())):
            raise ValueError('请选择同时包含 fire 和 smoke 类别的检测模型。')
        self.model.to(device)
        self._lock = threading.Lock()

    def predict(self, bgr: np.ndarray, conf: float, iou: float):
        with self._lock:
            result = self.model.predict(bgr, conf=conf, iou=iou, device=self.device, imgsz=640, verbose=False)[0]
            detections = []
            counts = {'fire': 0, 'smoke': 0}
            for x1, y1, x2, y2, confidence, class_id in result.boxes.data.cpu().tolist():
                label = result.names[int(class_id)]
                counts[label] = counts.get(label, 0) + 1
                detections.append({'class': label, 'confidence': confidence, 'xyxy': [x1, y1, x2, y2]})
            annotated = result.plot()
            # Bound each browser update; retain no list of historical images.
            factor = min(1.0, 960 / annotated.shape[1], 720 / annotated.shape[0])
            if factor < 1:
                annotated = cv2.resize(annotated, (round(annotated.shape[1] * factor), round(annotated.shape[0] * factor)))
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            return rgb, detections, counts, float(result.speed.get('inference') or 0)


class VideoSession:
    """Read every frame in order; pause retains the next frame and media clock."""

    def __init__(self, source: str | Path, detector: FireSmokeDetector, conf: float = 0.2, iou: float = 0.5,
                 alarm_config: AlarmConfig | None = None, log_path: str | Path | None = None,
                 source_name: str | None = None):
        if not 0 < conf <= 1 or not 0 < iou <= 1:
            raise ValueError('置信度和IoU阈值必须位于0到1之间。')
        self._lock = threading.RLock()
        self.source = Path(source)
        self.detector = detector
        self.conf = conf
        self.iou = iou
        self.capture = cv2.VideoCapture(str(self.source))
        if not self.capture.isOpened():
            self.capture.release()
            raise ValueError('无法打开这个视频，请使用可正常播放的MP4、MOV、AVI或MKV文件。')
        raw_fps = self.capture.get(cv2.CAP_PROP_FPS)
        estimated = not math.isfinite(raw_fps) or raw_fps <= 0
        self.info = VideoInfo(
            fps=30.0 if estimated else raw_fps, fps_estimated=estimated,
            frame_count=max(0, int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))),
            width=int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        self.status = 'running'
        self.error = None
        self.latest: DetectedFrame | None = None
        self.processed_frames = 0
        self.fire_frames = 0
        self.smoke_frames = 0
        self._elapsed = 0.0
        self._started = time.perf_counter()
        self.alarms = AlarmMonitor(alarm_config or AlarmConfig(), source_name or self.source.name, log_path,
                                   metadata={'confidence': conf, 'nms_iou': iou,
                                             'device': getattr(detector, 'device', 'unknown'),
                                             'weights': str(getattr(detector.model, 'ckpt_path', 'unknown')) if hasattr(detector, 'model') else 'test detector'})

    @property
    def is_open(self) -> bool:
        return self.capture.isOpened()

    @property
    def processing_fps(self) -> float:
        elapsed = self._elapsed
        if self._started is not None:
            elapsed += time.perf_counter() - self._started
        return self.processed_frames / elapsed if elapsed > 0 else 0.0

    def _end_interval(self):
        if self._started is not None:
            self._elapsed += time.perf_counter() - self._started
            self._started = None

    def _finish(self, status: str):
        self._end_interval()
        self.capture.release()
        self.status = status
        self.alarms.finish(status)

    def pause(self):
        with self._lock:
            if self.status == 'running':
                self._end_interval()
                self.status = 'paused'

    def resume(self):
        with self._lock:
            if self.status == 'paused':
                self._started = time.perf_counter()
                self.status = 'running'

    def stop(self):
        with self._lock:
            if self.status in {'running', 'paused'}:
                self._finish('stopped')

    def step(self) -> DetectedFrame | None:
        with self._lock:
            if self.status != 'running':
                return None
            try:
                ok, bgr = self.capture.read()
                if not ok:
                    if self.processed_frames == 0:
                        self.error = '视频中没有可解码的画面，请更换文件。'
                    elif self.info.frame_count > self.processed_frames + 1:
                        self.error = f'视频在第{self.processed_frames}帧后提前结束，文件可能不完整。'
                    self._finish('error' if self.error else 'completed')
                    return None
                pts_ms = self.capture.get(cv2.CAP_PROP_POS_MSEC)
                media_seconds = pts_ms / 1000
                time_estimated = not math.isfinite(media_seconds) or media_seconds < 0
                if self.latest is not None and media_seconds <= self.latest.media_seconds:
                    time_estimated = True
                if time_estimated:
                    media_seconds = self.processed_frames / self.info.fps
                    if self.latest is not None:
                        media_seconds = max(media_seconds, self.latest.media_seconds + 1 / self.info.fps)
                rgb, detections, counts, inference_ms = self.detector.predict(bgr, self.conf, self.iou)
                self.processed_frames += 1
                self.fire_frames += int(counts.get('fire', 0) > 0)
                self.smoke_frames += int(counts.get('smoke', 0) > 0)
                self.latest = DetectedFrame(
                    number=self.processed_frames, media_seconds=media_seconds,
                    rgb=rgb, detections=detections, counts=counts, inference_ms=inference_ms,
                    time_estimated=time_estimated,
                )
                self.alarms.update(self.processed_frames, media_seconds, detections)
                return self.latest
            except Exception:
                self.error = '读取或检测失败，请停止后尝试其他视频或切换计算设备。'
                self._finish('error')
                raise

    def __del__(self):
        capture = getattr(self, 'capture', None)
        if capture is not None:
            capture.release()
