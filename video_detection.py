"""Frame-by-frame fire/smoke detection, independent of the Streamlit interface."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from model_runtime import YoloDetector

from alarm import AlarmConfig
from alarm_log import AlarmMonitor, alarm_media_root
from recording import EventRecorder, RecordingConfig
from monitoring import MonitoringPoint


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
    result: object = None


class FireSmokeDetector(YoloDetector):
    """Compatibility entry point; the shared runtime validates fire/smoke."""


class VideoSession:
    """Read every frame in order; pause retains the next frame and media clock."""

    def __init__(self, source: str | Path, detector: FireSmokeDetector, conf: float = 0.2, iou: float = 0.5,
                 alarm_config: AlarmConfig | None = None, log_path: str | Path | None = None,
                 source_name: str | None = None, recording_config: RecordingConfig | None = None,
                 media_root: str | Path | None = None, monitoring_point: MonitoringPoint | None = None):
        if not 0 < conf <= 1 or not 0 < iou <= 1:
            raise ValueError('置信度和IoU阈值必须位于0到1之间。')
        self._lock = threading.RLock()
        self.source = Path(source)
        self.detector = detector
        self.conf = conf
        self.iou = iou
        self.monitoring_point = monitoring_point
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
        self.input_type = 'video'
        self._initialize_pipeline(source_name or self.source.name, alarm_config, log_path, recording_config, media_root,
                                  {'input_type': 'video', 'clock': 'source_pts'})

    def _initialize_pipeline(self, source_name, alarm_config, log_path, recording_config, media_root, metadata):
        alarms = AlarmMonitor(alarm_config or AlarmConfig(), source_name, log_path,
                              metadata={'confidence': self.conf, 'nms_iou': self.iou,
                                        'device': getattr(self.detector, 'device', 'unknown'),
                                        'weights': str(getattr(self.detector.model, 'ckpt_path', 'unknown')) if hasattr(self.detector, 'model') else 'test detector',
                                        **({'model': self.detector.metadata} if hasattr(self.detector, 'metadata') else {}),
                                        **metadata,
                                        **({'monitoring_point': self.monitoring_point.metadata()} if self.monitoring_point else {})})
        recorder = EventRecorder(recording_config, media_root or alarm_media_root(log_path), alarms.store, self.info.fps,
                                 input_type=self.input_type) if (
            recording_config is not None and recording_config.enabled and alarms.store is not None) else None
        self.alarms, self.recorder = alarms, recorder
        self.status = 'running'
        self.error = None
        self.latest: DetectedFrame | None = None
        self.processed_frames = 0
        self.fire_frames = 0
        self.smoke_frames = 0
        self._elapsed = 0.0
        self._started = time.perf_counter()

    def _detect_frame(self, bgr, media_seconds, time_estimated=False):
        result = self.detector.predict_frame(bgr, self.conf, self.iou) if hasattr(self.detector, 'predict_frame') else None
        if result is None:
            rgb, detections, counts, inference_ms = self.detector.predict(bgr, self.conf, self.iou)
        else:
            rgb, detections, counts, inference_ms = result.rgb, result.detections, result.counts, sum(result.timing_ms.values())
        factor = min(1., 960 / rgb.shape[1], 720 / rgb.shape[0])
        if factor < 1:
            rgb = cv2.resize(rgb, (round(rgb.shape[1] * factor), round(rgb.shape[0] * factor)))
        self.processed_frames += 1
        self.fire_frames += int(counts.get('fire', 0) > 0)
        self.smoke_frames += int(counts.get('smoke', 0) > 0)
        self.latest = DetectedFrame(number=self.processed_frames, media_seconds=media_seconds,
                                    rgb=rgb, detections=detections, counts=counts, inference_ms=inference_ms,
                                    time_estimated=time_estimated, result=result)
        self.alarms.update(self.processed_frames, media_seconds, detections)
        if self.recorder:
            event = self.alarms.current_event
            self.recorder.add_frame(self.latest, event['id'] if event else None)
        return self.latest

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
        self.alarms.finish(status)
        if self.recorder:
            self.recorder.finish(status)
        self.status = status

    def wait_for_media(self, timeout=90):
        if self.recorder:
            self.recorder.wait(timeout)

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
                return self._detect_frame(bgr, media_seconds, time_estimated)
            except Exception:
                self.error = '读取或检测失败，请停止后尝试其他视频或切换计算设备。'
                self._finish('error')
                raise

    def __del__(self):
        capture = getattr(self, 'capture', None)
        if capture is not None:
            capture.release()
