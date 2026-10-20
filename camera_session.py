"""Live-camera observations reuse the video detection/alarm/media pipeline."""

from __future__ import annotations

import math
import threading
import time
import uuid

from alarm import AlarmConfig
from camera_source import CameraConfig, CameraReadError, CameraSource
from video_detection import VideoInfo, VideoSession


class CameraSession(VideoSession):
    def __init__(self, camera_config: CameraConfig, detector, conf=0.2, iou=0.5,
                 alarm_config=None, log_path=None, source_name=None, recording_config=None,
                 media_root=None, source_factory=None, monotonic=None):
        if not 0 < conf <= 1 or not 0 < iou <= 1:
            raise ValueError('置信度和IoU阈值必须位于0到1之间。')
        self._lock = threading.RLock()
        self._camera_config = camera_config
        self._source_factory = source_factory or CameraSource
        self._clock = monotonic or time.monotonic
        self.input_type = 'camera'
        self.source = f'camera:{camera_config.index}'
        self.detector, self.conf, self.iou = detector, conf, iou
        self._source_name = source_name or f'摄像头 {camera_config.index}'
        self._alarm_config = alarm_config or AlarmConfig()
        self._recording_config = recording_config
        self._log_path, self._media_root = log_path, media_root
        self.camera_session_id = uuid.uuid4().hex
        self.segment_number = 1
        self.total_processed_frames = 0
        self._past_recorders = []
        self._capture_watch_stop = threading.Event()
        self.capture = self._source_factory(camera_config)
        try:
            self._start_segment()
            self._start_capture_watch()
        except Exception:
            self.capture.release()
            raise

    def _start_segment(self):
        initial = self.capture.read_latest(0)
        if initial is None:
            raise ValueError('摄像头尚未提供首帧，请重新开始。')
        raw_fps = float(self.capture.info.get('fps', 0))
        estimated = not math.isfinite(raw_fps) or raw_fps <= 0
        self.info = VideoInfo(fps=self._camera_config.fps if estimated else raw_fps, fps_estimated=estimated,
                              frame_count=0, width=initial.bgr.shape[1], height=initial.bgr.shape[0])
        self._initialize_pipeline(self._source_name, self._alarm_config, self._log_path,
                                  self._recording_config, self._media_root,
                                  {**self.capture.metadata(), 'camera_session_id': self.camera_session_id,
                                   'observation_segment': self.segment_number})
        self._epoch = None
        self._last_capture_sequence = 0
        self._last_capture_stamp = None
        self.dropped_frames = 0

    def _start_capture_watch(self):
        self._capture_watch_stop = threading.Event()
        threading.Thread(target=self._watch_capture, args=(self.capture, self._capture_watch_stop),
                         name='camera-observation-watch', daemon=True).start()

    def _watch_capture(self, capture, stopped):
        while not stopped.wait(0.1):
            if getattr(capture, 'error', None) or not capture.isOpened():
                with self._lock:
                    if self.status == 'running' and self.capture is capture:
                        self.error = getattr(capture, 'error', None) or '摄像头已关闭，当前观测已中断。'
                        self._finish('disconnected')
                break

    def _finish(self, status):
        self._capture_watch_stop.set()
        super()._finish(status)

    @property
    def frame_age_seconds(self):
        return None if self._last_capture_stamp is None else max(0, self._clock() - self._last_capture_stamp)

    def pause(self):
        with self._lock:
            if self.status == 'running':
                self._capture_watch_stop.set()
                self.capture.release()
                self._end_interval()
                self.alarms.finish('paused')
                if self.recorder:
                    self.recorder.finish('paused')
                self.status = 'paused'

    def resume(self):
        with self._lock:
            if self.status != 'paused':
                return
            self._past_recorders = [recorder for recorder in self._past_recorders if recorder.pending_jobs]
            pending = sum(recorder.pending_jobs for recorder in self._past_recorders)
            if self.recorder:
                pending += self.recorder.pending_jobs
            if self._recording_config and pending >= self._recording_config.max_pending_clips:
                raise RuntimeError('前一观测段的片段仍在保存，请稍后再继续。')
            previous_capture, previous_info, previous_recorder = self.capture, self.info, self.recorder
            self.capture = self._source_factory(self._camera_config)
            self.segment_number += 1
            try:
                self._start_segment()
            except Exception:
                self.capture.release()
                self.capture, self.info = previous_capture, previous_info
                self.segment_number -= 1
                raise
            if previous_recorder and previous_recorder.pending_jobs:
                self._past_recorders.append(previous_recorder)
            self._start_capture_watch()

    def wait_for_media(self, timeout=90):
        for recorder in [*self._past_recorders, self.recorder]:
            if recorder:
                recorder.wait(timeout)

    def step(self):
        with self._lock:
            if self.status != 'running':
                return None
            try:
                packet = self.capture.read_latest(self._last_capture_sequence)
                if packet is None:
                    return None
                if packet.sequence <= self._last_capture_sequence or not math.isfinite(packet.captured_at):
                    raise CameraReadError('摄像头采集序号或时间无效，已结束本次观测。')
                if self._epoch is None:
                    self._epoch = packet.captured_at
                seconds = packet.captured_at - self._epoch
                if self.latest is not None:
                    gap = seconds - self.latest.media_seconds
                    if gap <= 0 or gap > self._alarm_config.max_frame_gap_seconds + 1e-6:
                        raise CameraReadError('实时检测采样间隔过大或时间异常，已结束观测，请重新开始；未判定火情恢复。')
                self.dropped_frames += packet.sequence - self._last_capture_sequence - 1
                self._last_capture_sequence = packet.sequence
                self._last_capture_stamp = packet.captured_at
                result = self._detect_frame(packet.bgr, seconds)
                self.total_processed_frames += 1
                return result
            except CameraReadError as exc:
                self.error = str(exc)
                self._finish('disconnected')
                return None
            except Exception:
                self.error = '摄像头画面检测失败，设备已关闭，请检查模型或计算设备后重新开始。'
                self._finish('error')
                raise
