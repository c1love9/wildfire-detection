"""Bounded latest-frame camera IPC, with startup/stall/idle timeouts."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from camera_capture import HEADER, FRAME


class CameraReadError(RuntimeError):
    pass


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0
    backend: str = 'AUTO'
    width: int = 640
    height: int = 480
    fps: int = 30
    open_timeout_seconds: float = 8.0
    read_timeout_seconds: float = 3.0
    idle_timeout_seconds: float = 15.0

    def __post_init__(self):
        for name, minimum, maximum in (('index', 0, 9), ('width', 160, 1920), ('height', 120, 1080), ('fps', 1, 60)):
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f'摄像头参数{name}须为{minimum}～{maximum}的整数。')
        if self.backend not in {'AUTO', 'DSHOW', 'MSMF'}:
            raise ValueError('摄像头后端须为AUTO、DSHOW或MSMF。')
        for name in ('open_timeout_seconds', 'read_timeout_seconds', 'idle_timeout_seconds'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 120:
                raise ValueError('摄像头超时必须大于0且不超过120秒。')
        if self.idle_timeout_seconds <= self.read_timeout_seconds:
            raise ValueError('页面空闲关闭时间须长于读取超时。')

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding='utf-8')))


@dataclass(frozen=True)
class CapturedFrame:
    sequence: int
    captured_at: float
    bgr: np.ndarray


class CameraSource:
    _lease_lock = threading.Lock()
    _leases = set()

    def __init__(self, config: CameraConfig):
        self.config = config
        self.info = {}
        self.error = None
        self._condition = threading.Condition()
        self._stopping = threading.Event()
        self._done = threading.Event()
        self._latest = None
        self._last_received = time.monotonic()
        self._last_poll = self._last_received
        self._leased = False
        self._process = None
        self._reader_thread = None
        self._watchdog_thread = None
        with self._lease_lock:
            if config.index in self._leases:
                raise ValueError('该摄像头已在本应用的其他会话中使用，请先停止另一处检测。')
            self._leases.add(config.index)
            self._leased = True
        try:
            script = Path(__file__).with_name('camera_capture.py')
            self._process = subprocess.Popen(
                [sys.executable, '-u', str(script), '--index', str(config.index), '--backend', config.backend,
                 '--width', str(config.width), '--height', str(config.height), '--fps', str(config.fps)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            )
            self._reader_thread = threading.Thread(target=self._read_loop, name='camera-latest-frame', daemon=True)
            self._reader_thread.start()
            deadline = time.monotonic() + config.open_timeout_seconds
            with self._condition:
                while self._latest is None and self.error is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise ValueError('摄像头打开或首次取帧超时，请检查权限、占用或切换采集后端。')
                    self._condition.wait(remaining)
                if self.error:
                    raise ValueError(self.error)
            self._last_poll = time.monotonic()
            self._watchdog_thread = threading.Thread(target=self._watch, name='camera-timeout', daemon=True)
            self._watchdog_thread.start()
        except Exception:
            self.release()
            raise

    def _unlease(self):
        with self._lease_lock:
            if self._leased:
                self._leases.discard(self.config.index)
                self._leased = False

    def _read_exact(self, size):
        output = bytearray()
        while len(output) < size:
            chunk = self._process.stdout.read(size - len(output))
            if not chunk:
                return None
            output.extend(chunk)
        return bytes(output)

    def _read_loop(self):
        try:
            while not self._stopping.is_set():
                header = self._read_exact(HEADER.size)
                if header is None:
                    break
                kind, size = HEADER.unpack(header)
                if size > 16 * 1024**2:
                    raise CameraReadError('摄像头返回的数据包超过上限。')
                body = self._read_exact(size)
                if body is None:
                    break
                if kind == b'I':
                    self.info = json.loads(body)
                elif kind == b'E':
                    raise CameraReadError(body.decode('utf-8', errors='replace'))
                elif kind == b'F':
                    sequence, stamp = FRAME.unpack(body[:FRAME.size])
                    bgr = cv2.imdecode(np.frombuffer(body[FRAME.size:], dtype=np.uint8), cv2.IMREAD_COLOR)
                    if bgr is None or not math.isfinite(stamp):
                        raise CameraReadError('摄像头画面或时间无效。')
                    with self._condition:
                        self._latest = CapturedFrame(sequence, stamp, bgr)
                        self._last_received = time.monotonic()
                        self._condition.notify_all()
                else:
                    raise CameraReadError('摄像头返回未知数据。')
            if not self._stopping.is_set():
                raise CameraReadError('摄像头采集进程意外结束。')
        except Exception as exc:
            if not self._stopping.is_set():
                self.error = str(exc)
        finally:
            self._stopping.set()
            try:
                self._kill_process()
                if self._process:
                    self._process.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                self.error = self.error or '摄像头采集进程未能正常关闭。'
            finally:
                if self._process:
                    self._process.stdout.close()
                self._unlease()
                self._done.set()
                with self._condition:
                    self._condition.notify_all()

    def _kill_process(self):
        try:
            if self._process and self._process.poll() is None:
                self._process.kill()
        except OSError:
            pass

    def _fail(self, message):
        self.error = message
        self._stopping.set()
        self._kill_process()

    def _watch(self):
        while not self._stopping.wait(0.1):
            now = time.monotonic()
            if now - self._last_received > self.config.read_timeout_seconds:
                self._fail('摄像头取帧超时，已关闭设备；未将断流判定为火情恢复。')
            elif now - self._last_poll > self.config.idle_timeout_seconds:
                self._fail('页面长时间未刷新，摄像头已自动关闭，请重新开始检测。')

    def read_latest(self, after_sequence):
        self._last_poll = time.monotonic()
        with self._condition:
            if self.error:
                raise CameraReadError(self.error)
            if self._stopping.is_set():
                raise CameraReadError('摄像头已关闭。')
            if self._latest is not None and self._latest.sequence > after_sequence:
                if time.monotonic() - self._latest.captured_at > self.config.read_timeout_seconds:
                    raise CameraReadError('摄像头画面已过时，请重新开始检测。')
                return self._latest
            return None

    def isOpened(self):
        return not self._stopping.is_set() and self._process is not None and self._process.poll() is None

    def release(self):
        self._stopping.set()
        self._kill_process()
        if self._reader_thread and self._reader_thread is not threading.current_thread():
            self._reader_thread.join(timeout=4)
        if self._watchdog_thread and self._watchdog_thread is not threading.current_thread():
            self._watchdog_thread.join(timeout=1)
        if self._reader_thread is None:
            self._unlease()

    def metadata(self):
        return {'input_type': 'camera', 'clock': 'host_monotonic', 'camera_config': asdict(self.config), **self.info}
