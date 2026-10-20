"""Bounded pre-event JPEG buffer and asynchronous H.264 event clips."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class RecordingConfig:
    enabled: bool = True
    pre_seconds: float = 5.0
    post_seconds: float = 10.0
    output_fps: int = 30
    jpeg_quality: int = 85
    max_buffer_mb: float = 32.0
    max_buffer_frames: int = 300
    max_pending_clips: int = 4
    min_free_mb: float = 256.0
    ffmpeg_binary: str = ''

    def __post_init__(self):
        if type(self.enabled) is not bool or not isinstance(self.ffmpeg_binary, str):
            raise ValueError('录制开关或编码器配置格式不正确。')
        for name in ('pre_seconds', 'post_seconds', 'max_buffer_mb', 'min_free_mb'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('录制参数必须是有效数字。')
        if not 0 <= self.pre_seconds <= 30 or not 0 < self.post_seconds <= 60 or self.pre_seconds + self.post_seconds > 60:
            raise ValueError('前置时间应为0～30秒，后置时间须大于0，且片段总时长不超过60秒。')
        for name, lower, upper in (('output_fps', 1, 30), ('jpeg_quality', 1, 95),
                                   ('max_buffer_frames', 1, 1800), ('max_pending_clips', 1, 8)):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f'录制参数{name}应为{lower}～{upper}的整数。')
        if not 0 < self.max_buffer_mb <= 256 or self.min_free_mb < 0:
            raise ValueError('缓存大小须大于0且不超过256MB，剩余磁盘阈值不能为负。')

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding='utf-8')))


def resolve_media_asset(root, relative_path, suffix):
    """Only serve a stored asset from the configured media directory."""
    if not relative_path:
        return None
    try:
        root = Path(root).resolve()
        path = (root / relative_path).resolve()
        return path if path.is_relative_to(root) and path.suffix.lower() == suffix and path.is_file() else None
    except (OSError, ValueError):
        return None


@dataclass(frozen=True)
class _Frame:
    number: int
    seconds: float
    jpeg: bytes
    estimated: bool


@dataclass
class _Clip:
    event_id: str
    directory: Path
    snapshot: str | None
    details: dict
    frames: list = field(default_factory=list)


class EventRecorder:
    def __init__(self, config: RecordingConfig, root, store, source_fps=30.0):
        self.config, self.store = config, store
        self.root = Path(root).resolve()
        self.source_fps = source_fps
        self.buffer = deque()
        self.buffer_bytes = 0
        self.max_buffer_bytes_seen = 0
        self.max_buffer_frames_seen = 0
        self.last_error = None
        self._sample_slot = -1
        self._last_input = None
        self._last_interval = 1 / source_fps
        self._last_event_id = None
        self._active = []
        self._futures = []
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='event-clip')
        self._finished = False
        self.encoder = shutil.which(config.ffmpeg_binary or os.environ.get('FIRE_ALARM_FFMPEG', 'ffmpeg'))
        if config.enabled and self.encoder is None:
            self.last_error = '未找到FFmpeg，报警将只保存截图；请安装支持libx264的FFmpeg或设置编码器路径。'

    def _persist(self, clip, status, error=None, video=None):
        try:
            self.store.save_media(clip.event_id, clip.snapshot, video, status, clip.details, error)
        except Exception:
            self.last_error = '媒体信息写入日志失败，请检查日志目录或磁盘空间。'

    def _space(self):
        self.root.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(self.root).free < self.config.min_free_mb * 1024**2:
            raise OSError('磁盘可用空间低于录制配置的保留值。')

    def _append(self, clip, frame):
        if clip.frames and clip.frames[-1][0] == frame.number:
            return
        if len(clip.frames) >= math.ceil((self.config.pre_seconds + self.config.post_seconds) * self.config.output_fps) + 2:
            raise OSError('片段帧数超过录制上限。')
        if len(clip.frames) % 30 == 0:
            self._space()
        name = f'{len(clip.frames):06d}.jpg'
        destination = clip.directory / 'frames' / name
        try:
            destination.write_bytes(frame.jpeg)
        except OSError:
            try:
                destination.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        clip.frames.append((frame.number, frame.seconds, name))
        clip.details['time_estimated'] |= frame.estimated

    def _start(self, event_id, frame):
        if not re.fullmatch('[0-9a-f]{32}', event_id):
            self.last_error = '事件编号无效，未保存媒体。'
            return
        directory = self.root / event_id
        details = dict(trigger_frame=frame.number, trigger_seconds=frame.seconds,
                       target_start_seconds=frame.seconds - self.config.pre_seconds,
                       target_end_seconds=frame.seconds + self.config.post_seconds,
                       time_estimated=frame.estimated, config=asdict(self.config))
        clip = _Clip(event_id, directory, None, details)
        try:
            self._space()
            directory.mkdir(exist_ok=False)
            (directory / 'snapshot.jpg').write_bytes(frame.jpeg)
            clip.snapshot = f'{event_id}/snapshot.jpg'
            self._futures = [future for future in self._futures if not future.done()]
            if self.encoder is None:
                raise OSError('未找到FFmpeg，截图已保存，片段无法编码。')
            if len(self._active) + len(self._futures) >= self.config.max_pending_clips:
                raise OSError('待保存片段已达上限，截图已保存，本事件未录制视频。')
            (directory / 'frames').mkdir()
            for cached in self.buffer:
                self._append(clip, cached)
            # A forced trigger frame can be newer than the regular sampling tick.
            self._append(clip, frame)
            self._persist(clip, 'recording')
            self._active.append(clip)
        except Exception as exc:
            self.last_error = f'报警媒体保存失败：{exc}'
            self._persist(clip, 'failed', str(exc))
            self._cleanup(clip)

    def add_frame(self, frame, event_id):
        if self._finished or not self.config.enabled:
            return
        new_event = event_id is not None and event_id != self._last_event_id
        self._last_event_id = event_id
        if self._last_input is not None:
            self._last_interval = frame.media_seconds - self._last_input
        self._last_input = frame.media_seconds
        sample_slot = math.floor(frame.media_seconds * self.config.output_fps + 1e-8)
        due = sample_slot > self._sample_slot
        if (not due or self.encoder is None) and not new_event:
            # Even an unsampled frame can complete a media-time window.
            self._complete_windows(frame.media_seconds)
            return
        try:
            encoded = BytesIO()
            Image.fromarray(frame.rgb).save(encoded, format='JPEG', quality=self.config.jpeg_quality)
            sampled = _Frame(frame.number, frame.media_seconds, encoded.getvalue(), frame.time_estimated)
            if due:
                self._sample_slot = sample_slot
                if self.encoder:
                    self.buffer.append(sampled)
                    self.buffer_bytes += len(sampled.jpeg)
                    cfg = self.config
                    while self.buffer and (self.buffer[0].seconds < frame.media_seconds - cfg.pre_seconds - 1e-9 or
                                           len(self.buffer) > cfg.max_buffer_frames or self.buffer_bytes > cfg.max_buffer_mb * 1024**2):
                        self.buffer_bytes -= len(self.buffer.popleft().jpeg)
                    self.max_buffer_bytes_seen = max(self.max_buffer_bytes_seen, self.buffer_bytes)
                    self.max_buffer_frames_seen = max(self.max_buffer_frames_seen, len(self.buffer))
            # Complete older windows before allocating a slot for a new event.
            self._complete_windows(frame.media_seconds)
            if new_event:
                self._start(event_id, sampled)
            for clip in list(self._active):
                if frame.media_seconds < clip.details['target_end_seconds'] - 1e-9:
                    try:
                        self._append(clip, sampled)
                    except Exception as exc:
                        self.last_error = f'报警片段写入失败：{exc}'
                        self._persist(clip, 'failed', str(exc))
                        self._active.remove(clip)
                        self._cleanup(clip)
        except Exception as exc:
            self.last_error = f'报警画面编码失败：{exc}'

    def _complete_windows(self, seconds):
        for clip in list(self._active):
            if seconds + 1e-9 >= clip.details['target_end_seconds']:
                self._queue(clip, clip.details['target_end_seconds'], 'window_complete')

    def _queue(self, clip, end_seconds, reason):
        self._active.remove(clip)
        first, last = clip.frames[0], clip.frames[-1]
        end_seconds = max(last[1] + 1e-6, min(end_seconds, clip.details['target_end_seconds']))
        clip.details.update(start_frame=first[0], start_seconds=first[1], end_frame=last[0], end_seconds=end_seconds,
                            source_frames=len(clip.frames), duration_seconds=end_seconds - first[1], completion_reason=reason,
                            pre_truncated=first[1] > clip.details['target_start_seconds'] + 1 / self.config.output_fps + 1e-6,
                            post_truncated=end_seconds < clip.details['target_end_seconds'] - 1 / self.config.output_fps - 1e-6)
        self._persist(clip, 'encoding')
        self._futures.append(self._executor.submit(self._encode, clip))

    def _encode(self, clip):
        try:
            lines = ['ffconcat version 1.0']
            for index, (_, seconds, name) in enumerate(clip.frames):
                next_time = clip.frames[index + 1][1] if index + 1 < len(clip.frames) else clip.details['end_seconds']
                lines.extend([f"file 'frames/{name}'", 'option framerate 1000', f'duration {next_time - seconds:.9f}'])
            lines.extend([f"file 'frames/{clip.frames[-1][2]}'", 'option framerate 1000'])
            manifest = clip.directory / 'frames.ffconcat'
            manifest.write_text('\n'.join(lines) + '\n', encoding='ascii')
            output = clip.directory / 'clip.pending.mp4'
            command = [self.encoder, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0',
                       '-i', str(manifest), '-t', f"{clip.details['duration_seconds']:.9f}",
                       '-vf', f'fps={self.config.output_fps},scale=trunc(iw/2)*2:trunc(ih/2)*2:in_range=full:out_range=tv,format=yuv420p',
                       '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-threads', '2',
                       '-pix_fmt', 'yuv420p', '-color_range', 'tv', '-movflags', '+faststart', str(output)]
            self._space()
            completed = subprocess.run(command, capture_output=True, timeout=60,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            if completed.returncode or not output.is_file() or output.stat().st_size == 0:
                raise OSError('FFmpeg编码失败：' + completed.stderr.decode('utf-8', errors='replace')[-400:])
            output.replace(clip.directory / 'clip.mp4')
            status = 'partial' if clip.details['pre_truncated'] or clip.details['post_truncated'] else 'ready'
            self._persist(clip, status, video=f'{clip.event_id}/clip.mp4')
        except Exception as exc:
            self.last_error = f'报警片段编码失败：{exc}'
            self._persist(clip, 'failed', str(exc))
        finally:
            self._cleanup(clip)

    def _cleanup(self, clip):
        # Delete only generated, individually known files under this event.
        directory = clip.directory.resolve()
        if not directory.is_relative_to(self.root) or directory == self.root:
            self.last_error = '媒体临时目录不在配置目录内，未执行清理。'
            return
        for relative in [*(f'frames/{row[2]}' for row in clip.frames), 'frames.ffconcat', 'clip.pending.mp4']:
            path = (directory / relative).resolve()
            if path.is_relative_to(directory):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
        try:
            (directory / 'frames').rmdir()
        except OSError:
            pass

    def finish(self, reason):
        if self._finished:
            return
        self._finished = True
        end_seconds = (self._last_input or 0) + max(1 / self.config.output_fps, self._last_interval)
        for clip in list(self._active):
            self._queue(clip, end_seconds, reason)
        self.buffer.clear()
        self.buffer_bytes = 0
        self._executor.shutdown(wait=False)

    def wait(self, timeout=90):
        _, pending = wait(self._futures, timeout=timeout)
        if pending:
            raise TimeoutError('等待报警片段编码超时。')
