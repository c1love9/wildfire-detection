"""Media import, sequential video sampling and bounded, explicit USB collection."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
import uuid
from io import BytesIO
from pathlib import Path

import cv2
from PIL import Image

from camera_source import CameraConfig, CameraSource
from dataset_store import IMAGE_SUFFIXES, VIDEO_SUFFIXES, file_sha, now


def probe_video(path):
    path = Path(path)
    if path.suffix.lower() not in VIDEO_SUFFIXES or not path.is_file():
        raise ValueError('请选择有效的MP4/MOV/AVI/MKV文件。')
    binary = shutil.which('ffprobe')
    if binary is None:
        raise OSError('请安装FFmpeg/ffprobe并加入PATH后导入视频。')
    try:
        result = subprocess.run([binary, '-v', 'error', '-count_frames', '-select_streams', 'v:0', '-show_entries',
                                 'stream=width,height,duration,nb_read_frames:format=duration', '-of', 'json', str(path)],
                                capture_output=True, timeout=120, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        info = json.loads(result.stdout)
        stream = info['streams'][0]
        duration = float(stream.get('duration') or info['format']['duration'])
        frames = int(stream['nb_read_frames'])
        width, height = int(stream['width']), int(stream['height'])
        if result.returncode or not math.isfinite(duration) or duration <= 0 or min(frames, width, height) <= 0:
            raise ValueError('视频没有有效的画面或时长。')
        return dict(duration=duration, frames=frames, width=width, height=height)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
        raise ValueError('视频无法完整读取或缺少有效时间信息。') from exc


def import_paths(store, paths, batch_id, source_names=None):
    results = []
    for source in paths:
        path = Path(source)
        name = (source_names or {}).get(path.name, path.name)
        try:
            if path.suffix.lower() in IMAGE_SUFFIXES:
                result = store.import_image(path, batch_id, source_name=name)
            elif path.suffix.lower() in VIDEO_SUFFIXES:
                result = store.import_video(path, batch_id, source_name=name)
            else:
                raise ValueError('不支持此扩展名。')
            results.append({'文件': name, '结果': '已导入' if result['added'] else '重复，已关联来源', 'asset_id': result['asset_id']})
        except (ValueError, OSError) as exc:
            results.append({'文件': name, '结果': str(exc), 'asset_id': ''})
    return results


def extract_frames(store, asset_id, batch_id, interval=1.0, maximum=500):
    if isinstance(interval, bool) or not isinstance(interval, (int, float)) or not math.isfinite(interval) or not 0.1 <= interval <= 3600:
        raise ValueError('抽帧间隔须为0.1～3600秒。')
    if type(maximum) is not int or not 1 <= maximum <= 2000:
        raise ValueError('单次抽帧上限须为1～2000。')
    asset = next((row for row in store.assets() if row['id'] == asset_id and row['kind'] == 'video'), None)
    if asset is None:
        raise ValueError('请选择已导入的视频。')
    snapshot = store.snapshot()
    if not any(row['asset_id'] == asset_id and row['batch_id'] == batch_id for row in snapshot['occurrences']):
        raise ValueError('须使用原视频的采集批次抽帧。')
    source = store.asset_path(asset)
    if file_sha(source) != asset['sha256']:
        raise OSError('原视频内容已改变，未抽帧；请核对文件完整性。')
    capture = cv2.VideoCapture(str(source))
    selected = added = number = 0
    next_time = 0.0
    previous = -1.0
    try:
        while selected < maximum:
            ok, bgr = capture.read()
            if not ok:
                break
            seconds = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if not math.isfinite(seconds) or seconds < 0 or seconds <= previous:
                raise ValueError('视频时间戳无效或不连续，已停止抽帧；此前已导入的帧保留，请检查原视频。')
            previous = seconds
            if seconds + 1e-6 >= next_time:
                output = BytesIO()
                Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(output, format='PNG')
                result = store.import_image(output.getvalue(), batch_id, source_name=f'{asset_id[:12]}_frame_{number}.png',
                                            parent_id=asset_id, frame_number=number, media_seconds=seconds,
                                            metadata={'sampling': 'sequential_source_pts', 'interval_seconds': interval})
                selected += 1
                added += int(result['added'])
                next_time = (math.floor(seconds / interval) + 1) * interval
            number += 1
    finally:
        capture.release()
    if not selected:
        raise ValueError('视频没有可抽取的画面。')
    return {'sampled': selected, 'added': added, 'duplicates': selected - added, 'decoded': number}


def capture_snapshot(store, batch_id, config=None, source_factory=None):
    if store.batch(batch_id)['origin'] not in {'own', 'demo'}:
        raise ValueError('摄像头采集须使用自采或演示批次。')
    source = (source_factory or CameraSource)(config or CameraConfig())
    try:
        packet = source.read_latest(0)
        if packet is None:
            raise ValueError('摄像头尚无画面。')
        encoded = BytesIO()
        Image.fromarray(cv2.cvtColor(packet.bgr, cv2.COLOR_BGR2RGB)).save(encoded, format='PNG')
        return store.import_image(encoded.getvalue(), batch_id, source_name='camera_snapshot.png',
                                  metadata={'type': 'usb_snapshot', 'capture_id': uuid.uuid4().hex, 'recorded_at': now(), **source.metadata()})
    finally:
        source.release()


def capture_clip(store, batch_id, seconds=5.0, config=None, source_factory=None, clock=time.monotonic, sleep=time.sleep):
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 1 <= seconds <= 30:
        raise ValueError('单次采集短片须为1～30秒。')
    if store.batch(batch_id)['origin'] not in {'own', 'demo'}:
        raise ValueError('摄像头采集须使用自采或演示批次。')
    binary = shutil.which('ffmpeg')
    if binary is None:
        raise OSError('缺少FFmpeg，无法采集可回放的H.264短片。')
    if shutil.disk_usage(store.root).free < 256 * 1024**2:
        raise OSError('可用空间不足256MB。')
    directory = store.root / ('capture_' + uuid.uuid4().hex)
    directory.mkdir()
    source = None
    frames = []
    generated = []
    total_bytes = sequence = 0
    last_slot = -1
    initial = previous = None
    started_at = now()
    try:
        source = (source_factory or CameraSource)(config or CameraConfig())
        deadline = clock() + seconds + 5
        while clock() <= deadline:
            packet = source.read_latest(sequence)
            if packet is None:
                sleep(0.01)
                continue
            sequence = packet.sequence
            if initial is None:
                initial = packet.captured_at
            if previous is not None and not 0 < packet.captured_at - previous <= 1.0:
                raise ValueError('采集画面间隔异常，本次短片未入库，请重试。')
            previous = packet.captured_at
            elapsed = packet.captured_at - initial
            if elapsed >= seconds - 1e-6:
                break
            slot = math.floor(elapsed * 10 + 1e-6)
            if slot > last_slot:
                last_slot = slot
                ok, jpeg = cv2.imencode('.jpg', packet.bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if not ok:
                    raise OSError('采集画面编码失败。')
                total_bytes += len(jpeg)
                if len(frames) >= 300 or total_bytes > 64 * 1024**2:
                    raise OSError('采集超过300帧或64MB缓存上限。')
                path = directory / f'{len(frames):04d}.jpg'
                generated.append(path)
                path.write_bytes(jpeg.tobytes())
                frames.append((elapsed, path.name))
            sleep(0.005)
        if initial is None or previous - initial < seconds - 1e-6 or len(frames) < 2:
            raise ValueError('摄像头未完成指定时长，本次短片未入库。')
        metadata = {'type': 'usb_clip', 'capture_id': uuid.uuid4().hex, 'started_at': started_at, 'requested_seconds': seconds,
                    'sample_times_seconds': [row[0] for row in frames], **source.metadata()}
        source.release()
        source = None
        manifest = directory / 'frames.ffconcat'
        lines = ['ffconcat version 1.0']
        for index, (stamp, name) in enumerate(frames):
            end = frames[index + 1][0] if index + 1 < len(frames) else seconds
            lines.extend([f"file '{name}'", 'option framerate 1000', f'duration {end - stamp:.9f}'])
        lines.extend([f"file '{frames[-1][1]}'", 'option framerate 1000'])
        generated.append(manifest)
        manifest.write_text('\n'.join(lines) + '\n', encoding='ascii')
        output = directory / 'capture.mp4'
        generated.append(output)
        result = subprocess.run([binary, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', str(manifest),
                                 '-t', str(seconds), '-vf', 'fps=10,scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p',
                                 '-c:v', 'libx264', '-threads', '2', '-preset', 'veryfast', '-movflags', '+faststart', str(output)],
                                capture_output=True, timeout=60, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        if result.returncode:
            raise OSError('采集短片编码失败，未入库。')
        return store.import_video(output, batch_id, source_name='camera_clip.mp4', metadata=metadata)
    finally:
        if source is not None:
            source.release()
        # Delete only known generated files, after resolving the allowed root.
        if directory.resolve().is_relative_to(store.root) and directory.resolve() != store.root:
            for path in generated:
                if path.resolve().is_relative_to(directory.resolve()):
                    path.unlink(missing_ok=True)
            directory.rmdir()
