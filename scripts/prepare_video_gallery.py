"""Prepare continuous raw-video excerpts for demonstration, without detection overlays."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import cv2

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = [
    ('01_indoor_fire', '01｜室内明火', '清晰明火，适合演示连续确认与报警升级。',
     'firesense/videos/fire_videos.1406/pos/posVideo4.873.avi', 2., 22., ['fire']),
    ('02_building_fire_smoke', '02｜建筑火焰与烟雾', '观察同一画面内的火焰、烟雾及目标数量变化。',
     'firesense/videos/smoke_videos.1407/pos/testpos02.818.avi', 0., 14., ['fire', 'smoke']),
    ('03_forest_smoke', '03｜林地烟雾', '没有明显火焰时，持续烟雾也能独立触发报警。',
     'kmu/videos/Wildfire_smoke/wildfire_smoke_1.avi', 5., 22., ['smoke']),
    ('04_night_fire', '04｜夜间明火', '展示低光照条件下的明火检测。',
     'kmu/videos/Indoor_outdoor(short_distance)flame/outdoor_night_30m_gasoline_CCD_001.avi', 10., 22., ['fire']),
]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(root):
    root = Path(root).resolve()
    executable = shutil.which('ffmpeg')
    if not executable:
        raise RuntimeError('准备演示视频需要本机FFmpeg。')
    destination = root/'demo-videos/curated'
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {'schema_version': 1, 'purpose': 'Curated function demonstration; not independent performance evaluation.',
                'model_for_selection': 'train_984a486d3e6f459eb8eea567da186ef0 best',
                'detection_overlays_in_input': False, 'editing': 'Continuous excerpts, resampled to 10 fps; no looping or repeated segments.',
                'samples': []}
    for key, label, description, relative, start, seconds, content in SAMPLES:
        source = root/'data/public_sources'/relative
        if not source.is_file():
            raise FileNotFoundError(source)
        target = destination/(key+'.mp4')
        temporary = target.with_suffix('.pending.mp4')
        subprocess.run([executable, '-hide_banner', '-loglevel', 'error', '-y', '-ss', str(start), '-i', str(source),
                        '-t', str(seconds), '-an', '-vf', 'fps=10,scale=trunc(iw/2)*2:trunc(ih/2)*2',
                        '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart', str(temporary)], check=True, timeout=120)
        temporary.replace(target)
        capture = cv2.VideoCapture(str(target))
        ok, frame = capture.read()
        fps, count = capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if not ok or fps != 10 or count < 100:
            raise RuntimeError('演示视频解码或时长检查失败：'+target.name)
        poster = destination/(key+'.jpg')
        if not cv2.imwrite(str(poster), frame):
            raise OSError('预览图无法保存。')
        manifest['samples'].append({'file': target.name, 'poster': poster.name, 'label': label,
            'description': description, 'content': content, 'duration': count/fps, 'fps': fps,
            'width': frame.shape[1], 'height': frame.shape[0], 'sha256': digest(target),
            'source': source.relative_to(root).as_posix(), 'source_sha256': digest(source),
            'source_start_seconds': start, 'source_duration_seconds': seconds,
            'source_url': 'https://zenodo.org/records/836749' if relative.startswith('firesense/') else 'https://cvpr.kmu.ac.kr/Dataset/Dataset.htm'})
    pending = destination/'manifest.json.pending'
    pending.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    pending.replace(destination/'manifest.json')
    print(json.dumps({'videos': len(manifest['samples']), 'folder': str(destination), 'overlays_added': False}, ensure_ascii=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    prepare(parser.parse_args().root)
