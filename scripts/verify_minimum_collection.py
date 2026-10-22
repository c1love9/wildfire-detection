"""Validate actual selected media, provenance, grouping and draft label geometry."""
import concurrent.futures
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import torch
import cv2
from PIL import Image
from dataset_store import DatasetStore, file_sha, now

OUT = ROOT / 'data/minimum_collection'


def main():
    registration = json.loads((OUT / 'registration.json').read_text(encoding='utf-8'))
    rows = registration['records']; images = [row for row in rows if row['kind'] == 'image']; videos = [row for row in rows if row['kind'] == 'video']
    assert len(images) == 700 and len(videos) == 300
    assert sum(row['source'] == 'dfire' for row in images) == 200
    assert len({row['id'] for row in rows}) == len(rows)
    source_split = {}; groups = {}; spans = defaultdict(list)
    fingerprints = set(); missing_positive = []
    for row in rows:
        path = ROOT / row['path']; assert file_sha(path) == row['sha256']
        assert row['original_provenance'] == 'public'
        group = row['collection_group']
        assert group not in groups or groups[group] == row['split']; groups[group] = row['split']
        if row.get('source_video'):
            source = row['source_video']
            assert source not in source_split or source_split[source] == row['split']; source_split[source] = row['split']
            if row['kind'] == 'video': spans[source].append((row['source_start_seconds'], row['source_end_seconds']))
        if row['kind'] == 'image':
            with Image.open(path) as original:
                assert original.size == (row['width'], row['height']); original.verify()
            text = (OUT / 'yolo_draft/labels' / row['split'] / (row['id'] + '.txt')).read_text(encoding='utf-8')
            if not text.strip() and row['source'] != 'dfire': missing_positive.append(row['id'])
            for line in text.splitlines():
                label, x, y, width, height = map(float, line.split())
                assert label in (0, 1) and all(math.isfinite(value) for value in (x, y, width, height))
                assert 0 < width <= 1 and 0 < height <= 1
                assert -1e-7 <= x-width/2 < x+width/2 <= 1+1e-7
                assert -1e-7 <= y-height/2 < y+height/2 <= 1+1e-7
        key = (row['kind'], row['sha256']); assert key not in fingerprints; fingerprints.add(key)
    for source, values in spans.items():
        ordered = sorted(values)
        assert all(left[1] <= right[0] + 1e-6 for left, right in zip(ordered, ordered[1:])), source
    def check_video(row):
        path = ROOT / row['path']
        result = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
            'stream=codec_name,pix_fmt:format=duration', '-of', 'json', str(path)], creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0))
        duration = float(result['format']['duration']); stream = result['streams'][0]
        assert stream['codec_name'] == 'h264' and stream['pix_fmt'] == 'yuv420p'
        assert abs(duration - 3) < .08
        capture = cv2.VideoCapture(str(path)); count = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok: break
                assert frame is not None and frame.size; count += 1
        finally: capture.release()
        assert count >= 44, (path, count)
        return duration, count
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        checked = list(pool.map(check_video, videos))
    duration = sum(row[0] for row in checked); assert duration >= 600
    dataset = DatasetStore(ROOT / 'data/fire_dataset')
    assert len(dataset.assets()) == 1000
    assert dataset.statistics()['derived_images'] == 500
    report = {'status': 'passed', 'verified_at': now(), 'images_checked': len(images), 'videos_fully_decoded': len(videos),
        'video_duration_seconds': duration, 'decoded_video_frames': sum(row[1] for row in checked),
        'media_sha256_and_image_shapes': True, 'nonoverlapping_clips_per_source': True,
        'same_original_video_and_known_scene_groups_never_cross_splits': True, 'draft_yolo_geometry_valid': True,
        'unresolved_positive_images_not_negative_ground_truth': len(missing_positive),
        'annotation_status': 'All labels are candidates; no student human approval fabricated',
        'collection_rule': 'Public material curation counted as self-collection only under user-reported teacher approval',
        'coverage_status': 'Indoor/outdoor, day/night/backlight, different distances, level/aerial views selected; literal early sparks and combustion-stage reference still require review',
        'split_images': dict(Counter(row['split'] for row in images)), 'split_videos': dict(Counter(row['split'] for row in videos)),
        'source_original_videos': len(source_split), 'human_approved_images': 0}
    (ROOT / 'reports/minimum_collection_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
