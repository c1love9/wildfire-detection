"""Prepare traceable images/clips; original public provenance is never overwritten.

Stages can be resumed per source while large archives are still downloading.
No output is marked as a student's human-reviewed annotation.
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw
from dataset_store import file_sha, now

OUT = ROOT / 'data/minimum_collection'
QUOTAS = {'kmu': (250, 150), 'firesense': (200, 100), 'flame2': (50, 50)}
PAGES = {'kmu': 'https://cvpr.kmu.ac.kr/Dataset/Dataset.htm', 'firesense': 'https://zenodo.org/records/836749',
         'flame2': 'https://doi.org/10.21227/swyw-6j78', 'dfire': 'https://github.com/gaia-solutions-on-demand/DFireDataset'}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    pending.replace(path)


def profile(row):
    name, path = row['name'].lower(), row['path'].lower()
    config = {'setting': '室外', 'lighting': '白天', 'scale': '发展期明火', 'distance': '中景', 'view': '平视'}
    content = ['fire']; start, end = 1., row['duration'] - 1
    group = row['id']
    detail = '根据源视频缩略图筛选，逐帧燃烧阶段仍须复核。'
    if row['source'] == 'kmu':
        if name.startswith('outdoor_'):
            start = 5.
            group = 'kmu_controlled_outdoor_site'
            config['lighting'] = '夜间' if 'night' in name else '白天'
            config['distance'] = '近景' if '10m' in name else '中景' if '20m' in name else '远景'
            if 'gasoline' in name: content = ['fire', 'smoke']
            ends = {'outdoor_daytime_10m_gasoline_ccd_001.avi': 110, 'outdoor_night_10m_gasoline_ccd_002.avi': 30,
                    'outdoor_night_10m_heptane_ccd_002.avi': 17, 'outdoor_night_20m_gasoline_ccd_001.avi': 130,
                    'outdoor_night_20m_heptane_ccd_002.avi': 32, 'outdoor_night_30m_gasoline_ccd_001.avi': 210,
                    'outdoor_night_30m_heptane_ccd_001.avi': 150}
            end = min(end, ends.get(name, end))
        elif name.startswith('indoor_night_20m'):
            group = 'kmu_controlled_warehouse'; config.update(setting='室内', lighting='夜间')
            start = 8.; end = min(end, 64. if name.endswith('001.avi') else 150.)
        elif name in ('flame3.avi', 'flame4.avi', 'flame5.avi'):
            group = 'kmu_small_flame_lab'; config.update(setting='室内', lighting='逆光' if name != 'flame5.avi' else '未说明', distance='近景')
            detail = '室内小型明火；flame3/4窗户亮背景形成背光。不能仅凭画面判定为早期火星。'
        elif name == 'flame2.avi':
            config.update(setting='室内', lighting='未说明', distance='近景')
        elif name == 'flame1.avi':
            config.update(distance='近景'); start = 2.
        elif 'smoke' in name:
            content = ['smoke']; config.update(scale='充分燃烧浓烟')
            if name.startswith('indoor_'):
                group = 'kmu_office_smoke'; config.update(setting='室内', lighting='夜间' if 'night' in name else '白天', distance='近景')
                start = 8. if 'night' in name else 4.
                detail = '室内烟雾发生场景，阶段填写仅表示烟雾外观，不证明充分燃烧。'
            else:
                config.update(distance='远景')
                if name in ('wildfire_smoke_1.avi', 'wildfire_smoke_2.avi'): group = 'kmu_wildfire_camera_1'
                if name == 'wildfire_smoke_4.avi': return None  # Smoke not clearly visible in the review sheet.
    elif row['source'] == 'firesense':
        if '/pos/' not in path: return None
        if 'smoke_videos' in path:
            number = int(name.split('.')[0].removeprefix('testpos'))
            if number in (3, 5, 6, 7): return None  # Includes intermittent burnt-in masks, missed by sparse thumbnails.
            content = ['smoke']; config.update(scale='充分燃烧浓烟', distance='近景')
            if number == 2: content = ['fire', 'smoke']; config.update(distance='中景')
            if number == 3:
                group = 'firesense_fence_burn'; config.update(lighting='逆光', distance='中景')
            if number in (10, 11): group = 'firesense_barrel_smoke_site'
            if number in (12, 13): group = 'firesense_stair_smoke_site'
            if number == 8:
                detail = '烟火发烟装置，有喷焰但不是自然火灾的早期火星，不据此补造火星阶段。'
            if number == 4: content = ['fire', 'smoke']
        else:
            # Flame source profiles are visually reviewed after download.
            custom = ROOT / 'configs/collection_firesense.json'
            if not custom.exists(): return None
            overrides = json.loads(custom.read_text(encoding='utf-8')).get(row['name'])
            if not overrides or not overrides.get('use'): return None
            config.update(overrides.get('scene', {})); content = overrides.get('content', ['fire'])
            group = overrides.get('group', group); start = overrides.get('start', start); end = min(end, overrides.get('end', end))
            detail = overrides.get('note', detail)
    else:
        if 'ir' in name and 'rgb' not in name: return None
        group = 'flame2_prescribed_burn_2021'; content = ['fire', 'smoke']
        config.update(distance='远景', view='俯视', scale='充分燃烧浓烟')
        start, end = 0., min(153., row['duration'] - 3)
        content = ['smoke']
        detail = '逐10秒复核，仅选0至153秒的可见烟雾；后段镜头转向无火烟建筑，已排除。'
    return {**row, 'scene': config, 'content': content, 'group': group, 'start': start, 'end': end, 'visual_note': detail}


def allocate(rows, total, kind):
    spacing = 2 if kind == 'images' else 3
    capacities = [max(0, math.floor((row['end'] - row['start']) / spacing)) for row in rows]
    counts = [0] * len(rows)
    while sum(counts) < total:
        advanced = False
        for index in range(len(rows)):
            if counts[index] < capacities[index] and sum(counts) < total:
                counts[index] += 1; advanced = True
        if not advanced: raise ValueError(f'Not enough nonoverlapping source time for {total} {kind}')
    return counts


def frame_at(path, seconds):
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        ok, image = capture.read()
        if not ok: raise OSError(f'Frame unavailable at {seconds}: {path}')
        return image
    finally:
        capture.release()


def collect(source, replace=False):
    target = OUT / ('selected_' + source + '.json')
    if target.exists() and not replace:
        print(source + ': existing selection preserved', flush=True); return
    if target.exists():
        shutil_target = target.with_suffix('.previous.json')
        shutil_target.write_bytes(target.read_bytes())
    inventory = json.loads((OUT / 'source_inventory.json').read_text(encoding='utf-8'))
    rows = [value for row in inventory if row['source'] == source and (value := profile(row)) is not None]
    if not rows: raise ValueError('No eligible source videos')
    # This is a user-reported teacher-approved accounting rule, not original-camera ownership.
    records = []; tasks = []
    image_counts = allocate(rows, QUOTAS[source][0], 'images')
    clip_counts = allocate(rows, QUOTAS[source][1], 'clips')
    for row, image_count, clip_count in zip(rows, image_counts, clip_counts):
        original = ROOT / row['path']; raw_sha = file_sha(original)
        common = {'source': source, 'source_url': PAGES[source], 'source_video': row['path'], 'source_sha256': raw_sha,
                  'source_id': row['id'], 'collection_group': row['group'], 'scene': row['scene'], 'declared_content': row['content'],
                  'original_provenance': 'public', 'accounting_origin': 'own',
                  'accounting_basis': 'User reports teacher approval of selecting, clipping and extracting public unboxed video as self-collection; not personally filmed',
                  'visual_note': row['visual_note']}
        for index, stamp in enumerate(np.linspace(row['start'], row['end'] - 2, image_count)):
            identifier = f"{row['id']}_image_{index:03}"
            path = OUT / 'images/positive' / (identifier + '.png'); path.parent.mkdir(parents=True, exist_ok=True)
            image = frame_at(original, float(stamp)); Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(path)
            records.append({**common, 'id': identifier, 'kind': 'image', 'path': path.relative_to(ROOT).as_posix(),
                            'source_seconds': float(stamp), 'frame_number': round(stamp * row['fps']), 'width': image.shape[1], 'height': image.shape[0],
                            'sha256': file_sha(path), 'annotation_status': 'unreviewed'})
        for index, stamp in enumerate(np.linspace(row['start'], row['end'] - 3, clip_count)):
            identifier = f"{row['id']}_clip_{index:03}"
            path = OUT / 'videos' / (identifier + '.mp4'); path.parent.mkdir(parents=True, exist_ok=True)
            record = {**common, 'id': identifier, 'kind': 'video', 'path': path.relative_to(ROOT).as_posix(),
                      'source_start_seconds': float(stamp), 'source_end_seconds': float(stamp + 3), 'duration': 3.0}
            tasks.append((original, path, float(stamp), record))
    def encode(task):
        original, path, stamp, record = task
        command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-ss', str(stamp), '-i', str(original), '-t', '3',
                   '-vf', "scale='min(1280,iw)':-2:in_range=auto:out_range=tv,fps=15,format=yuv420p", '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20',
                   '-threads', '1', '-pix_fmt', 'yuv420p', '-color_range', 'tv', '-movflags', '+faststart', str(path)]
        result = subprocess.run(command, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
        if result.returncode: raise OSError(result.stderr.decode(errors='replace')[-500:])
        record['sha256'] = file_sha(path)
        return record
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records.extend(pool.map(encode, tasks))
    save(target, records)
    print(json.dumps({'source': source, 'images': image_counts and sum(image_counts), 'clips': sum(clip_counts), 'source_videos': len(rows)}), flush=True)


def prelabel(source):
    from grounding_prelabel import GroundingPrelabeler, PrelabelConfig
    from annotations import yolo_text
    torch.set_num_threads(8)
    rows = json.loads((OUT / ('selected_' + source + '.json')).read_text(encoding='utf-8'))
    model = GroundingPrelabeler('cuda:0' if torch.cuda.is_available() else 'cpu'); config = PrelabelConfig()
    directory = OUT / 'preannotations'; directory.mkdir(exist_ok=True)
    save(directory / 'grounding_dino_metadata.json', model.run_metadata(config))
    images = [row for row in rows if row['kind'] == 'image']
    for index, row in enumerate(images):
        path = directory / (row['id'] + '.json')
        if path.exists() and json.loads(path.read_text(encoding='utf-8')).get('input_sha256') == row['sha256']: continue
        with Image.open(ROOT / row['path']) as image:
            prediction = model.predict(image.convert('RGB'), config)
        save(path, {**prediction, 'input_sha256': row['sha256'], 'status': 'candidate' if prediction['boxes'] else 'unresolved_positive', 'human_reviewed': False})
        label = OUT / 'labels_draft' / (row['id'] + '.txt'); label.parent.mkdir(exist_ok=True)
        label.write_text(yolo_text(prediction['boxes'], row['width'], row['height']), encoding='utf-8')
        if index % 20 == 0: print(f'{source}: preannotated {index + 1}/{len(images)}', flush=True)
    print(f'{source}: candidate labels complete; human review still required', flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['collect', 'prelabel']); parser.add_argument('source', choices=list(QUOTAS)); parser.add_argument('--replace-selection', action='store_true'); args = parser.parse_args()
    collect(args.source, args.replace_selection) if args.stage == 'collect' else prelabel(args.source)


if __name__ == '__main__': main()
