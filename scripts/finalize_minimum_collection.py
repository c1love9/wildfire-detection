"""Select D-Fire negatives, register provenance, freeze groups and export draft YOLO."""
import argparse
import collections
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import zipfile
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from dataset_store import DatasetStore, csv_bytes, file_sha, now
from annotations import AnnotationStore, yolo_text
from build_minimum_collection import OUT, PAGES, frame_at, save


def phash(image):
    pixels = np.asarray(image.convert('L').resize((32, 32)), np.float32)
    values = cv2.dct(pixels)[:8, :8].ravel()[1:]
    return sum(int(value > np.median(values)) << index for index, value in enumerate(values))


def negatives():
    target = OUT / 'selected_dfire.json'
    exclusions = json.loads((ROOT / 'configs/collection_negative_exclusions.json').read_text(encoding='utf-8'))
    if target.exists():
        previous = json.loads(target.read_text(encoding='utf-8'))
        if not any(row['id'].removeprefix('dfire_') in exclusions for row in previous):
            print('D-Fire selection preserved'); return
        target.with_suffix('.previous.json').write_bytes(target.read_bytes())
    archive = ROOT / 'data/public_sources/dfire/dfire_kaggle.zip'
    rows = []; hashes = []; pixel_hashes = set()
    with zipfile.ZipFile(archive) as package:
        names = set(package.namelist())
        labels = [name for name in names if '/labels/' in name and name.endswith('.txt') and package.getinfo(name).file_size == 0]
        # Prefer non-PublicDataset sources, whose upstream provider terms are less ambiguous.
        labels.sort(key=lambda name: (Path(name).stem.startswith('PublicDataset'), hashlib.sha256(name.encode()).hexdigest()))
        for label in labels:
            if Path(label).stem in exclusions: continue
            stem = label.rsplit('.', 1)[0].replace('/labels/', '/images/')
            image_name = next((stem + suffix for suffix in ('.jpg', '.jpeg', '.png') if stem + suffix in names), None)
            if image_name is None: continue
            with Image.open(BytesIO(package.read(image_name))) as original:
                image = ImageOps.exif_transpose(original).convert('RGB')
            if min(image.size) < 160 or image.entropy() < 4.5: continue
            fingerprint = hashlib.sha256(image.tobytes()).hexdigest(); perceptual = phash(image)
            if fingerprint in pixel_hashes or any((perceptual ^ previous).bit_count() < 10 for previous in hashes): continue
            identifier = 'dfire_' + Path(image_name).stem
            path = OUT / 'images/negative' / (identifier + '.png'); path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
            rows.append({'id': identifier, 'kind': 'image', 'source': 'dfire', 'source_url': PAGES['dfire'],
                'archive_member': image_name, 'original_label_member': label, 'original_provenance': 'public', 'accounting_origin': 'public',
                'collection_group': 'dfire_aof_surveillance_unknown_event' if Path(image_name).stem.startswith('AoF') else 'dfire_negative_' + fingerprint[:16], 'declared_content': ['normal'],
                'scene': {'setting': '未说明', 'lighting': '未说明', 'scale': '正常无火', 'distance': '未说明', 'view': '未说明'},
                'path': path.relative_to(ROOT).as_posix(), 'width': image.width, 'height': image.height, 'sha256': file_sha(path),
                'annotation_status': 'provided_empty_label_pending_student_review', 'perceptual_hash': hex(perceptual)})
            hashes.append(perceptual); pixel_hashes.add(fingerprint)
            if len(rows) == 200: break
    if len(rows) != 200: raise ValueError('Insufficient distinct valid negative images')
    save(target, rows); print('Selected 200 diverse D-Fire empty-label negatives', flush=True)


def records():
    return [row for source in ('kmu', 'firesense', 'flame2', 'dfire')
            for row in json.loads((OUT / ('selected_' + source + '.json')).read_text(encoding='utf-8'))]


def sheets():
    items = [row for row in records() if row['kind'] == 'image']
    directory = OUT / 'review/final'; directory.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(items), 50):
        subset = items[start:start + 50]
        sheet = Image.new('RGB', (1500, math.ceil(len(subset) / 10) * 150), '#eeeeee'); draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(subset):
            with Image.open(ROOT / row['path']) as raw: image = raw.convert('RGB'); image.thumbnail((148, 116))
            x, y = index % 10 * 150, index // 10 * 150
            sheet.paste(image, (x, y + 15)); draw.text((x + 2, y), str(start + index) + ' ' + row['source'], fill='black')
        sheet.save(directory / f'images_{start // 50 + 1:02}.jpg', quality=94)
    save(directory / 'image_index.json', [{'index': i, 'id': r['id'], 'path': r['path']} for i, r in enumerate(items)])
    clips = [row for row in records() if row['kind'] == 'video']
    for start in range(0, len(clips), 50):
        subset = clips[start:start + 50]
        sheet = Image.new('RGB', (1500, math.ceil(len(subset) / 10) * 150), '#eeeeee')
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(subset):
            frame = frame_at(ROOT / row['path'], row['duration'] / 2)
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image.thumbnail((148, 116))
            x, y = index % 10 * 150, index // 10 * 150
            sheet.paste(image, (x, y + 15))
            draw.text((x + 2, y), str(start + index) + ' ' + row['source'], fill='black')
        sheet.save(directory / f'clips_{start // 50 + 1:02}.jpg', quality=94)
    save(directory / 'clip_index.json', [{'index': i, 'id': r['id'], 'path': r['path']} for i, r in enumerate(clips)])


class CachedProposals:
    def __init__(self, cache, metadata): self.cache, self.metadata = cache, metadata
    def run_metadata(self, config): return self.metadata
    def predict(self, image, config):
        key = hashlib.sha256(f'image:{image.width}:{image.height}:'.encode() + image.tobytes()).hexdigest()
        return self.cache[key]


def register():
    result_path = OUT / 'registration.json'
    if result_path.exists(): print('Registration already exists; refusing duplicate batches'); return
    rows = records(); assert sum(r['kind'] == 'image' for r in rows) == 700 and sum(r['kind'] == 'video' for r in rows) == 300
    identities = set()
    for row in rows:
        if row['kind'] == 'image':
            with Image.open(ROOT / row['path']) as image:
                rgb = image.convert('RGB')
                identity = hashlib.sha256(f'image:{rgb.width}:{rgb.height}:'.encode() + rgb.tobytes()).hexdigest()
        else:
            identity = row['sha256']
        if identity in identities: raise ValueError('Duplicate selected media; replace first: ' + row['id'])
        identities.add(identity)
    dataset = DatasetStore(ROOT / 'data/fire_dataset'); annotations = AnnotationStore(dataset)
    if dataset.assets(): raise ValueError('Expected an empty official workspace; refusing to merge silently')
    batches = {}; imported = []
    for index, row in enumerate(rows):
        # Group by scene as well: all clips/images still share their original collection group.
        signature = (row['collection_group'], row['accounting_origin'], tuple(row['declared_content']), json.dumps(row['scene'], sort_keys=True))
        if signature not in batches:
            explanation = ('原始公开素材自主选片/剪辑/抽帧；用户已告知教师同意计入自采，非本人实拍。' if row['accounting_origin'] == 'own' else '公开D-Fire扩充，原始空标签。')
            batches[signature] = dataset.create_batch(row['source'] + ' ' + row['id'], row['collection_group'], row['accounting_origin'],
                explanation + ' ' + row['source_url'], row['declared_content'], row['scene'])
        batch = batches[signature]
        if row['kind'] == 'image':
            result = dataset.import_image(ROOT / row['path'], batch, source_name=row['id'] + '.png',
                frame_number=row.get('frame_number'), media_seconds=row.get('source_seconds'), metadata=row)
        else:
            result = dataset.import_video(ROOT / row['path'], batch, source_name=row['id'] + '.mp4', metadata=row)
        if not result['added']: raise ValueError('Duplicate selected pixels/bytes; replace before finalizing: ' + row['id'])
        imported.append({**row, 'asset_id': result['asset_id'], 'batch_id': batch})
        if index % 100 == 0: print(f'Registered {index + 1}/{len(rows)}', flush=True)
    for source in ('kmu', 'firesense', 'flame2', 'dfire'):
        assets = [row for row in imported if row['source'] == source and row['kind'] == 'image']; cache = {}
        for row in assets:
            proposal = {'boxes': [], 'elapsed_seconds': 0.} if source == 'dfire' else json.loads((OUT / 'preannotations' / (row['id'] + '.json')).read_text(encoding='utf-8'))
            cache[row['asset_id']] = proposal
        metadata = {'original_source': PAGES[source], 'method': 'provided_DFire_empty_label' if source == 'dfire' else 'GroundingDINO cached exact-image inference',
                    'human_reviewed': False, 'runtime': json.loads((OUT / 'preannotations/grounding_dino_metadata.json').read_text(encoding='utf-8')) if source != 'dfire' else None}
        model = CachedProposals(cache, metadata)
        for start in range(0, len(assets), 100):
            result = annotations.prelabel([r['asset_id'] for r in assets[start:start + 100]], model, None, '自动整理 原始候选待本人复核')
            if any(row['status'] != 'ready' for row in result): raise ValueError('Candidate registration failed')
    plan = dataset.plan_split(); split = dataset.save_split(plan)
    assigned = {row['id']: row['split'] for row in plan['assets']}
    for row in imported: row['split'] = assigned[row['asset_id']]
    save(result_path, {'created_at': now(), 'split_id': split, 'records': imported})
    print(json.dumps({'images': plan['image_counts'], 'split': split, 'statistics': dataset.statistics()}, ensure_ascii=False), flush=True)


def export():
    registration = json.loads((OUT / 'registration.json').read_text(encoding='utf-8')); rows = registration['records']
    target = OUT / 'yolo_draft'; target.mkdir(exist_ok=True)
    counts = collections.Counter(); class_images = collections.Counter(); unresolved = []
    for row in rows:
        counts[(row['kind'], row['split'])] += 1
        if row['kind'] != 'image': continue
        image = target / 'images' / row['split'] / (row['id'] + '.png'); image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / row['path'], image)
        proposal = {'boxes': []} if row['source'] == 'dfire' else json.loads((OUT / 'preannotations' / (row['id'] + '.json')).read_text(encoding='utf-8'))
        if not proposal['boxes'] and row['source'] != 'dfire': unresolved.append(row['id'])
        label = target / 'labels' / row['split'] / (row['id'] + '.txt'); label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text(yolo_text(proposal['boxes'], row['width'], row['height']), encoding='utf-8')
        for name in set(box['class'] for box in proposal['boxes']): class_images[name] += 1
    (target / 'data.yaml').write_text('train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: fire\n  1: smoke\n', encoding='utf-8')
    (target / 'DO_NOT_TRAIN_BEFORE_REVIEW.txt').write_text('所有标签为候选，尚未人工复核。unresolved_positive为模型未检出的正样本，不得作为负样本训练。请在项目Annotations复核后重新生成正式YOLO训练目录。\n', encoding='utf-8')
    save(target / 'unresolved_positive.json', unresolved)
    # Keep a convenient CSV while retaining nested metadata in the complete JSON.
    flat = [{**{key: row.get(key) for key in ('id', 'kind', 'source', 'accounting_origin', 'original_provenance', 'collection_group', 'split', 'path', 'source_video', 'source_seconds', 'source_start_seconds', 'source_end_seconds', 'duration', 'asset_id')},
             **row['scene'], 'declared_content': ','.join(row['declared_content'])} for row in rows]
    (OUT / 'dataset_manifest.csv').write_bytes(csv_bytes(flat))
    summary = {'status': 'collected_with_draft_labels_pending_human_review', 'created_at': now(), 'images': 700, 'fire_scene_images': 500, 'normal_images': 200,
               'video_clips': 300, 'video_seconds': sum(r.get('duration', 0) for r in rows if r['kind'] == 'video'),
               'accounting_self_images': 500, 'accounting_self_videos': 300, 'accounting_self_fraction_all_assets': .8,
               'self_definition': 'User-reported teacher approval of public-source curation; original public provenance retained; not personally filmed',
               'image_splits': {name: counts['image', name] for name in ('train', 'val', 'test')},
               'video_splits': {name: counts['video', name] for name in ('train', 'val', 'test')},
               'source_counts': {source: {kind: sum(r['source'] == source and r['kind'] == kind for r in rows) for kind in ('image', 'video')} for source in PAGES},
               'declared_class_images_not_frame_ground_truth': {name: sum(r['kind'] == 'image' and name in r['declared_content'] for r in rows) for name in ('fire', 'smoke', 'normal')},
               'preannotation_class_images_not_ground_truth': dict(class_images), 'unresolved_positive_images': len(unresolved),
               'scene_distribution_images': {key: dict(collections.Counter(r['scene'][key] for r in rows if r['kind'] == 'image')) for key in ('setting', 'lighting', 'scale', 'distance', 'view')},
               'source_videos': len({r['source_video'] for r in rows if 'source_video' in r}),
               'human_approved_images': 0, 'formal_accuracy_evaluation': 'not run', 'split_id': registration['split_id'],
               'coverage_limits': ['Early sparks are not claimed solely from a small apparent flame; inspect dedicated evidence before marking fully covered',
                                   'Smoke appearance is not proof of a fully developed combustion stage', 'Only one FLAME2 burn event; all its derivatives stay in one split',
                                   'D-Fire original event identity is not always available; visual deduplication does not prove universal independence']}
    save(OUT / 'dataset_statistics.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['negatives', 'sheets', 'register', 'export']); arguments = parser.parse_args()
    {'negatives': negatives, 'sheets': sheets, 'register': register, 'export': export}[arguments.stage]()
