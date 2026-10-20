"""Local, provenance-aware media catalog and immutable grouped split manifests."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from io import BytesIO, StringIO
from pathlib import Path

from PIL import Image, ImageOps


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VIDEO_SUFFIXES = {'.mp4', '.mov', '.avi', '.mkv'}
ORIGINS = {'own': '自采', 'public': '公开', 'demo': '演示/测试'}
SCENES = {'setting': ('室内', '室外', '未说明'), 'lighting': ('白天', '夜间', '逆光', '未说明'),
          'scale': ('早期小火星', '发展期明火', '充分燃烧浓烟', '正常无火', '未说明'),
          'distance': ('近景', '中景', '远景', '未说明'), 'view': ('平视', '俯视', '仰视', '未说明')}
SCENE_LABELS = {'setting': '室内外', 'lighting': '光照', 'scale': '火灾规模', 'distance': '拍摄距离', 'view': '拍摄视角'}


def now():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def dataset_root(project_root):
    return Path(os.environ.get('FIRE_DATASET_ROOT', Path(project_root) / 'data' / 'fire_dataset')).resolve()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def csv_bytes(rows):
    output = StringIO(newline='')
    if rows:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "'" + value if isinstance(value, str) and value.startswith(('=', '+', '-', '@')) else value
                             for key, value in row.items()})
    return output.getvalue().encode('utf-8-sig')


class DatasetStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'catalog.sqlite3'
        with self.connection() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS batches (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, group_key TEXT NOT NULL,
                    origin TEXT NOT NULL, source_ref TEXT NOT NULL, content_json TEXT NOT NULL,
                    scene_json TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS assets (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, path TEXT NOT NULL UNIQUE,
                    sha256 TEXT NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL,
                    duration REAL, frames INTEGER, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS occurrences (
                    id TEXT PRIMARY KEY, batch_id TEXT NOT NULL REFERENCES batches(id),
                    asset_id TEXT NOT NULL REFERENCES assets(id), source_name TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL, parent_id TEXT REFERENCES assets(id),
                    frame_number INTEGER, media_seconds REAL, metadata_json TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS occurrences_asset ON occurrences(asset_id);
                CREATE TABLE IF NOT EXISTS splits (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, fingerprint TEXT NOT NULL,
                    manifest_path TEXT NOT NULL, summary_json TEXT NOT NULL
                );
            ''')
            if 'manifest_sha256' not in {row['name'] for row in db.execute('PRAGMA table_info(splits)')}:
                db.execute('BEGIN IMMEDIATE')
                if 'manifest_sha256' not in {row['name'] for row in db.execute('PRAGMA table_info(splits)')}:
                    db.execute('ALTER TABLE splits ADD COLUMN manifest_sha256 TEXT')

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            with db:
                yield db
        finally:
            db.close()

    def create_batch(self, name, group_key, origin, source_ref, content, scene=None):
        for value in (name, group_key, source_ref):
            if not isinstance(value, str) or not value.strip() or len(value) > 500:
                raise ValueError('批次名称、采集组和来源说明须填写，且不超过500字。')
        content = sorted(set(content))
        if origin not in ORIGINS or not content or not set(content) <= {'fire', 'smoke', 'normal', 'unknown'}:
            raise ValueError('素材来源或场景声明无效。')
        if 'normal' in content and len(content) != 1 or 'unknown' in content and len(content) != 1:
            raise ValueError('正常/未核实场景不能同时声明火焰或烟雾。')
        scene = {key: (scene or {}).get(key, '未说明') for key in SCENES}
        if any(scene[key] not in options for key, options in SCENES.items()):
            raise ValueError('场景标签无效。')
        batch_id = uuid.uuid4().hex
        with self.connection() as db:
            db.execute('INSERT INTO batches VALUES(?,?,?,?,?,?,?,?)',
                       (batch_id, name.strip(), group_key.strip(), origin, source_ref.strip(), json.dumps(content), json.dumps(scene), now()))
        return batch_id

    def batches(self):
        with self.connection() as db:
            return [dict(row) for row in db.execute('SELECT * FROM batches ORDER BY created_at DESC,id')]

    def batch(self, batch_id):
        with self.connection() as db:
            row = db.execute('SELECT * FROM batches WHERE id=?', (batch_id,)).fetchone()
        if row is None:
            raise ValueError('找不到该采集批次。')
        return dict(row)

    def _register(self, batch_id, identity, kind, content, suffix, width, height, source_name, source_sha,
                  duration=None, frames=None, parent_id=None, frame_number=None, media_seconds=None, metadata=None):
        relative = f'media/{kind}s/{identity}{suffix}'
        target = self.root / relative
        sha = hashlib.sha256(content).hexdigest() if isinstance(content, bytes) else file_sha(content)
        occurrence_key = [batch_id, identity, source_sha, parent_id, frame_number]
        if metadata and metadata.get('capture_id'):
            occurrence_key.append(metadata['capture_id'])
        occurrence = hashlib.sha256(json.dumps(occurrence_key, ensure_ascii=False).encode()).hexdigest()
        staging = self.root / (uuid.uuid4().hex + '.pending')
        try:
            with self.connection() as db:
                db.execute('BEGIN IMMEDIATE')
                if db.execute('SELECT 1 FROM batches WHERE id=?', (batch_id,)).fetchone() is None:
                    raise ValueError('批次不存在。')
                if parent_id:
                    parent = db.execute('SELECT 1 FROM occurrences WHERE asset_id=? AND batch_id=?', (parent_id, batch_id)).fetchone()
                    if parent is None:
                        raise ValueError('抽帧必须沿用原视频的采集批次。')
                added = db.execute('SELECT 1 FROM assets WHERE id=?', (identity,)).fetchone() is None
                if added:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if isinstance(content, bytes):
                        staging.write_bytes(content)
                    else:
                        with Path(content).open('rb') as source, staging.open('wb') as destination:
                            for block in iter(lambda: source.read(1024 * 1024), b''):
                                destination.write(block)
                    if file_sha(staging) != sha:
                        raise OSError('导入时源文件发生变化，请重新导入。')
                    staging.replace(target)
                    db.execute('INSERT INTO assets VALUES(?,?,?,?,?,?,?,?,?)',
                               (identity, kind, relative, sha, width, height, duration, frames, now()))
                db.execute('INSERT OR IGNORE INTO occurrences VALUES(?,?,?,?,?,?,?,?,?,?)',
                           (occurrence, batch_id, identity, Path(source_name).name, source_sha, parent_id, frame_number, media_seconds,
                            json.dumps(metadata or {}, sort_keys=True), now()))
            return {'asset_id': identity, 'added': added}
        finally:
            staging.unlink(missing_ok=True)

    def import_image(self, source, batch_id, source_name=None, parent_id=None, frame_number=None, media_seconds=None, metadata=None):
        data = source if isinstance(source, bytes) else Path(source).read_bytes()
        source_sha = hashlib.sha256(data).hexdigest()
        try:
            with Image.open(BytesIO(data)) as raw:
                if raw.width * raw.height > 24_000_000:
                    raise ValueError('图片超过2400万像素，请先缩小。')
                image = ImageOps.exif_transpose(raw).convert('RGB')
                image.load()
        except (OSError, Image.DecompressionBombError) as exc:
            raise ValueError('图片内容无法解码。') from exc
        identity = hashlib.sha256(f'image:{image.width}:{image.height}:'.encode() + image.tobytes()).hexdigest()
        normalized = BytesIO()
        image.save(normalized, format='PNG')
        return self._register(batch_id, identity, 'image', normalized.getvalue(), '.png', image.width, image.height,
                              source_name or ('image.png' if isinstance(source, bytes) else str(source)), source_sha,
                              parent_id=parent_id, frame_number=frame_number, media_seconds=media_seconds, metadata=metadata)

    def import_video(self, source, batch_id, source_name=None, metadata=None):
        from data_collection import probe_video
        source = Path(source)
        info = probe_video(source)
        sha = file_sha(source)
        return self._register(batch_id, sha, 'video', source, source.suffix.lower(), info['width'], info['height'],
                              source_name or source.name, sha, duration=info['duration'], frames=info['frames'], metadata=metadata)

    def assets(self):
        with self.connection() as db:
            return [dict(row) for row in db.execute('SELECT * FROM assets ORDER BY id')]

    def snapshot(self):
        with self.connection() as db:
            # One read transaction prevents partial snapshots during a concurrent import.
            db.execute('BEGIN')
            return {table: [dict(row) for row in db.execute(f'SELECT * FROM {table} ORDER BY id')]
                    for table in ('assets', 'batches', 'occurrences')}

    def asset_path(self, asset):
        path = (self.root / asset['path']).resolve()
        if not path.is_relative_to(self.root / 'media') or not path.is_file():
            raise OSError('素材不在数据目录中或文件已丢失。')
        return path

    @staticmethod
    def fingerprint(snapshot):
        return hashlib.sha256(json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    @staticmethod
    def _facts(snapshot):
        batches = {row['id']: row for row in snapshot['batches']}
        links = {}
        for row in snapshot['occurrences']:
            links.setdefault(row['asset_id'], []).append(row)
        facts = []
        for asset in snapshot['assets']:
            occurrences = links.get(asset['id'], [])
            related = [batches[row['batch_id']] for row in occurrences]
            origins = {row['origin'] for row in related}
            content = set().union(*(set(json.loads(row['content_json'])) for row in related))
            conflict = ('normal' in content and bool(content & {'fire', 'smoke'}))
            origin = next(iter(origins)) if len(origins) == 1 else 'conflict'
            facts.append({**asset, 'origin': origin, 'content': sorted(content), 'content_conflict': conflict,
                          'eligible': bool(origins) and origins <= {'own', 'public'},
                          'batch_ids': sorted({row['batch_id'] for row in occurrences}),
                          'parents': sorted({row['parent_id'] for row in occurrences if row['parent_id']}),
                          'groups': sorted({row['group_key'] for row in related}),
                          'derived': any(row['parent_id'] for row in occurrences)})
        return facts

    def statistics(self):
        snapshot = self.snapshot()
        facts = self._facts(snapshot)
        eligible = [row for row in facts if row['eligible']]
        own = [row for row in eligible if row['origin'] == 'own']
        fire_scene = lambda row: bool(set(row['content']) & {'fire', 'smoke'}) and not row['content_conflict']
        own_images = sum(row['kind'] == 'image' and fire_scene(row) for row in own)
        own_videos = [row for row in own if row['kind'] == 'video' and fire_scene(row)]
        normal_images = sum(row['kind'] == 'image' and row['content'] == ['normal'] for row in eligible)
        distribution = []
        scenes = {row['id']: json.loads(row['scene_json']) for row in snapshot['batches']}
        for key in SCENES:
            counts = {}
            for asset in eligible:
                values = {scenes[batch_id][key] for batch_id in asset['batch_ids']}
                value = next(iter(values)) if len(values) == 1 else '多种/待复核'
                counts[value] = counts.get(value, 0) + 1
            distribution.extend({'维度': SCENE_LABELS[key], '场景': value, '唯一素材数': count} for value, count in sorted(counts.items()))
        return dict(total_images=sum(row['kind'] == 'image' for row in facts), total_videos=sum(row['kind'] == 'video' for row in facts),
                    duplicate_occurrences=len(snapshot['occurrences']) - len(facts),
                    derived_images=sum(row['kind'] == 'image' and row['derived'] for row in facts),
                    eligible_assets=len(eligible), own_assets=len(own), own_ratio=len(own) / len(eligible) if eligible else 0,
                    own_fire_images=own_images, own_fire_videos=len(own_videos),
                    own_fire_video_seconds=sum(row['duration'] for row in own_videos), normal_images=normal_images,
                    excluded_demo_or_test=sum(not row['eligible'] for row in facts),
                    provenance_conflicts=sum(row['origin'] == 'conflict' for row in facts),
                    scene_conflicts=sum(row['content_conflict'] for row in facts), scene_distribution=distribution,
                    declared_classes={label: sum(row['kind'] == 'image' and label in row['content'] and not row['content_conflict'] for row in eligible)
                                      for label in ('fire', 'smoke', 'normal', 'unknown')},
                    fingerprint=self.fingerprint(snapshot))

    def manifest_rows(self):
        snapshot = self.snapshot()
        batches = {row['id']: row for row in snapshot['batches']}
        facts = {row['id']: row for row in self._facts(snapshot)}
        return [{'asset_id': row['asset_id'], 'kind': facts[row['asset_id']]['kind'],
                 'path': facts[row['asset_id']]['path'], 'sha256': facts[row['asset_id']]['sha256'],
                 'batch': batches[row['batch_id']]['name'], 'group': batches[row['batch_id']]['group_key'],
                 'origin': batches[row['batch_id']]['origin'], 'source_ref': batches[row['batch_id']]['source_ref'],
                 'source_name': row['source_name'], 'source_sha256': row['source_sha256'],
                 'declared_content': batches[row['batch_id']]['content_json'], 'scene': batches[row['batch_id']]['scene_json'],
                 'parent_id': row['parent_id'], 'frame_number': row['frame_number'], 'media_seconds': row['media_seconds'],
                 'duration_seconds': facts[row['asset_id']]['duration'], 'capture_metadata': row['metadata_json']} for row in snapshot['occurrences']]

    def plan_split(self, ratios=(0.7, 0.2, 0.1), seed=42):
        if len(ratios) != 3 or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in ratios) or not math.isclose(sum(ratios), 1):
            raise ValueError('训练/验证/测试比例须为正数且合计为100%。')
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError('随机种子须为0～4294967295的整数。')
        snapshot = self.snapshot()
        # Include ALL origins in the graph: a demo/public duplicate can still
        # connect two real groups, even though it is excluded from output lists.
        facts = self._facts(snapshot)
        parent = {row['id']: row['id'] for row in facts}
        def find(key):
            while parent[key] != key:
                parent[key] = parent[parent[key]]
                key = parent[key]
            return key
        def union(left, right):
            left, right = find(left), find(right)
            parent[max(left, right)] = min(left, right)
        groups = {}
        for row in facts:
            for group in row['groups']:
                if group in groups:
                    union(row['id'], groups[group])
                groups[group] = row['id']
            for original in row['parents']:
                union(row['id'], original)
        components = {}
        for row in facts:
            if row['eligible']:
                components.setdefault(find(row['id']), []).append(row)
        weighted = [(key, members, sum(row['kind'] == 'image' for row in members)) for key, members in sorted(components.items())]
        image_groups = [item for item in weighted if item[2] > 0]
        if len(image_groups) < 3:
            raise ValueError('至少需要3个互不关联、含图像的正式采集组；同组或重复素材不能拆开凑比例。演示素材不参加正式划分。')
        rng = random.Random(seed)
        rng.shuffle(image_groups)
        image_groups.sort(key=lambda item: item[2], reverse=True)
        counts = [0, 0, 0]
        target = [sum(item[2] for item in image_groups) * value for value in ratios]
        initial_order = sorted(range(3), key=lambda index: ratios[index], reverse=True)
        assignments = {}
        for index, (key, members, weight) in enumerate(image_groups):
            split = initial_order[index] if index < 3 else min(range(3), key=lambda choice: sum(((counts[j] + (weight if j == choice else 0) - target[j]) / max(target[j], 1))**2 for j in range(3)))
            counts[split] += weight
            for row in members:
                assignments[row['id']] = ('train', 'val', 'test')[split]
        for key, members, weight in weighted:
            if not weight:
                split = rng.choices(('train', 'val', 'test'), weights=ratios, k=1)[0]
                assignments.update({row['id']: split for row in members})
        return {'schema_version': 1, 'created_at': now(), 'fingerprint': self.fingerprint(snapshot), 'seed': seed,
                'requested_ratios': list(ratios), 'path_base': 'dataset_root', 'labels_status': 'not_annotated',
                'grouping': 'connected components of collection group, parent video, and duplicate image pixels/video bytes',
                'image_counts': dict(zip(('train', 'val', 'test'), counts)), 'group_count': len(components),
                'assets': [{**row, 'split': assignments[row['id']], 'component': find(row['id'])} for row in facts if row['id'] in assignments],
                'provenance': snapshot}

    def save_split(self, plan):
        # A caller cannot save a stale or modified assignment and silently leak data.
        expected = self.plan_split(tuple(plan['requested_ratios']), plan['seed'])
        for key in ('fingerprint', 'assets', 'image_counts'):
            if plan[key] != expected[key]:
                raise ValueError('数据或划分预览已变化，请重新生成预览。')
        for asset in plan['assets']:
            if file_sha(self.asset_path(asset)) != asset['sha256']:
                raise OSError('素材内容已改变，未保存划分；请核对数据完整性。')
        version = 'split_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:8]
        directory = self.root / 'splits' / version
        directory.mkdir(parents=True, exist_ok=False)
        manifest = {**expected, 'version': version}
        files = {'manifest.json': json.dumps(manifest, ensure_ascii=False, indent=2) + '\n'}
        for split in ('train', 'val', 'test'):
            files[split + '.txt'] = ''.join(row['path'] + '\n' for row in expected['assets'] if row['kind'] == 'image' and row['split'] == split)
        for name, content in files.items():
            (directory / name).write_text(content, encoding='utf-8')
        with self.connection() as db:
            db.execute('INSERT INTO splits(id,created_at,fingerprint,manifest_path,summary_json,manifest_sha256) VALUES(?,?,?,?,?,?)',
                       (version, now(), expected['fingerprint'], f'splits/{version}/manifest.json',
                        json.dumps(expected['image_counts']), file_sha(directory / 'manifest.json')))
        return version

    def splits(self):
        fingerprint = self.fingerprint(self.snapshot())
        with self.connection() as db:
            return [{**dict(row), 'current': row['fingerprint'] == fingerprint} for row in db.execute('SELECT * FROM splits ORDER BY created_at DESC')]
