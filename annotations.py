"""Immutable model proposals, append-only human reviews and reviewed YOLO exports."""

from __future__ import annotations

import json
import math
import re
import shutil
import time
import uuid
import zipfile
from pathlib import Path

from PIL import Image

from dataset_store import DatasetStore, file_sha, now


CLASSES = ('fire', 'smoke')


def validate_boxes(boxes, width, height):
    if not isinstance(boxes, list) or len(boxes) > 300:
        raise ValueError('单幅图像须为不超过300个框的列表。')
    result, seen = [], set()
    for box in boxes:
        if not isinstance(box, dict) or box.get('class') not in CLASSES:
            raise ValueError('边界框类别只能为fire或smoke。')
        identifier = box.get('id') or uuid.uuid4().hex
        if not isinstance(identifier, str) or not re.fullmatch('[0-9a-f]{32}', identifier) or identifier in seen:
            raise ValueError('边界框编号无效或重复。')
        values = box.get('xyxy')
        if not isinstance(values, (list, tuple)) or len(values) != 4 or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
            raise ValueError('边界框坐标须为4个有限数字。')
        x1, y1, x2, y2 = map(float, values)
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height and x2 - x1 >= 1 and y2 - y1 >= 1):
            raise ValueError('边界框须在原图范围内，且宽高至少1像素。')
        seen.add(identifier)
        result.append({'id': identifier, 'class': box['class'], 'xyxy': [round(value, 4) for value in (x1, y1, x2, y2)]})
    return result


def box_diff(before, after):
    left, right = ({box['id']: box for box in group} for group in (before, after))
    return {'added': [right[key] for key in sorted(right.keys() - left.keys())],
            'removed': [left[key] for key in sorted(left.keys() - right.keys())],
            'changed': [{'id': key, 'before': left[key], 'after': right[key]} for key in sorted(left.keys() & right.keys()) if left[key] != right[key]]}


def yolo_text(boxes, width, height):
    rows = []
    for box in validate_boxes(boxes, width, height):
        x1, y1, x2, y2 = box['xyxy']
        rows.append(f"{CLASSES.index(box['class'])} {(x1 + x2) / (2 * width):.8f} {(y1 + y2) / (2 * height):.8f} {(x2 - x1) / width:.8f} {(y2 - y1) / height:.8f}\n")
    return ''.join(rows)


def _decode(row):
    if row is None:
        return None
    row = dict(row)
    for key in list(row):
        if key.endswith('_json'):
            row[key[:-5]] = json.loads(row.pop(key))
    return row


class AnnotationStore:
    def __init__(self, dataset: DatasetStore):
        self.dataset = dataset
        with dataset.connection() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS annotation_runs (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, finished_at TEXT, operator TEXT NOT NULL,
                    metadata_json TEXT NOT NULL, status TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS annotation_proposals (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL,
                    run_id TEXT NOT NULL REFERENCES annotation_runs(id), asset_id TEXT NOT NULL REFERENCES assets(id),
                    input_sha256 TEXT NOT NULL, boxes_json TEXT NOT NULL, status TEXT NOT NULL,
                    error TEXT, elapsed_seconds REAL NOT NULL, created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS proposals_asset ON annotation_proposals(asset_id, seq);
                CREATE TABLE IF NOT EXISTS annotation_reviews (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL,
                    asset_id TEXT NOT NULL REFERENCES assets(id), input_sha256 TEXT NOT NULL,
                    parent_id TEXT REFERENCES annotation_reviews(id), proposal_id TEXT REFERENCES annotation_proposals(id),
                    reviewer TEXT NOT NULL, decision TEXT NOT NULL, negative_confirmed INTEGER NOT NULL,
                    note TEXT NOT NULL, boxes_json TEXT NOT NULL, diff_json TEXT NOT NULL,
                    elapsed_seconds REAL NOT NULL, created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS reviews_asset ON annotation_reviews(asset_id, seq);
                CREATE TABLE IF NOT EXISTS annotation_exports (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, kind TEXT NOT NULL,
                    manifest_json TEXT NOT NULL, path TEXT NOT NULL
                );
            ''')

    def image(self, asset_id, verify=True):
        with self.dataset.connection() as db:
            row = db.execute('SELECT * FROM assets WHERE id=? AND kind=?', (asset_id, 'image')).fetchone()
        if row is None:
            raise ValueError('找不到待标注的图像。视频请先抽帧。')
        asset = dict(row)
        if verify and file_sha(self.dataset.asset_path(asset)) != asset['sha256']:
            raise OSError('归档图像内容已改变，标注操作已停止。')
        return asset

    def latest(self, asset_id):
        with self.dataset.connection() as db:
            return _decode(db.execute('SELECT * FROM annotation_reviews WHERE asset_id=? ORDER BY seq DESC LIMIT 1', (asset_id,)).fetchone())

    def history(self, asset_id):
        with self.dataset.connection() as db:
            return [_decode(row) for row in db.execute('SELECT * FROM annotation_reviews WHERE asset_id=? ORDER BY seq', (asset_id,))]

    def proposals(self, asset_id):
        with self.dataset.connection() as db:
            return [_decode(row) for row in db.execute('''SELECT p.*,r.operator,r.metadata_json FROM annotation_proposals p
                JOIN annotation_runs r ON p.run_id=r.id WHERE p.asset_id=? ORDER BY p.seq DESC''', (asset_id,))]

    def catalog(self):
        with self.dataset.connection() as db:
            return [dict(row) for row in db.execute('''SELECT a.*,r.id AS review_id,r.decision,r.reviewer,
                (SELECT status FROM annotation_proposals p WHERE p.asset_id=a.id ORDER BY p.seq DESC LIMIT 1) AS proposal_status
                FROM assets a LEFT JOIN annotation_reviews r ON r.seq=(SELECT MAX(seq) FROM annotation_reviews rr WHERE rr.asset_id=a.id)
                WHERE a.kind='image' ORDER BY a.created_at,a.id''')]

    def prelabel(self, asset_ids, model, config, operator, progress=None):
        if not isinstance(operator, str) or not operator.strip() or len(operator) > 100:
            raise ValueError('请填写执行者姓名或编号，不超过100字。')
        asset_ids = list(dict.fromkeys(asset_ids))
        if not 1 <= len(asset_ids) <= 100:
            raise ValueError('单次预标注须选择1～100幅图像。')
        assets = [self.image(key, verify=False) for key in asset_ids]
        run_id = uuid.uuid4().hex
        metadata = {**model.run_metadata(config), 'requested_asset_ids': asset_ids}
        with self.dataset.connection() as db:
            db.execute('INSERT INTO annotation_runs VALUES(?,?,?,?,?,?)', (run_id, now(), None, operator.strip(), json.dumps(metadata), 'running'))
        results = []
        for index, asset in enumerate(assets):
            started = time.perf_counter()
            try:
                self.image(asset['id'])
                with Image.open(self.dataset.asset_path(asset)) as image:
                    prediction = model.predict(image.convert('RGB'), config)
                canonical = validate_boxes(prediction['boxes'], asset['width'], asset['height'])
                # Preserve confidences/phrases in immutable model output, while
                # human-reviewed boxes use only class and original pixel bounds.
                boxes = [{**raw, **box} for raw, box in zip(prediction['boxes'], canonical)]
                self.image(asset['id'])
                status, error = 'ready', None
            except Exception as exc:
                boxes, status, error = [], 'failed', str(exc)[:1000]
            identifier = uuid.uuid4().hex
            with self.dataset.connection() as db:
                db.execute('''INSERT INTO annotation_proposals(id,run_id,asset_id,input_sha256,boxes_json,status,error,elapsed_seconds,created_at)
                    VALUES(?,?,?,?,?,?,?,?,?)''', (identifier, run_id, asset['id'], asset['sha256'], json.dumps(boxes), status, error,
                                                 time.perf_counter() - started, now()))
            results.append({'asset_id': asset['id'], 'proposal_id': identifier, 'status': status, 'error': error, 'box_count': len(boxes)})
            if progress:
                progress(index + 1, len(assets))
        failures = sum(row['status'] == 'failed' for row in results)
        with self.dataset.connection() as db:
            db.execute('UPDATE annotation_runs SET status=?,finished_at=? WHERE id=?',
                       ('failed' if failures == len(results) else 'partial' if failures else 'completed', now(), run_id))
        return results

    def save_review(self, asset_id, boxes, reviewer, note, decision='draft', expected_revision=None,
                    proposal_id=None, negative_confirmed=False, elapsed_seconds=0):
        asset = self.image(asset_id)
        boxes = validate_boxes(boxes, asset['width'], asset['height'])
        if not isinstance(reviewer, str) or not reviewer.strip() or len(reviewer) > 100:
            raise ValueError('复核者姓名/编号须填写且不超过100字。')
        if not isinstance(note, str) or not note.strip() or len(note) > 2000:
            raise ValueError('请填写复核/修改说明，不超过2000字。')
        if decision not in {'draft', 'approved'}:
            raise ValueError('复核状态无效。')
        if decision == 'approved' and not boxes and negative_confirmed is not True:
            raise ValueError('无目标图片须人工明确确认，不能将模型空结果直接当作负样本。')
        if isinstance(elapsed_seconds, bool) or not isinstance(elapsed_seconds, (int, float)) or not math.isfinite(elapsed_seconds) or not 0 <= elapsed_seconds <= 86400:
            raise ValueError('复核停留时长无效。')
        with self.dataset.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            previous = _decode(db.execute('SELECT * FROM annotation_reviews WHERE asset_id=? ORDER BY seq DESC LIMIT 1', (asset_id,)).fetchone())
            if (previous['id'] if previous else None) != expected_revision:
                raise ValueError('另一窗口已更新此图的复核，请重新载入后再保存。')
            proposal = None
            if proposal_id:
                proposal = _decode(db.execute('SELECT * FROM annotation_proposals WHERE id=?', (proposal_id,)).fetchone())
                if not proposal or proposal['asset_id'] != asset_id or proposal['status'] != 'ready' or proposal['input_sha256'] != asset['sha256']:
                    raise ValueError('预标注不存在、失败或不属于当前图像。')
            before = previous['boxes'] if previous else validate_boxes(proposal['boxes'], asset['width'], asset['height']) if proposal else []
            identifier = uuid.uuid4().hex
            db.execute('''INSERT INTO annotation_reviews(id,asset_id,input_sha256,parent_id,proposal_id,reviewer,decision,
                negative_confirmed,note,boxes_json,diff_json,elapsed_seconds,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                       (identifier, asset_id, asset['sha256'], expected_revision, proposal_id, reviewer.strip(), decision,
                        int(bool(negative_confirmed and not boxes)), note.strip(), json.dumps(boxes), json.dumps(box_diff(before, boxes)), elapsed_seconds, now()))
        return identifier

    def _approved(self, asset_ids):
        snapshots = []
        # One read transaction freezes the revision selection for this export.
        with self.dataset.connection() as db:
            db.execute('BEGIN')
            for asset_id in asset_ids:
                asset = self.image(asset_id)
                review = _decode(db.execute('SELECT * FROM annotation_reviews WHERE asset_id=? ORDER BY seq DESC LIMIT 1', (asset_id,)).fetchone())
                if not review or review['decision'] != 'approved' or review['input_sha256'] != asset['sha256']:
                    raise ValueError('所选图片中仍有未复核或新草稿，不能导出为已复核标签。')
                if not review['boxes'] and not review['negative_confirmed']:
                    raise ValueError('空标签缺少人工负样本确认。')
                history = [_decode(row) for row in db.execute('SELECT * FROM annotation_reviews WHERE asset_id=? ORDER BY seq', (asset_id,))]
                proposals = [_decode(row) for row in db.execute('''SELECT p.*,r.metadata_json,r.operator FROM annotation_proposals p
                    JOIN annotation_runs r ON p.run_id=r.id WHERE p.asset_id=? ORDER BY p.seq''', (asset_id,))]
                snapshots.append({'asset': asset, 'review': review, 'history': history, 'proposals': proposals})
        return snapshots

    def exports(self):
        with self.dataset.connection() as db:
            return [_decode(row) for row in db.execute('SELECT * FROM annotation_exports ORDER BY created_at DESC,id')]

    def _record_export(self, identifier, kind, manifest, path):
        with self.dataset.connection() as db:
            db.execute('INSERT INTO annotation_exports VALUES(?,?,?,?,?)', (identifier, now(), kind, json.dumps(manifest), str(path.relative_to(self.dataset.root))))

    def export_labels(self, asset_ids):
        asset_ids = list(dict.fromkeys(asset_ids))
        if not asset_ids:
            raise ValueError('请选择至少一幅已复核图像。')
        snapshots = self._approved(asset_ids)
        identifier = 'labels_' + uuid.uuid4().hex
        path = self.dataset.root / 'exports' / (identifier + '.zip')
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {'version': identifier, 'kind': 'reviewed_labels', 'created_at': now(), 'classes': dict(enumerate(CLASSES)),
                    'coordinate_format': 'class x_center y_center width height; normalized to original image',
                    'asset_ids': asset_ids, 'approved_revisions': {row['asset']['id']: row['review']['id'] for row in snapshots}}
        with zipfile.ZipFile(path, 'x', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
            archive.writestr('audit.json', json.dumps(snapshots, ensure_ascii=False, indent=2))
            for row in snapshots:
                asset = row['asset']
                archive.writestr(f"labels/{asset['id']}.txt", yolo_text(row['review']['boxes'], asset['width'], asset['height']))
        self._record_export(identifier, 'labels', {**manifest, 'zip_sha256': file_sha(path)}, path)
        return path

    def export_split(self, version):
        record = next((row for row in self.dataset.splits() if row['id'] == version), None)
        if not record or not record['current']:
            raise ValueError('请选择与当前素材清单一致的划分版本；历史划分请先在数据管理重新生成。')
        if not record['manifest_sha256']:
            raise ValueError('旧版本划分未记录清单校验值，请在数据管理重新保存一个版本。')
        source = (self.dataset.root / record['manifest_path']).resolve()
        if not source.is_relative_to(self.dataset.root / 'splits') or file_sha(source) != record['manifest_sha256']:
            raise OSError('划分清单被改动或不完整，已停止导出。')
        split = json.loads(source.read_text(encoding='utf-8'))
        if split['fingerprint'] != self.dataset.fingerprint(self.dataset.snapshot()):
            raise ValueError('数据在导出前变化，请重新生成划分。')
        images = [row for row in split['assets'] if row['kind'] == 'image']
        snapshots = self._approved([row['id'] for row in images])
        by_id = {row['asset']['id']: row for row in snapshots}
        image_bytes = sum(self.dataset.asset_path(row['asset']).stat().st_size for row in snapshots)
        if shutil.disk_usage(self.dataset.root).free < image_bytes * (2 if image_bytes <= 128 * 1024**2 else 1) + 256 * 1024**2:
            raise OSError('可用磁盘空间不足以保存训练数据及256MB保留空间。')
        identifier = 'yolo_' + uuid.uuid4().hex
        directory = self.dataset.root / 'exports' / identifier
        directory.mkdir(parents=True, exist_ok=False)
        manifest = {'version': identifier, 'kind': 'reviewed_split', 'created_at': now(), 'classes': dict(enumerate(CLASSES)),
                    'source_split': version, 'source_split_sha256': record['manifest_sha256'],
                    'image_counts': split['image_counts'], 'approved_revisions': {key: row['review']['id'] for key, row in by_id.items()}}
        for name in ('train', 'val', 'test'):
            (directory / 'images' / name).mkdir(parents=True)
            (directory / 'labels' / name).mkdir(parents=True)
        for asset in images:
            current = by_id[asset['id']]['asset']
            image_path = self.dataset.asset_path(current)
            if current['sha256'] != asset['sha256'] or file_sha(image_path) != asset['sha256']:
                raise OSError('导出图像与划分快照不一致。')
            shutil.copyfile(image_path, directory / 'images' / asset['split'] / (asset['id'] + '.png'))
            (directory / 'labels' / asset['split'] / (asset['id'] + '.txt')).write_text(
                yolo_text(by_id[asset['id']]['review']['boxes'], asset['width'], asset['height']), encoding='utf-8')
        (directory / 'data.yaml').write_text('train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: fire\n  1: smoke\n', encoding='utf-8')
        (directory / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        (directory / 'audit.json').write_text(json.dumps(snapshots, ensure_ascii=False, indent=2), encoding='utf-8')
        (directory / 'source_split.json').write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding='utf-8')
        # Keep large training exports on disk instead of holding a multi-GB ZIP
        # in Streamlit memory. Small exports can also be downloaded as one ZIP.
        if image_bytes <= 128 * 1024**2:
            archive_path = directory.with_suffix('.zip')
            with zipfile.ZipFile(archive_path, 'x', zipfile.ZIP_DEFLATED) as archive:
                for file in sorted(directory.rglob('*')):
                    if file.is_file():
                        archive.write(file, file.relative_to(directory).as_posix())
            manifest['zip_path'] = archive_path.relative_to(self.dataset.root).as_posix()
            manifest['zip_sha256'] = file_sha(archive_path)
        self._record_export(identifier, 'split', manifest, directory)
        return directory
