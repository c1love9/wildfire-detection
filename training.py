"""Versioned local training jobs over verified, reviewed YOLO exports."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass

import psutil
import yaml

from annotations import AnnotationStore, CLASSES, yolo_text
from dataset_store import DatasetStore, file_sha, now


ROOT = Path(__file__).resolve().parent
ACTIVE = ('queued', 'running', 'cancelling')
DATA_YAML = {'train': 'images/train', 'val': 'images/val', 'test': 'images/test', 'names': {0: 'fire', 1: 'smoke'}}


def write_json(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    pending.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def bounded(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('文件路径超出任务目录。')
    return path


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 50
    imgsz: int = 640
    batch: int = 8
    device: str = '0'
    optimizer: str = 'AdamW'
    lr0: float = 0.001
    patience: int = 15
    seed: int = 42
    multi_scale: bool = False

    def __post_init__(self):
        if type(self.multi_scale) is not bool:
            raise ValueError('多尺度训练开关须为布尔值。')
        for name, low, high in [('epochs', 1, 500), ('imgsz', 64, 1280), ('batch', 1, 64), ('patience', 0, 500), ('seed', 0, 2147483647)]:
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'{name}须为{low}～{high}的整数。')
        if self.imgsz % 32 or self.device not in ('cpu', '0') or self.optimizer not in ('AdamW', 'SGD'):
            raise ValueError('图像尺寸须为32的倍数，设备为CPU/单GPU，优化器为AdamW/SGD。')
        if isinstance(self.lr0, bool) or not isinstance(self.lr0, (int, float)) or not math.isfinite(self.lr0) or not 0 < self.lr0 <= 0.1:
            raise ValueError('初始学习率须在0～0.1之间。')

    @classmethod
    def load(cls):
        return cls(**read_json(ROOT / 'configs/training.json'))


@dataclass(frozen=True)
class EvalConfig:
    split: str = 'test'
    imgsz: int = 640
    batch: int = 8
    device: str = '0'
    conf: float = 0.001
    iou: float = 0.7
    max_det: int = 300

    def __post_init__(self):
        TrainConfig(imgsz=self.imgsz, batch=self.batch, device=self.device)
        if self.split not in ('val', 'test') or type(self.max_det) is not int or not 1 <= self.max_det <= 1000:
            raise ValueError('评估仅使用val/test，最多检测数须为1～1000。')
        for value in (self.conf, self.iou):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1:
                raise ValueError('评估置信度和NMS IoU须在0～1之间。')


def verify_inventory(directory, inventory):
    for relative, sha in inventory.items():
        if file_sha(bounded(directory, relative)) != sha:
            raise OSError(f'数据/标签校验失败：{relative}')
    expected_images = {name for name in inventory if name.startswith('images/')}
    actual_images = {path.relative_to(directory).as_posix() for path in (Path(directory) / 'images').rglob('*') if path.is_file()}
    if actual_images != expected_images:
        raise OSError('图像清单发生变化。')


def inspect_export(dataset, export_id):
    """Validate against catalog revisions, not editable TXT/JSON assertions."""
    store = AnnotationStore(dataset)
    record = next((row for row in store.exports() if row['id'] == export_id and row['kind'] == 'split'), None)
    if not record:
        raise ValueError('请选择已复核导出的YOLO训练数据集，标签ZIP不能直接训练。')
    directory = bounded(dataset.root / 'exports', Path(record['path']).relative_to('exports'))
    manifest = read_json(directory / 'manifest.json')
    trusted = {key: value for key, value in record['manifest'].items() if key not in ('zip_path', 'zip_sha256')}
    if manifest != trusted or manifest['classes'] != {'0': 'fire', '1': 'smoke'}:
        raise OSError('导出清单与版本库索引不一致。')
    split_record = next((row for row in dataset.splits() if row['id'] == manifest['source_split']), None)
    if not split_record or split_record['manifest_sha256'] != manifest['source_split_sha256']:
        raise OSError('来源划分缺少可信校验记录。')
    source_path = bounded(dataset.root, split_record['manifest_path'])
    if file_sha(source_path) != split_record['manifest_sha256']:
        raise OSError('原始划分被修改。')
    split = read_json(directory / 'source_split.json')
    if split != read_json(source_path) or yaml.safe_load((directory / 'data.yaml').read_text(encoding='utf-8')) != DATA_YAML:
        raise OSError('划分或data.yaml被修改。')
    images = {row['id']: row for row in split['assets'] if row['kind'] == 'image'}
    audit = read_json(directory / 'audit.json')
    if set(images) != set(manifest['approved_revisions']) or len(audit) != len(images) or {row['asset']['id'] for row in audit} != set(images):
        raise OSError('图像、复核记录与划分不匹配。')
    inventory = {name: file_sha(directory / name) for name in ('data.yaml', 'manifest.json', 'audit.json', 'source_split.json')}
    counts = {name: {label: 0 for label in CLASSES} for name in ('train', 'val', 'test')}
    groups = {}
    for item in audit:
        asset, review = item['asset'], item['review']
        key = asset['id']
        frozen = images[key]
        history = store.history(key)
        original = next((row for row in history if row['id'] == manifest['approved_revisions'][key]), None)
        if original != review or review['decision'] != 'approved' or review['input_sha256'] != frozen['sha256']:
            raise OSError('标签没有对应的已通过人工版本。')
        if item['history'] != [row for row in history if row['seq'] <= review['seq']]:
            raise OSError('修改记录与原始版本不一致。')
        if not review['boxes'] and not review['negative_confirmed']:
            raise OSError('空标签缺少明确负样本确认。')
        name = frozen['split']
        for group in frozen['groups']:
            if group in groups and groups[group] != name:
                raise OSError('采集组跨集合，存在数据泄漏。')
            groups[group] = name
        image_name, label_name = f'images/{name}/{key}.png', f'labels/{name}/{key}.txt'
        if file_sha(directory / image_name) != frozen['sha256']:
            raise OSError('导出图像与划分校验不一致。')
        expected = yolo_text(review['boxes'], frozen['width'], frozen['height'])
        if (directory / label_name).read_text(encoding='utf-8') != expected:
            raise OSError('YOLO标签与已通过人工版本不一致。')
        inventory[image_name], inventory[label_name] = file_sha(directory / image_name), file_sha(directory / label_name)
        for box in review['boxes']:
            counts[name][box['class']] += 1
    for name in ('train', 'val', 'test'):
        expected = {f'{key}.txt' for key, row in images.items() if row['split'] == name}
        if {path.name for path in (directory / 'labels' / name).glob('*.txt')} != expected:
            raise OSError('标签清单有额外或缺失文件。')
        if not expected:
            raise ValueError('训练、验证、测试集都须有图片。')
    verify_inventory(directory, inventory)
    if any(sum(counts[name].values()) == 0 for name in ('train', 'val')):
        raise ValueError('训练集和验证集都须至少有一个已复核火焰或烟雾框。')
    return directory, {'export_id': export_id, 'source_split': manifest['source_split'], 'inventory': inventory,
                       'fingerprint': digest_json(inventory), 'image_counts': manifest['image_counts'], 'box_counts': counts,
                       'approved_revisions': manifest['approved_revisions']}


class TrainingStore:
    def __init__(self, dataset: DatasetStore):
        self.dataset = dataset
        self.root = dataset.root / 'training'
        self.root.mkdir(parents=True, exist_ok=True)
        with dataset.connection() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS training_jobs (
                id TEXT PRIMARY KEY, kind TEXT NOT NULL, parent_id TEXT REFERENCES training_jobs(id),
                status TEXT NOT NULL, actor TEXT NOT NULL, note TEXT NOT NULL, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, pid INTEGER, process_started REAL,
                request_json TEXT NOT NULL, progress_json TEXT NOT NULL, error TEXT)''')

    def directory(self, job_id):
        if not isinstance(job_id, str) or not job_id.startswith(('train_', 'eval_')) or len(job_id.split('_')[-1]) != 32:
            raise ValueError('任务编号无效。')
        return bounded(self.root, job_id)

    def jobs(self, refresh=True):
        with self.dataset.connection() as db:
            rows = [dict(row) for row in db.execute('SELECT * FROM training_jobs ORDER BY created_at DESC,id')]
        for row in rows:
            row['request'] = json.loads(row.pop('request_json'))
            row['progress'] = json.loads(row.pop('progress_json'))
            if refresh and row['status'] in ACTIVE:
                alive = False
                try:
                    process = psutil.Process(row['pid']) if row['pid'] else None
                    alive = bool(process and abs(process.create_time() - row['process_started']) < 0.1 and process.is_running())
                except (psutil.Error, TypeError):
                    pass
                # Leave a short window for the parent to register a newly spawned worker.
                from datetime import datetime, timezone
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(row['created_at'])).total_seconds()
                if not alive and age > 15:
                    with self.dataset.connection() as db:
                        db.execute("UPDATE training_jobs SET status='interrupted',updated_at=?,error=? WHERE id=? AND status IN ('queued','running','cancelling')",
                                   (now(), '工作进程已退出，保留日志与已有文件；请新建任务重试。', row['id']))
                    row.update(status='interrupted', error='工作进程已退出，保留日志与已有文件；请新建任务重试。')
        return rows

    def get(self, job_id, refresh=False):
        return next((row for row in self.jobs(refresh) if row['id'] == job_id), None)

    def update(self, job_id, *, status=None, progress=None, error=None):
        with self.dataset.connection() as db:
            db.execute('UPDATE training_jobs SET status=COALESCE(?,status),progress_json=COALESCE(?,progress_json),error=?,updated_at=? WHERE id=?',
                       (status, json.dumps(progress) if progress is not None else None, error, now(), job_id))

    def _create(self, kind, request, actor, note, parent=None, launch=True):
        if not isinstance(actor, str) or not actor.strip() or len(actor) > 100 or not isinstance(note, str) or not note.strip() or len(note) > 2000:
            raise ValueError('请填写执行者（最多100字）与实验说明（最多2000字）。')
        self.jobs()
        identifier = kind + '_' + uuid.uuid4().hex
        with self.dataset.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute("SELECT 1 FROM training_jobs WHERE status IN ('queued','running','cancelling')").fetchone():
                raise ValueError('当前数据目录已有训练/评估任务，请等待完成或取消。')
            db.execute('INSERT INTO training_jobs VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',
                       (identifier, kind, parent, 'queued', actor.strip(), note.strip(), now(), now(), None, None, json.dumps(request), '{}', None))
        directory = self.directory(identifier)
        directory.mkdir()
        write_json(directory / 'request.json', {'id': identifier, 'kind': kind, 'actor': actor.strip(), 'note': note.strip(), **request})
        if launch:
            try:
                env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'MPLBACKEND': 'Agg', 'YOLO_OFFLINE': 'true', 'YOLO_AUTOINSTALL': 'false'}
                with (directory / 'worker.log').open('ab', buffering=0) as log:
                    process = subprocess.Popen([sys.executable, str(ROOT / 'scripts/training_worker.py'), '--root', str(self.dataset.root), '--job', identifier],
                                               cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                               # Separate from the Windows service/terminal job as well
                                               # as its console, so closing it does not kill training.
                                               creationflags=(subprocess.CREATE_NO_WINDOW | 0x01000000) if os.name == 'nt' else 0,
                                               start_new_session=os.name != 'nt')
                with self.dataset.connection() as db:
                    db.execute('UPDATE training_jobs SET pid=?,process_started=? WHERE id=?', (process.pid, psutil.Process(process.pid).create_time(), identifier))
            except Exception as exc:
                self.update(identifier, status='failed', error=str(exc))
                raise
        return identifier

    def start_train(self, export_id, config, actor, note, launch=True):
        if not isinstance(config, TrainConfig):
            raise ValueError('训练参数无效。')
        if not any(row['id'] == export_id and row['kind'] == 'split' for row in AnnotationStore(self.dataset).exports()):
            raise ValueError('请先导出已复核的完整YOLO训练数据集。')
        weights = ROOT / 'general-models' / 'yolov8n.pt'
        return self._create('train', {'export_id': export_id, 'config': asdict(config),
                                     'initial_weights': str(weights), 'initial_weights_sha256': file_sha(weights),
                                     'initialization': 'Upstream YOLOv8n COCO pretrained; train a new 2-class fire/smoke detector'}, actor, note, launch=launch)

    def start_eval(self, training_id, config, actor, note, launch=True):
        parent = self.get(training_id, refresh=True)
        if not parent or parent['kind'] != 'train' or parent['status'] != 'completed' or not isinstance(config, EvalConfig):
            raise ValueError('请选择已完成训练的最佳权重和有效评估参数。')
        return self._create('eval', {'training_id': training_id, 'config': asdict(config)}, actor, note, parent=training_id, launch=launch)

    def cancel(self, job_id):
        with self.dataset.connection() as db:
            db.execute("UPDATE training_jobs SET status='cancelling',updated_at=? WHERE id=? AND status IN ('queued','running')", (now(), job_id))

    def check_cancelled(self, job_id):
        if self.get(job_id)['status'] == 'cancelling':
            raise TrainingCancelled('用户请求取消；已有日志/轮次文件保留。')

    def result(self, job_id):
        job = self.get(job_id)
        if not job or job['status'] != 'completed':
            raise ValueError('任务尚未成功完成。')
        directory = self.directory(job_id)
        for name, key in [('result.json', 'result_sha256'), ('dataset_receipt.json', 'receipt_sha256')]:
            if file_sha(directory / name) != job['progress'][key]:
                raise OSError('任务结果或数据凭据被改动。')
        result, receipt = read_json(directory / 'result.json'), read_json(directory / 'dataset_receipt.json')
        if digest_json(receipt['inventory']) != receipt['fingerprint'] or result['dataset_fingerprint'] != receipt['fingerprint']:
            raise OSError('任务数据快照指纹不一致。')
        return result

    def read_log(self, job_id, maximum=18000):
        path = self.directory(job_id) / 'worker.log'
        if not path.is_file():
            return ''
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - maximum))
            return stream.read().decode('utf-8', errors='replace')


class TrainingCancelled(Exception):
    pass


def prepare_training(store, job):
    source, receipt = inspect_export(store.dataset, job['request']['export_id'])
    directory = store.directory(job['id'])
    size = sum((source / name).stat().st_size for name in receipt['inventory'])
    if shutil.disk_usage(directory).free < size + 512 * 1024**2:
        raise OSError('磁盘不足：数据快照之外至少保留512MiB。')
    target = directory / 'input'
    target.mkdir()
    for name, sha in receipt['inventory'].items():
        store.check_cancelled(job['id'])
        destination = bounded(target, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, destination)
        if file_sha(destination) != sha:
            raise OSError('数据在复制时改变，任务已停止。')
    initial = Path(job['request']['initial_weights'])
    shutil.copyfile(initial, directory / 'initial.pt')
    if file_sha(directory / 'initial.pt') != job['request']['initial_weights_sha256']:
        raise OSError('初始权重发生变化。')
    write_json(directory / 'dataset_receipt.json', receipt)
    return target, receipt


def history_rows(directory):
    path = Path(directory) / 'output' / 'results.csv'
    if not path.is_file():
        return []
    lines = path.read_text(encoding='utf-8').splitlines(keepends=True)
    if lines and not lines[-1].endswith('\n'):
        lines.pop()  # A worker may still be writing the final row.
    rows = []
    for row in csv.DictReader(lines):
        if None in row or any(value is None or not value.strip() for value in row.values()):
            continue
        try:
            rows.append({key.strip(): float(value) for key, value in row.items()})
        except ValueError:
            continue
    return rows
