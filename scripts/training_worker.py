"""Independent local worker. UI reruns and service restarts do not own its loop."""

import argparse
import os
os.environ.setdefault('YOLO_OFFLINE', 'true')
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
os.environ.setdefault('MPLBACKEND', 'Agg')

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # Load before native image/Arrow libraries on Windows.
import importlib.metadata
import json
import platform
import shutil
import subprocess
import time
import traceback

import psutil
from ultralytics import YOLO
from ultralytics.utils.torch_utils import init_seeds

from dataset_store import DatasetStore, file_sha, now
from training import (CLASSES, EvalConfig, TrainConfig, TrainingCancelled, TrainingStore, history_rows,
                      prepare_training, read_json, verify_inventory, write_json)


def environment():
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
            'cuda_runtime': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
            'git_commit_at_launch': commit, 'worker_sha256': file_sha(Path(__file__)),
            'training_module_sha256': file_sha(ROOT / 'training.py'), 'recorded_at': now()}


def metric_report(metrics, counts, config, wall_seconds):
    """Do not impute absent-class AP from Ultralytics' box.maps fallback."""
    by_class = {int(key): index for index, key in enumerate(metrics.box.ap_class_index)}
    rows = []
    for key, name in enumerate(CLASSES):
        index = by_class.get(key)
        values = [float(v) for v in metrics.box.class_result(index)] if index is not None else [0.0] * 4
        p, r, ap50, ap = values
        if not counts[name]:
            p = r = ap50 = ap = f1 = None
        else:
            f1 = 2 * p * r / (p + r) if p + r else 0.0
        rows.append({'class_id': key, 'class': name, 'instances': counts[name], 'precision': p, 'recall': r,
                     'f1': f1, 'map50': ap50, 'map50_95': ap})
    present = [row for row in rows if row['instances']]
    mean = {key: sum(row[key] for row in present) / len(present) if present else None for key in ('precision', 'recall', 'f1', 'map50', 'map50_95')}
    speed = {key: float(value) for key, value in metrics.speed.items()}
    inference_ms = speed.get('inference', 0)
    pipeline_ms = sum(speed.get(key, 0) for key in ('preprocess', 'inference', 'postprocess'))
    return {'split': config['split'], 'overall': mean, 'per_class': rows, 'speed_ms_per_image': speed,
            'inference_fps': 1000 / inference_ms if inference_ms > 0 else None,
            'preprocess_inference_postprocess_fps': 1000 / pipeline_ms if pipeline_ms > 0 else None,
            'wall_seconds_including_setup_and_plots': wall_seconds,
            'protocol': {**config, 'precision_recall_iou': 0.5, 'ap_iou_range': '0.50:0.05:0.95',
                         'pr_operating_point': 'Ultralytics maximum mean F1 point on this evaluated split, not a fixed deployment confidence',
                         'averaging': 'macro over classes with GT; absent GT -> null; F1 is mean per-class F1',
                         'latency_scope': 'warmed model validation timing per image at recorded batch/device; excludes disk/UI/network'},
            'warning': 'Data provenance/review quality determine validity. Test fixtures do not establish internship accuracy.'}


def execute(store, job_id):
    job = store.get(job_id)
    if not job or job['status'] not in ('queued', 'cancelling'):
        raise ValueError('任务不存在或已经执行，不能覆盖旧结果。')
    directory = store.directory(job_id)
    with store.dataset.connection() as db:
        db.execute('UPDATE training_jobs SET pid=?,process_started=? WHERE id=?', (os.getpid(), psutil.Process().create_time(), job_id))
    try:
        store.check_cancelled(job_id)
        store.update(job_id, status='running', progress={'stage': '校验与准备'})
        torch.set_num_threads(min(8, torch.get_num_threads()))
        write_json(directory / 'environment.json', environment())
        cfg = job['request']['config']
        if cfg['device'] == '0' and not torch.cuda.is_available():
            raise RuntimeError('当前环境没有可用CUDA，请选择CPU新建任务。')
        if job['kind'] == 'train':
            config = TrainConfig(**cfg)
            data, receipt = prepare_training(store, job)
            model = YOLO(str(directory / 'initial.pt'))
            def check(_):
                store.check_cancelled(job_id)
            def started(trainer):
                check(trainer)
                write_json(directory / 'effective_parameters.json', json.loads(json.dumps(vars(trainer.args), default=str)))
            def progress(trainer):
                check(trainer)
                rows = history_rows(directory)
                store.update(job_id, progress={'stage': '训练与验证', 'completed_epochs': len(rows), 'requested_epochs': config.epochs,
                                               'latest': rows[-1] if rows else {}})
            for event in ('on_train_batch_start', 'on_val_batch_start'):
                model.add_callback(event, check)
            model.add_callback('on_train_start', started)
            model.add_callback('on_fit_epoch_end', progress)
            started_at = time.perf_counter()
            model.train(data=str(data / 'data.yaml'), epochs=config.epochs, imgsz=config.imgsz, batch=config.batch,
                        device=config.device, optimizer=config.optimizer, lr0=config.lr0, patience=config.patience,
                        seed=config.seed, deterministic=True, workers=0, amp=False, cache=False, plots=True,
                        save=True, val=True, split='val', conf=0.001, iou=0.7, max_det=300,
                        project=str(directory), name='output', exist_ok=False, resume=False,
                        close_mosaic=min(10, config.epochs // 2), verbose=False)
            store.check_cancelled(job_id)
            verify_inventory(data, receipt['inventory'])
            if model.names != {0: 'fire', 1: 'smoke'}:
                raise RuntimeError('训练后的类别映射不正确。')
            params = {'split': 'val', 'imgsz': config.imgsz, 'batch': model.trainer.args.batch,
                      'device': config.device, 'conf': 0.001, 'iou': 0.7, 'max_det': 300, 'half': False}
            report = metric_report(model.metrics, receipt['box_counts']['val'], params, time.perf_counter() - started_at)
            report['wall_seconds_including_setup_and_plots'] = None
            report['training_wall_seconds'] = time.perf_counter() - started_at
            report['selection'] = 'best.pt selected only by validation mAP50-95; test split not evaluated during training'
            rows = history_rows(directory)
            if not rows or not (directory / 'output/results.png').is_file():
                raise RuntimeError('未生成训练CSV或曲线，请查看日志。')
            weights = {name: {'path': 'output/weights/' + name + '.pt', 'sha256': file_sha(directory / 'output/weights' / (name + '.pt'))} for name in ('best', 'last')}
            result = {'job_id': job_id, 'kind': 'train', 'completed_epochs': len(rows), 'requested_epochs': config.epochs,
                      'dataset_fingerprint': receipt['fingerprint'], 'image_counts': receipt['image_counts'],
                      'weights': weights, 'metrics': report, 'finished_at': now()}
        else:
            config = EvalConfig(**cfg)
            parent_dir = store.directory(job['parent_id'])
            parent_result = store.result(job['parent_id'])
            receipt = read_json(parent_dir / 'dataset_receipt.json')
            if parent_result['dataset_fingerprint'] != receipt['fingerprint']:
                raise OSError('训练数据快照记录不一致。')
            data = parent_dir / 'input'
            verify_inventory(data, receipt['inventory'])
            # Rebuild the selected split's derived label cache from the checked
            # TXT files; a stale/edited .cache must not replace the reviewed GT.
            (data / 'labels' / (config.split + '.cache')).unlink(missing_ok=True)
            best = parent_result['weights']['best']
            shutil.copyfile(parent_dir / best['path'], directory / 'checkpoint.pt')
            if file_sha(directory / 'checkpoint.pt') != best['sha256']:
                raise OSError('训练最佳权重校验失败。')
            model = YOLO(str(directory / 'checkpoint.pt'))
            if model.names != {0: 'fire', 1: 'smoke'}:
                raise RuntimeError('评估模型类别不正确。')
            init_seeds(42, deterministic=True)
            model.add_callback('on_val_batch_start', lambda _: store.check_cancelled(job_id))
            write_json(directory / 'dataset_receipt.json', receipt)
            write_json(directory / 'effective_parameters.json', {**cfg, 'half': False, 'workers': 0, 'augment': False, 'rect': True, 'seed': 42})
            store.update(job_id, progress={'stage': f'评估{config.split}集'})
            started_at = time.perf_counter()
            metrics = model.val(data=str(data / 'data.yaml'), split=config.split, imgsz=config.imgsz, batch=config.batch,
                                device=config.device, conf=config.conf, iou=config.iou, max_det=config.max_det,
                                half=False, workers=0, plots=True, save_json=False, augment=False, rect=True,
                                project=str(directory), name='output', exist_ok=False, verbose=False)
            store.check_cancelled(job_id)
            verify_inventory(data, receipt['inventory'])
            report = metric_report(metrics, receipt['box_counts'][config.split], {**cfg, 'half': False}, time.perf_counter() - started_at)
            result = {'job_id': job_id, 'kind': 'eval', 'training_id': job['parent_id'], 'checkpoint_sha256': best['sha256'],
                      'dataset_fingerprint': receipt['fingerprint'], 'image_counts': receipt['image_counts'], 'metrics': report, 'finished_at': now()}
        write_json(directory / 'metrics.json', report)
        result['artifacts_sha256'] = {path.relative_to(directory).as_posix(): file_sha(path) for path in (directory / 'output').rglob('*') if path.is_file()}
        write_json(directory / 'result.json', result)
        store.update(job_id, status='completed', progress={'stage': '完成', 'completed_epochs': result.get('completed_epochs'), 'split': report['split'],
                                                          'result_sha256': file_sha(directory / 'result.json'),
                                                          'receipt_sha256': file_sha(directory / 'dataset_receipt.json')})
        print(json.dumps({'status': 'completed', 'job': job_id, 'metrics': report['overall']}), flush=True)
    except TrainingCancelled as exc:
        store.update(job_id, status='cancelled', error=str(exc))
        print(str(exc), flush=True)
    except Exception as exc:
        traceback.print_exc()
        store.update(job_id, status='failed', error=str(exc)[-2000:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    store = TrainingStore(DatasetStore(Path(args.root)))
    execute(store, args.job)
    if store.get(args.job)['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
