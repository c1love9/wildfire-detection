"""Real short YOLO training/evaluation on explicitly synthetic, isolated data."""

import os
os.environ.setdefault('YOLO_OFFLINE', 'true')
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from dataclasses import asdict
from io import BytesIO
import json
import random
import time
import unittest
import uuid

from PIL import Image, ImageDraw
from annotations import AnnotationStore
from dataset_store import DatasetStore, file_sha, now
from training import EvalConfig, TrainConfig, TrainingStore, history_rows, read_json, write_json


def seed_fixture(root):
    dataset = DatasetStore(root)
    store = AnnotationStore(dataset)
    for group in range(6):
        batch = dataset.create_batch(f'synthetic verification {group}', f'synthetic_scene_{group}', 'public',
                                     'Generated functional fixture; not real fire, internship data, or independent human GT', ['unknown'])
        for negative in (False, True):
            rng = random.Random(group * 13 + negative)
            image = Image.new('RGB', (192, 128), tuple(rng.randrange(20, 90) for _ in range(3)))
            boxes = []
            if not negative:
                x = rng.randrange(15, 45)
                fire, smoke = [x, 60, x + 35, 115], [x + 18, 10, x + 85, 65]
                draw = ImageDraw.Draw(image)
                draw.rectangle(fire, fill=(245, 80, 20))
                draw.ellipse(smoke, fill=(170, 170, 170))
                boxes = [{'id': uuid.uuid4().hex, 'class': label, 'xyxy': xyxy} for label, xyxy in [('fire', fire), ('smoke', smoke)]]
            buffer = BytesIO()
            image.save(buffer, format='PNG')
            key = dataset.import_image(buffer.getvalue(), batch, source_name=f'generated_{group}_{negative}.png')['asset_id']
            store.save_review(key, boxes, 'synthetic automated fixture', 'Generated reference solely for functional testing; not human GT',
                              decision='approved', negative_confirmed=negative)
    split = dataset.save_split(dataset.plan_split())
    exported = store.export_split(split)
    return dataset, exported.name


def wait(store, identifier, timeout=180):
    started = time.monotonic()
    last = None
    while time.monotonic() - started < timeout:
        job = store.get(identifier, refresh=True)
        summary = (job['status'], job['progress'].get('completed_epochs'))
        if summary != last:
            print(json.dumps({'job': identifier, 'status': summary[0], 'epochs': summary[1]}), flush=True)
            last = summary
        if job['status'] not in ('queued', 'running', 'cancelling'):
            assert job['status'] == 'completed', (job['error'], store.read_log(identifier))
            return store.result(identifier)
        time.sleep(0.5)
    raise TimeoutError(store.read_log(identifier))


def main():
    if '--fixtures-only' in sys.argv:
        root = ROOT / 'artifacts/training_checks' / ('browser_' + uuid.uuid4().hex)
        dataset, export_id = seed_fixture(root)
        write_json(ROOT / 'reports/training_browser_fixture.json', {'root': str(root.relative_to(ROOT)), 'export_id': export_id})
        print(root, flush=True)
        return
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    assert result.wasSuccessful() and not result.skipped
    root = ROOT / 'artifacts/training_checks' / ('integration_' + uuid.uuid4().hex)
    dataset, export_id = seed_fixture(root / 'catalog')
    store = TrainingStore(dataset)
    config = TrainConfig(epochs=2, imgsz=160, batch=2, device='0' if torch.cuda.is_available() else 'cpu', patience=0)
    identifier = store.start_train(export_id, config, 'automated training verification', 'Two-epoch synthetic smoke test; not formal training evidence')
    trained = wait(store, identifier)
    directory = store.directory(identifier)
    rows = history_rows(directory)
    assert len(rows) == 2 and all('train/box_loss' in row and 'metrics/mAP50-95(B)' in row for row in rows)
    assert (directory / 'output/results.png').stat().st_size > 1000
    for name, value in trained['weights'].items():
        assert file_sha(directory / value['path']) == value['sha256']
    from ultralytics import YOLO
    initial, final = YOLO(str(directory / 'initial.pt')), YOLO(str(directory / trained['weights']['best']['path']))
    assert final.names == {0: 'fire', 1: 'smoke'}
    assert not torch.allclose(next(initial.model.parameters()).float(), next(final.model.parameters()).float(), atol=1e-6)
    evaluation = store.start_eval(identifier, EvalConfig(imgsz=160, batch=2, device='cpu'), 'automated CPU test evaluation', 'Explicit held-out synthetic test; not internship accuracy')
    evaluated = wait(store, evaluation)
    assert evaluated['metrics']['split'] == 'test' and evaluated['checkpoint_sha256'] == trained['weights']['best']['sha256']
    assert all(isinstance(evaluated['metrics']['overall'][key], float) for key in ('precision', 'recall', 'f1', 'map50', 'map50_95'))
    cpu = store.start_train(export_id, TrainConfig(epochs=1, imgsz=160, batch=2, device='cpu', patience=0),
                            'automated CPU training', 'One-epoch CPU fallback fixture')
    cpu_result = wait(store, cpu)
    assert cpu_result['completed_epochs'] == 1
    # A second, independent catalog is reserved for real browser interactions.
    browser_dataset, browser_export = seed_fixture(root / 'browser_catalog')
    TrainingStore(browser_dataset)
    report = {'generated_at': now(), 'status': 'passed', 'scope': 'Synthetic isolated functional fixtures, not real human GT or formal accuracy evidence.',
              'unit_tests_passed': result.testsRun, 'root': str(dataset.root.relative_to(ROOT)),
              'browser_root': str(browser_dataset.root.relative_to(ROOT)), 'browser_export': browser_export,
              'train_job': identifier, 'eval_job': evaluation, 'cpu_train_job': cpu, 'training_config': asdict(config),
              'completed_epochs': len(rows), 'loss_and_map_csv': True, 'results_plot_sha256': file_sha(directory / 'output/results.png'),
              'best_and_last_checkpoints': trained['weights'], 'backbone_parameters_updated': True,
              'test_split_not_used_for_model_selection': True, 'explicit_cpu_test_evaluation': evaluated['metrics'],
              'cpu_training_completed': True, 'environment': read_json(directory / 'environment.json')}
    write_json(ROOT / 'reports/training_verification.json', report)
    print(json.dumps({'status': 'passed', 'tests': result.testsRun, 'browser_root': str(browser_dataset.root)}), flush=True)


if __name__ == '__main__':
    main()
