"""Content-aware selection and class-name/provenance invariants."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import torch
import numpy as np
from PIL import Image

from alarm import AlarmConfig
from alarm_log import AlarmLog, AlarmMonitor, events_csv
from dataset_store import DatasetStore, file_sha
from inference_ui import _load, load_detector
from model_catalog import ModelChoice, fire_models, model_columns
from model_runtime import YoloDetector
from training import TrainConfig, TrainingStore, digest_json, write_json
from test_training import seed


ROOT = Path(__file__).resolve().parents[1]


class FakeYolo:
    task = 'detect'

    def __init__(self, names=None):
        self.names = names or {0: 'smoke', 1: 'fire'}
        self.active = self.peak = 0

    def to(self, device):
        self.device = device

    def predict(self, source, **kwargs):
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.kwargs = kwargs
        time.sleep(0.005)
        self.active -= 1
        data = torch.tensor([[1, 2, 20, 30, 0.9, next(key for key, value in self.names.items() if value == 'fire')]])
        return [SimpleNamespace(names=self.names, boxes=SimpleNamespace(data=data),
                                plot=lambda: np.zeros((48, 64, 3), np.uint8), speed={'inference': 1.0, 'preprocess': 0.1, 'postprocess': 0.2})]


class ModelIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / 'artifacts/model_checks' / ('unit_' + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.weights = self.root / 'controlled.pt'
        self.weights.write_bytes(b'controlled unit fixture, not actual weights')

    def tearDown(self):
        _load.clear()

    def completed(self):
        dataset, _, exported = seed(self.root / 'catalog')
        store = TrainingStore(dataset)
        identifier = store.start_train(exported, TrainConfig(imgsz=160, device='cpu'), 'unit simulation', 'Synthetic test completion, not a real training run', launch=False)
        directory = store.directory(identifier)
        (directory / 'output/weights').mkdir(parents=True)
        path = directory / 'output/weights/best.pt'
        path.write_bytes(self.weights.read_bytes())
        inventory = {'test': 'test'}
        receipt = {'inventory': inventory, 'fingerprint': digest_json(inventory)}
        write_json(directory / 'dataset_receipt.json', receipt)
        write_json(directory / 'result.json', {'dataset_fingerprint': receipt['fingerprint'], 'completed_epochs': 1,
                                             'weights': {'best': {'path': 'output/weights/best.pt', 'sha256': file_sha(path)}}})
        store.update(identifier, status='completed', progress={'result_sha256': file_sha(directory / 'result.json'),
                                                               'receipt_sha256': file_sha(directory / 'dataset_receipt.json')})
        return dataset, store, identifier, path

    def test_only_successful_training_jobs_are_candidates(self):
        dataset, store, identifier, _ = self.completed()
        options = fire_models(ROOT, dataset.root)
        key = 'trained:' + identifier + ':best'
        self.assertIn(key, options)
        self.assertIn('fire_n', options)
        self.assertEqual(options[key].imgsz, 160)
        for status in ('queued', 'running', 'cancelled', 'failed', 'interrupted'):
            store.update(identifier, status=status)
            self.assertNotIn(key, fire_models(ROOT, dataset.root))

    def test_modified_and_missing_checkpoint_stays_explicitly_unavailable(self):
        dataset, _, identifier, path = self.completed()
        key = 'trained:' + identifier + ':best'
        path.write_bytes(b'modified')
        choice = fire_models(ROOT, dataset.root)[key]
        self.assertIsNotNone(choice.error)
        with self.assertRaises(ValueError):
            choice.verify()
        path.unlink()
        self.assertIsNotNone(fire_models(ROOT, dataset.root)[key].error)

    def test_altered_result_cannot_publish_a_model(self):
        dataset, store, identifier, _ = self.completed()
        (store.directory(identifier) / 'result.json').write_text('{}')
        self.assertIsNotNone(fire_models(ROOT, dataset.root)['trained:' + identifier + ':best'].error)

    def test_bad_hash_blocks_loading_before_yolo_is_called(self):
        with patch('model_runtime.YOLO') as factory, self.assertRaises(OSError):
            YoloDetector(self.weights, expected_sha256='0' * 64)
        factory.assert_not_called()

    def test_loading_time_modification_is_detected(self):
        def changed(_):
            self.weights.write_bytes(b'changed while loading')
            return FakeYolo()
        with patch('model_runtime.YOLO', side_effect=changed), self.assertRaises(OSError):
            YoloDetector(self.weights)

    def test_class_names_handle_both_numeric_orders(self):
        for names in ({0: 'smoke', 1: 'fire'}, {0: 'fire', 1: 'smoke'}):
            with patch('model_runtime.YOLO', return_value=FakeYolo(names)):
                runtime = YoloDetector(self.weights)
                _, boxes, counts, _ = runtime.predict(np.zeros((48, 64, 3), np.uint8), 0.2, 0.5)
                self.assertEqual((boxes[0]['class'], counts['fire'], counts['smoke']), ('fire', 1, 0))

    def test_general_or_wrong_task_is_rejected_by_fire_runtime(self):
        for fake in (FakeYolo({0: 'person'}), SimpleNamespace(task='segment', names={0: 'fire', 1: 'smoke'})):
            with patch('model_runtime.YOLO', return_value=fake), self.assertRaises(ValueError):
                YoloDetector(self.weights)

    def test_trained_mapping_must_match_review_export(self):
        with patch('model_runtime.YOLO', return_value=FakeYolo()), self.assertRaises(ValueError):
            YoloDetector(self.weights, model_metadata={'origin': 'trained'})

    def test_image_and_video_calls_are_serialized_and_size_is_pinned(self):
        fake = FakeYolo()
        with patch('model_runtime.YOLO', return_value=fake):
            runtime = YoloDetector(self.weights, imgsz=160)
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda index: runtime.predict_image(Image.new('RGB', (64, 48)), 0.2, 0.5) if index % 2 else
                              runtime.predict(np.zeros((48, 64, 3), np.uint8), 0.2, 0.5), range(8)))
        self.assertEqual(fake.peak, 1)
        self.assertEqual(fake.kwargs['imgsz'], 160)
        self.assertFalse(fake.kwargs['half'])

    def test_cache_hit_rechecks_hash_and_device_is_part_of_cache_key(self):
        choice = ModelChoice('test', 'test', 'upstream', str(self.weights), file_sha(self.weights))
        with patch('model_runtime.YOLO', side_effect=lambda _: FakeYolo()):
            first = load_detector(choice, 'cpu')
            self.assertIs(first, load_detector(choice, 'cpu'))
            self.assertIsNot(first, load_detector(choice, 'cuda:0'))
            self.weights.write_bytes(b'changed after cache')
            with self.assertRaises(OSError):
                load_detector(choice, 'cpu')

    def test_metadata_copy_and_historical_log_are_not_changed_by_switch(self):
        with patch('model_runtime.YOLO', return_value=FakeYolo()):
            runtime = YoloDetector(self.weights, model_metadata={'key': 'first', 'origin': 'upstream', 'label': 'first'})
        metadata = runtime.metadata
        metadata['label'] = 'tampered copy'
        self.assertEqual(runtime.metadata['label'], 'first')
        log = self.root / 'alarms.sqlite3'
        monitor = AlarmMonitor(replace(AlarmConfig(), confirm_frames=2, attention_seconds=0), 'fixture', log, {'model': runtime.metadata})
        for index in range(3):
            monitor.update(index + 1, index / 10, [{'class': 'fire', 'confidence': 0.9, 'xyxy': [1, 1, 20, 30]}])
        monitor.finish('stopped')
        original = AlarmLog(log).events()[0]['metadata_json']
        runtime._metadata['label'] = 'later different model'
        event = AlarmLog(log).events()[0]
        self.assertEqual(event['metadata_json'], original)
        self.assertIn('first', events_csv([event]).decode('utf-8-sig'))

    def test_old_log_does_not_invent_weight_hash_or_training_id(self):
        row = model_columns(json.dumps({'weights': 'fire_n.pt', 'device': 'cpu'}))
        self.assertEqual(row['检测模型'], 'fire_n.pt')
        self.assertIsNone(row['权重SHA-256'])
        self.assertIsNone(row['训练任务'])
        self.assertEqual(model_columns('{}')['模型来源'], '旧记录未保存')


if __name__ == '__main__':
    unittest.main()
