"""Training provenance and lifecycle invariants; fixtures are never formal GT."""

from pathlib import Path
import json
import unittest
from unittest.mock import patch
import uuid

from annotations import AnnotationStore
from dataset_store import DatasetStore, file_sha
from training import (EvalConfig, TrainConfig, TrainingCancelled, TrainingStore, digest_json,
                      history_rows, inspect_export, prepare_training, verify_inventory, write_json)
from test_dataset import image_bytes


ROOT = Path(__file__).resolve().parents[1]


def seed(root):
    dataset = DatasetStore(root)
    annotations = AnnotationStore(dataset)
    ids = []
    for index in range(6):
        batch = dataset.create_batch(f'synthetic{index}', f'synthetic_group{index}', 'public', 'Automated synthetic training fixture; not internship data or independent human GT', ['unknown'])
        key = dataset.import_image(image_bytes(index + 50), batch)['asset_id']
        boxes = [{'id': uuid.uuid4().hex, 'class': 'fire', 'xyxy': [2, 2, 25, 32]},
                 {'id': uuid.uuid4().hex, 'class': 'smoke', 'xyxy': [30, 1, 60, 40]}]
        annotations.save_review(key, boxes, 'automated_fixture', 'Simulated review, not real human labels', decision='approved')
        ids.append(key)
    split = dataset.save_split(dataset.plan_split())
    directory = annotations.export_split(split)
    return dataset, ids, directory.name


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / 'artifacts/training_checks' / ('unit_' + uuid.uuid4().hex)
        self.dataset, self.ids, self.export = seed(self.root)
        self.store = TrainingStore(self.dataset)

    def job(self, **kwargs):
        return self.store.start_train(self.export, TrainConfig(epochs=2, imgsz=160, batch=2, device='cpu'), 'automated_test', 'Functional fixture only', launch=False, **kwargs)

    def test_config_rejects_invalid_values_and_nonfinite_numbers(self):
        for values in ({'epochs': True}, {'batch': 0}, {'imgsz': 161}, {'device': '0,1'}, {'lr0': float('nan')}, {'optimizer': 'auto'}, {'seed': -1}):
            with self.assertRaises(ValueError):
                TrainConfig(**values)
        for values in ({'split': 'train'}, {'iou': 1}, {'conf': False}, {'max_det': 0}):
            with self.assertRaises(ValueError):
                EvalConfig(**values)

    def test_export_requires_real_recorded_approved_training_directory(self):
        with self.assertRaises(ValueError):
            inspect_export(self.dataset, 'unregistered')
        directory, receipt = inspect_export(self.dataset, self.export)
        self.assertEqual(sum(receipt['image_counts'].values()), 6)
        self.assertEqual(receipt['fingerprint'], digest_json(receipt['inventory']))
        verify_inventory(directory, receipt['inventory'])

    def test_manually_edited_label_is_rejected(self):
        directory = self.root / 'exports' / self.export
        next((directory / 'labels').rglob('*.txt')).write_text('0 0.5 0.5 1 1\n')
        with self.assertRaisesRegex(OSError, '标签'):
            inspect_export(self.dataset, self.export)

    def test_modified_image_and_yaml_are_rejected(self):
        directory = self.root / 'exports' / self.export
        yaml = directory / 'data.yaml'
        original = yaml.read_bytes()
        yaml.write_text('path: C:/outside\ntrain: train\n')
        with self.assertRaises(OSError):
            inspect_export(self.dataset, self.export)
        yaml.write_bytes(original)
        next((directory / 'images').rglob('*.png')).write_bytes(b'changed')
        with self.assertRaises(OSError):
            inspect_export(self.dataset, self.export)

    def test_modified_audit_cannot_approve_a_label(self):
        path = self.root / 'exports' / self.export / 'audit.json'
        audit = json.loads(path.read_text(encoding='utf-8'))
        audit[0]['review']['reviewer'] = 'invented'
        write_json(path, audit)
        with self.assertRaisesRegex(OSError, '人工版本'):
            inspect_export(self.dataset, self.export)

    def test_extra_image_cannot_enter_training(self):
        directory = self.root / 'exports' / self.export
        (directory / 'images/train/extra.png').write_bytes(image_bytes(20))
        with self.assertRaisesRegex(OSError, '图像清单'):
            inspect_export(self.dataset, self.export)

    def test_old_reviewed_version_remains_usable_after_new_draft(self):
        annotations = AnnotationStore(self.dataset)
        key = self.ids[0]
        previous = annotations.latest(key)
        annotations.save_review(key, previous['boxes'], 'test', 'New draft does not rewrite frozen export', expected_revision=previous['id'])
        _, receipt = inspect_export(self.dataset, self.export)
        self.assertEqual(receipt['approved_revisions'][key], previous['id'])

    def test_snapshot_is_copied_and_survives_later_export_edits(self):
        identifier = self.job()
        path, receipt = prepare_training(self.store, self.store.get(identifier))
        source = next((self.root / 'exports' / self.export / 'images').rglob('*.png'))
        source.write_bytes(b'changed after copy')
        verify_inventory(path, receipt['inventory'])
        label = next((path / 'labels').rglob('*.txt'))
        label.write_text('')
        with self.assertRaises(OSError):
            verify_inventory(path, receipt['inventory'])

    def test_source_weight_change_blocks_snapshot(self):
        identifier = self.job()
        job = self.store.get(identifier)
        job['request']['initial_weights_sha256'] = '0' * 64
        with self.assertRaisesRegex(OSError, '初始权重'):
            prepare_training(self.store, job)

    def test_only_one_active_job_and_cancellation_is_preserved(self):
        identifier = self.job()
        with self.assertRaisesRegex(ValueError, '已有'):
            self.job()
        self.store.cancel(identifier)
        with self.assertRaises(TrainingCancelled):
            self.store.check_cancelled(identifier)
        from scripts.training_worker import execute
        execute(self.store, identifier)
        self.assertEqual(self.store.get(identifier)['status'], 'cancelled')
        self.assertFalse((self.store.directory(identifier) / 'result.json').exists())

    def test_unfinished_job_cannot_be_evaluated(self):
        identifier = self.job()
        with self.assertRaises(ValueError):
            self.store.start_eval(identifier, EvalConfig(), 'test', 'not ready', launch=False)

    def test_actor_and_experiment_note_required(self):
        for actor, note in [('', 'note'), ('actor', '')]:
            with self.assertRaises(ValueError):
                self.store.start_train(self.export, TrainConfig(), actor, note, launch=False)

    def test_failed_worker_is_not_success_and_new_job_can_start(self):
        identifier = self.job()
        from scripts.training_worker import execute
        with patch('scripts.training_worker.prepare_training', side_effect=OSError('controlled disk failure')):
            execute(self.store, identifier)
        self.assertEqual(self.store.get(identifier)['status'], 'failed')
        self.assertIn('controlled disk failure', self.store.get(identifier)['error'])
        self.assertNotEqual(self.job(), identifier)

    def test_missing_gt_class_is_null_and_no_predictions_is_zero(self):
        from types import SimpleNamespace
        from scripts.training_worker import metric_report
        metrics = SimpleNamespace(box=SimpleNamespace(ap_class_index=[]), speed={'inference': 10.0})
        report = metric_report(metrics, {'fire': 2, 'smoke': 0}, {'split': 'test'}, 1)
        self.assertEqual(report['overall']['map50'], 0)
        self.assertIsNone(report['per_class'][1]['map50'])
        self.assertEqual(report['inference_fps'], 100)
        empty = metric_report(metrics, {'fire': 0, 'smoke': 0}, {'split': 'test'}, 1)
        self.assertIsNone(empty['overall']['map50'])

    def test_abandoned_process_marks_interrupted_without_touching_completed(self):
        identifier = self.job()
        with self.dataset.connection() as db:
            db.execute("UPDATE training_jobs SET created_at='2000-01-01T00:00:00+00:00' WHERE id=?", (identifier,))
        self.assertEqual(self.store.get(identifier, refresh=True)['status'], 'interrupted')

    def test_live_history_ignores_an_incomplete_worker_write(self):
        output = self.root / 'fake_job/output'
        output.mkdir(parents=True)
        (output / 'results.csv').write_text('epoch,train/box_loss\n1,2.5\n2,1e-')
        self.assertEqual(history_rows(output.parent), [{'epoch': 1.0, 'train/box_loss': 2.5}])

    def test_result_and_receipt_must_match_recorded_completion_hashes(self):
        identifier = self.job()
        directory = self.store.directory(identifier)
        inventory = {'data.yaml': 'fake-test-hash'}
        receipt = {'inventory': inventory, 'fingerprint': digest_json(inventory)}
        write_json(directory / 'dataset_receipt.json', receipt)
        write_json(directory / 'result.json', {'dataset_fingerprint': receipt['fingerprint']})
        self.store.update(identifier, status='completed', progress={'result_sha256': file_sha(directory / 'result.json'),
                                                                  'receipt_sha256': file_sha(directory / 'dataset_receipt.json')})
        self.assertEqual(self.store.result(identifier)['dataset_fingerprint'], receipt['fingerprint'])
        write_json(directory / 'result.json', {'dataset_fingerprint': 'edited'})
        with self.assertRaises(OSError):
            self.store.result(identifier)


if __name__ == '__main__':
    unittest.main()
