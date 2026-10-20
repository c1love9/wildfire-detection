"""Review/export invariants; controlled outputs are not model accuracy data."""

import json
import sqlite3
import unittest
import uuid
import zipfile
from pathlib import Path
from unittest.mock import patch

from annotations import AnnotationStore, box_diff, validate_boxes, yolo_text
from dataset_store import DatasetStore, file_sha
from grounding_prelabel import PrelabelConfig
from test_dataset import image_bytes


ROOT = Path(__file__).resolve().parents[1]


def box(label='fire', xyxy=None, identifier=None):
    return {'id': identifier or uuid.uuid4().hex, 'class': label, 'xyxy': xyxy or [1, 2, 30, 35]}


class Model:
    def __init__(self, boxes=None, fail=False):
        self.boxes = [] if boxes is None else boxes
        self.fail = fail

    def run_metadata(self, config):
        return {'runtime': {'model_id': 'controlled-test-only', 'revision': 'test'}, 'parameters': {'prompts': config.prompts()}}

    def predict(self, image, config):
        if self.fail:
            raise RuntimeError('controlled inference failure')
        return {'boxes': self.boxes}


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / 'artifacts' / 'annotation_checks' / ('unit_' + uuid.uuid4().hex)
        self.dataset = DatasetStore(self.root)
        self.store = AnnotationStore(self.dataset)
        self.batch = self.dataset.create_batch('fixture', 'group0', 'own', 'synthetic unit fixture; not actual self-collected data', ['unknown'])
        self.asset = self.dataset.import_image(image_bytes(10), self.batch)['asset_id']

    def proposal(self, boxes=None, asset=None, fail=False):
        return self.store.prelabel([asset or self.asset], Model(boxes, fail), PrelabelConfig(), 'test_operator')[0]

    def review(self, boxes=None, asset=None, decision='approved', negative=False, **kwargs):
        return self.store.save_review(asset or self.asset, boxes or [], 'reviewer_A', 'controlled test review',
                                      decision=decision, negative_confirmed=negative, **kwargs)

    def split(self):
        for index in (1, 2):
            batch = self.dataset.create_batch(f'b{index}', f'g{index}', 'own', 'synthetic fixture', ['unknown'])
            asset = self.dataset.import_image(image_bytes(index), batch)['asset_id']
            self.review([box()], asset=asset)
        return self.dataset.save_split(self.dataset.plan_split())

    def test_prompt_validation_and_effective_normalization(self):
        self.assertEqual(PrelabelConfig(fire_prompt='Fire, flame; blaze').prompts()['fire'], 'fire. flame. blaze.')
        for values in ({'fire_prompt': ' . '}, {'text_threshold': float('nan')}, {'box_threshold': True}, {'revision': 'main'}):
            with self.assertRaises(ValueError):
                PrelabelConfig(**values)

    def test_invalid_boxes_are_rejected_and_yolo_uses_original_size(self):
        for value in (box('other'), box(xyxy=[-1, 0, 20, 30]), box(xyxy=[0, 0, 65, 30]),
                      box(xyxy=[0, 0, float('nan'), 30]), box(xyxy=[0, 0, True, 30]), box(xyxy=[1, 1, 1.1, 5])):
            with self.assertRaises(ValueError):
                validate_boxes([value], 64, 48)
        label = box('smoke', [0, 0, 64, 48])
        self.assertEqual(yolo_text([label], 64, 48), '1 0.50000000 0.50000000 1.00000000 1.00000000\n')
        with self.assertRaises(ValueError):
            validate_boxes([label, label], 64, 48)

    def test_proposal_retains_metadata_without_approving_image(self):
        proposal = self.proposal([box()])
        self.assertEqual(proposal['status'], 'ready')
        self.assertIsNone(self.store.latest(self.asset))
        row = self.store.proposals(self.asset)[0]
        self.assertIn('parameters', row['metadata'])
        self.assertEqual(row['input_sha256'], self.store.image(self.asset)['sha256'])
        with self.assertRaises(ValueError):
            self.store.export_labels([self.asset])

    def test_failed_inference_is_never_a_negative_annotation(self):
        proposal = self.proposal(fail=True)
        self.assertEqual(proposal['status'], 'failed')
        with self.assertRaisesRegex(ValueError, '预标注'):
            self.review([], negative=True, proposal_id=proposal['proposal_id'])
        self.assertEqual(self.store.history(self.asset), [])

    def test_empty_model_output_requires_explicit_human_negative_confirmation(self):
        proposal = self.proposal([])
        with self.assertRaisesRegex(ValueError, '无目标'):
            self.review([], proposal_id=proposal['proposal_id'])
        self.review([], negative=True, proposal_id=proposal['proposal_id'])
        path = self.store.export_labels([self.asset])
        with zipfile.ZipFile(path) as archive:
            self.assertEqual(archive.read(f'labels/{self.asset}.txt'), b'')
            audit = json.loads(archive.read('audit.json'))
            self.assertTrue(audit[0]['review']['negative_confirmed'])

    def test_edits_store_added_deleted_moved_and_reclassified_boxes(self):
        one, two = box(), box('smoke')
        proposal = self.proposal([one, two])
        changed = {**one, 'class': 'smoke', 'xyxy': [3, 4, 32, 37]}
        extra = box('fire', [40, 1, 60, 20])
        self.review([changed, extra], proposal_id=proposal['proposal_id'])
        revision = self.store.latest(self.asset)
        self.assertEqual([len(revision['diff'][key]) for key in ('added', 'removed', 'changed')], [1, 1, 1])
        self.assertEqual(self.store.proposals(self.asset)[0]['boxes'], [one, two])

    def test_new_prediction_does_not_replace_an_approved_human_version(self):
        identifier = self.review([box()])
        self.proposal([box('smoke')])
        self.assertEqual(self.store.latest(self.asset)['id'], identifier)

    def test_revisions_are_append_only_and_stale_edit_is_rejected(self):
        first = self.review([box()])
        second = self.review([box('smoke')], expected_revision=first)
        with self.assertRaisesRegex(ValueError, '另一窗口'):
            self.review([box()], expected_revision=first)
        history = self.store.history(self.asset)
        self.assertEqual([row['id'] for row in history], [first, second])
        self.assertEqual(history[1]['parent_id'], first)

    def test_new_draft_blocks_export_of_previous_approval(self):
        approved = self.review([box()])
        self.review([box()], expected_revision=approved, decision='draft')
        with self.assertRaises(ValueError):
            self.store.export_labels([self.asset])

    def test_review_requires_actor_note_and_image_matching_proposal(self):
        for name, note in (('', 'note'), ('person', '')):
            with self.assertRaises(ValueError):
                self.store.save_review(self.asset, [box()], name, note)
        other = self.dataset.import_image(image_bytes(20), self.batch)['asset_id']
        proposal = self.proposal([box()], asset=other)
        with self.assertRaises(ValueError):
            self.review([box()], proposal_id=proposal['proposal_id'])

    def test_image_tampering_blocks_prediction_review_and_export(self):
        self.review([box()])
        path = self.dataset.asset_path(self.store.image(self.asset))
        path.write_bytes(b'changed')
        self.assertEqual(self.proposal([box()])['status'], 'failed')
        with self.assertRaises(OSError):
            self.review([box()], expected_revision=self.store.latest(self.asset)['id'])
        with self.assertRaises(OSError):
            self.store.export_labels([self.asset])

    def test_label_export_freezes_review_ids_and_full_change_history(self):
        first = self.review([box()])
        path = self.store.export_labels([self.asset])
        digest = file_sha(path)
        second = self.review([box('smoke')], expected_revision=first)
        self.assertNotEqual(first, second)
        self.assertEqual(file_sha(path), digest)
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read('manifest.json'))
            self.assertEqual(manifest['classes'], {'0': 'fire', '1': 'smoke'})
            self.assertEqual(manifest['approved_revisions'][self.asset], first)
            self.assertEqual(len(json.loads(archive.read('audit.json'))[0]['history']), 1)

    def test_complete_reviewed_split_exports_valid_yolo_directory(self):
        self.review([box()])
        version = self.split()
        directory = self.store.export_split(version)
        self.assertEqual(len(list((directory / 'labels').rglob('*.txt'))), 3)
        self.assertEqual(len(list((directory / 'images').rglob('*.png'))), 3)
        from ultralytics.data.utils import check_det_dataset
        with patch('ultralytics.data.utils.check_font'):
            data = check_det_dataset(str(directory / 'data.yaml'), autodownload=False)
        self.assertEqual(data['names'], {0: 'fire', 1: 'smoke'})
        self.assertEqual(Path(data['train']).parent, directory / 'images')
        self.assertTrue((directory / 'audit.json').is_file())
        self.assertTrue(directory.with_suffix('.zip').is_file())

    def test_unreviewed_image_cannot_be_silently_exported_as_background(self):
        version = self.split()
        with self.assertRaisesRegex(ValueError, '未复核'):
            self.store.export_split(version)
        self.assertEqual(self.store.exports(), [])

    def test_modified_split_manifest_and_stale_data_are_rejected(self):
        self.review([box()])
        version = self.split()
        path = self.root / 'splits' / version / 'manifest.json'
        original = path.read_bytes()
        path.write_bytes(original + b' ')
        with self.assertRaises(OSError):
            self.store.export_split(version)
        path.write_bytes(original)
        self.dataset.import_image(image_bytes(50), self.batch)
        with self.assertRaisesRegex(ValueError, '当前素材'):
            self.store.export_split(version)

    def test_legacy_split_schema_upgrades_without_changing_original_data(self):
        other_root = self.root / 'legacy'
        other_root.mkdir()
        with sqlite3.connect(other_root / 'catalog.sqlite3') as db:
            db.execute('CREATE TABLE splits(id TEXT PRIMARY KEY,created_at TEXT,fingerprint TEXT,manifest_path TEXT,summary_json TEXT)')
        upgraded = DatasetStore(other_root)
        with upgraded.connection() as db:
            self.assertIn('manifest_sha256', {row['name'] for row in db.execute('PRAGMA table_info(splits)')})
        self.assertEqual(upgraded.assets(), [])


if __name__ == '__main__':
    unittest.main()
