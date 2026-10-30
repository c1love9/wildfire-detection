"""Frozen queue membership and review UI must not mutate any saved labels."""

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from annotations import AnnotationStore
from dataset_store import DatasetStore
from review_priorities import QUEUE_PATH, load_review_queue, queue_in_catalog, validate_review_queue
from scripts.prepare_priority_review import prepare
from grounding_prelabel import PrelabelConfig
from test_annotations import Model
from test_dataset import image_bytes


ROOT = Path(__file__).resolve().parents[1]


class ReviewQueueTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.document = {'schema_version': 1, 'count': 1, 'task_id': 'completed_task',
                         'batch_id': 'frozen_batch', 'source_report_sha256': 'f' * 64,
                         'entries': [{'number': 'Q103', 'asset_id': 'a' * 64, 'input_sha256': 'b' * 64,
                                      'class': 'smoke', 'original_model_had_class': True, 'proposal_id': 'p1'}]}

    def write(self, document=None):
        path = self.root / QUEUE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document or self.document), encoding='utf-8')

    def test_missing_or_unreadable_queue_does_not_fabricate_members(self):
        self.assertIsNone(load_review_queue(self.root))
        self.write()
        (self.root / QUEUE_PATH).write_text('{bad json', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, '清单无法读取'):
            load_review_queue(self.root)

    def test_invalid_count_duplicate_or_ambiguous_class_is_rejected(self):
        changes = [lambda d: d.update(count=26),
                   lambda d: d.update(entries=d['entries'] * 2, count=2),
                   lambda d: d['entries'][0].update({'class': 'fog'}),
                   lambda d: d['entries'][0].update(original_model_had_class='false'),
                   lambda d: d['entries'][0].update(input_sha256='wrong'),
                   lambda d: d.pop('batch_id')]
        for change in changes:
            document = copy.deepcopy(self.document)
            change(document)
            with self.subTest(document=document), self.assertRaises(ValueError):
                validate_review_queue(document)

    def test_catalog_excludes_missing_or_changed_images_without_substitutes(self):
        catalog = [{'id': 'a' * 64, 'sha256': 'b' * 64, 'decision': 'approved'},
                   {'id': 'c' * 64, 'sha256': 'd' * 64, 'decision': None}]
        available, unavailable = queue_in_catalog(self.document, catalog)
        self.assertEqual([row['asset_id'] for row in available], ['a' * 64])
        self.assertFalse(unavailable)
        for values in (catalog[1:], [{**catalog[0], 'sha256': 'e' * 64}]):
            available, unavailable = queue_in_catalog(self.document, values)
            self.assertFalse(available)
            self.assertEqual(len(unavailable), 1)

    def prepare_fixture(self):
        dataset = DatasetStore(self.root)
        store = AnnotationStore(dataset)
        batch = dataset.create_batch('fixture', 'queue_source', 'own', 'Synthetic source test only', ['unknown'])
        key = dataset.import_image(image_bytes(10), batch)['asset_id']
        candidate = store.prelabel([key], Model([]), PrelabelConfig(), 'synthetic operator')[0]
        snapshot = [{'asset': {'id': key}, 'proposal': {'boxes': [{'class': 'smoke'}]}}]
        with dataset.connection() as db:
            db.executescript('''CREATE TABLE quality_tasks(id TEXT, snapshot_json TEXT);
                CREATE TABLE quality_submissions(task_id TEXT,asset_id TEXT,role TEXT,boxes_json TEXT);''')
            db.execute('INSERT INTO quality_tasks VALUES(?,?)', ('task', json.dumps(snapshot)))
            db.execute('INSERT INTO quality_submissions VALUES(?,?,?,?)', ('task', key, 'A', '[{"class":"smoke"}]'))
            db.execute('UPDATE annotation_runs SET metadata_json=?', (json.dumps({'batch_id': 'batch'}),))
        source = self.root / 'source.json'
        source.write_text(json.dumps([{'number': 'Q001', 'asset_id': key, 'class': 'smoke',
                                      'original_model_had_class': True}]), encoding='utf-8')
        return dataset, key, candidate, source

    def test_prepare_requires_completed_ab_and_never_changes_database(self):
        dataset, key, _, source = self.prepare_fixture()
        with self.assertRaisesRegex(ValueError, '尚未全部提交'):
            prepare(self.root, source, 'task', 'batch')
        self.assertFalse((self.root / QUEUE_PATH).exists())
        with dataset.connection() as db:
            db.execute('INSERT INTO quality_submissions VALUES(?,?,?,?)', ('task', key, 'B', '[{"class":"smoke"}]'))
        with dataset.connection() as db:
            before = list(db.iterdump())
        output, document = prepare(self.root, source, 'task', 'batch')
        self.assertEqual(document['count'], 1)
        original = output.read_bytes()
        prepare(self.root, source, 'task', 'batch')
        self.assertEqual(output.read_bytes(), original)
        with dataset.connection() as db:
            self.assertEqual(list(db.iterdump()), before)

    def test_prepare_rejects_claimed_missing_class_when_candidate_has_it(self):
        dataset, key, candidate, source = self.prepare_fixture()
        with dataset.connection() as db:
            db.execute('INSERT INTO quality_submissions VALUES(?,?,?,?)', ('task', key, 'B', '[{"class":"smoke"}]'))
            db.execute('UPDATE annotation_proposals SET boxes_json=? WHERE id=?',
                       ('[{"class":"smoke"}]', candidate['proposal_id']))
        with self.assertRaisesRegex(ValueError, '缺类依据不成立'):
            prepare(self.root, source, 'task', 'batch')
        self.assertFalse((self.root / QUEUE_PATH).exists())

    def test_ui_scope_pending_empty_and_saved_human_labels(self):
        dataset = DatasetStore(self.root)
        store = AnnotationStore(dataset)
        batch = dataset.create_batch('fixture', 'review_queue', 'own', 'Synthetic test only', ['unknown'])
        first = dataset.import_image(image_bytes(10), batch)['asset_id']
        second = dataset.import_image(image_bytes(11), batch)['asset_id']
        human_boxes = [{'id': '1' * 32, 'class': 'smoke', 'xyxy': [1, 2, 30, 35]}]
        store.save_review(first, human_boxes, 'test_only', 'synthetic saved version', decision='approved')
        document = copy.deepcopy(self.document)
        document['entries'][0].update(asset_id=first, input_sha256=store.image(first)['sha256'])
        self.write(document)
        before = store.history(first)

        def widget(app, kind, label):
            return next(item for item in getattr(app, kind) if item.label == label)

        def run(app):
            app.run()
            self.assertFalse(app.exception, [item.message for item in app.exception])
            return app

        with patch.dict(os.environ, {'FIRE_DATASET_ROOT': str(self.root)}), patch('annotation_ui.box_editor') as editor:
            editor.side_effect = lambda path, width, height, boxes, document_key: {'document_key': document_key, 'boxes': boxes}
            app = run(AppTest.from_file(str(next((ROOT / 'pages').glob('6_*.py'))), default_timeout=30))
            widget(app, 'radio', '标注工作区').set_value('人工复核')
            run(app)
            self.assertEqual(widget(app, 'radio', '复核范围').value, 'priority')
            self.assertEqual(len(widget(app, 'selectbox', '复核图片').options), 1)
            self.assertIn('Q103', widget(app, 'selectbox', '复核图片').options[0])
            self.assertEqual(editor.call_args.args[3], human_boxes)
            self.assertTrue(any('已通过 1 幅' in item.value for item in app.caption))
            widget(app, 'checkbox', '只看尚未通过').check()
            run(app)
            self.assertFalse(any(item.label == '复核图片' for item in app.selectbox))
            self.assertTrue(any('当前筛选下没有图片' in item.value for item in app.info))
            widget(app, 'radio', '复核范围').set_value('all')
            run(app)
            self.assertEqual(widget(app, 'selectbox', '复核图片').value, second)
            widget(app, 'checkbox', '只看尚未通过').uncheck()
            run(app)
            self.assertEqual(len(widget(app, 'selectbox', '复核图片').options), 2)
            widget(app, 'radio', '复核范围').set_value('priority')
            run(app)
            self.assertEqual(widget(app, 'selectbox', '复核图片').value, first)
            self.assertEqual(editor.call_args.args[3], human_boxes)
        self.assertEqual(store.history(first), before)
        self.assertEqual(store.history(second), [])


if __name__ == '__main__':
    unittest.main()
