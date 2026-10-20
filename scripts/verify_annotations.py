"""Real local GroundingDINO, review/export checks and isolated browser fixtures."""

import json
import os
import sys
import unittest
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

import torch
from PIL import Image
from streamlit.testing.v1 import AppTest

from annotations import AnnotationStore
from dataset_store import DatasetStore, file_sha
from grounding_prelabel import GroundingPrelabeler, PrelabelConfig


def seed(root):
    dataset = DatasetStore(root)
    ids = []
    for index in range(3):
        batch = dataset.create_batch(f'公开/合成验证{index}', f'fixture_group_{index}', 'public',
                                     'Isolated functional fixture, not internship data: upstream public demo or synthetic blank', ['unknown'])
        if index < 2:
            path = ROOT / 'demo-images' / f'image{index}.jpeg'
            image = path.read_bytes()
        else:
            from io import BytesIO
            output = BytesIO()
            Image.new('RGB', (640, 480), 'white').save(output, format='PNG')
            image = output.getvalue()
        ids.append(dataset.import_image(image, batch, source_name=['public_scene_0.jpeg', 'public_scene_1.jpeg', 'blank_fixture.png'][index])['asset_id'])
    version = dataset.save_split(dataset.plan_split())
    return dataset, ids, version


def run(app):
    app.run()
    assert not app.exception, [item.message for item in app.exception]
    return app


def widget(app, kind, label):
    return next(item for item in getattr(app, kind) if item.label == label)


def main():
    if '--browser-fixtures-only' in sys.argv:
        path = ROOT / 'reports' / 'annotation_verification.json'
        report = json.loads(path.read_text(encoding='utf-8'))
        assert report['status'] == 'passed'
        dataset, ids, version = seed(ROOT / 'artifacts' / 'annotation_checks' / ('browser_' + uuid.uuid4().hex))
        AnnotationStore(dataset)
        report.update(browser_root=str(dataset.root.relative_to(ROOT)), browser_asset_ids=ids, browser_split=version)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'status': 'browser fixtures ready', 'browser_root': str(dataset.root)}), flush=True)
        return
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    assert result.wasSuccessful() and not result.skipped
    torch.set_num_threads(min(8, torch.get_num_threads()))
    output = ROOT / 'artifacts' / 'annotation_checks' / ('integration_' + uuid.uuid4().hex)
    dataset, ids, version = seed(output / 'catalog')
    store = AnnotationStore(dataset)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = GroundingPrelabeler(device)
    cfg = PrelabelConfig.load()
    predicted = store.prelabel(ids, model, cfg, '自动化功能验证，不是真实人工标注')
    assert all(row['status'] == 'ready' for row in predicted), predicted
    assert predicted[0]['box_count'] > 0 and predicted[1]['box_count'] > 0
    assert not any(store.latest(key) for key in ids)
    try:
        store.export_labels(ids)
        raise AssertionError('Unreviewed predictions must not export')
    except ValueError:
        pass
    # Explicitly simulated edits verify records and export formats, never GT.
    for index, asset_id in enumerate(ids):
        proposal = store.proposals(asset_id)[0]
        boxes = proposal['boxes'] if index < 2 else []
        if index == 0:
            boxes = [{**box, 'xyxy': [box['xyxy'][0] + 1, *box['xyxy'][1:]]} if number == 0 else box for number, box in enumerate(boxes)]
        draft = store.save_review(asset_id, boxes, '自动化测试A', '模拟修改，仅验证流程，不是人工参考标签',
                                  proposal_id=proposal['id'], decision='draft')
        store.save_review(asset_id, boxes, '自动化测试B', '模拟复核通过，仅验证版本链', proposal_id=proposal['id'],
                          expected_revision=draft, decision='approved', negative_confirmed=index == 2)
    labels = store.export_labels(ids)
    training = store.export_split(version)
    with zipfile.ZipFile(labels) as archive:
        assert archive.read(f'labels/{ids[2]}.txt') == b''
        audit = json.loads(archive.read('audit.json'))
        assert all(len(row['history']) == 2 for row in audit)
        assert audit[0]['proposals'][0]['metadata']['runtime']['revision'] == cfg.revision
    from ultralytics.data.utils import check_det_dataset
    with patch('ultralytics.data.utils.check_font'):
        loaded = check_det_dataset(str(training / 'data.yaml'), autodownload=False)
    assert loaded['names'] == {0: 'fire', 1: 'smoke'}
    assert len(list((training / 'images').rglob('*.png'))) == len(list((training / 'labels').rglob('*.txt'))) == 3
    print('Real GroundingDINO, explicit simulated review chain and YOLO training loader passed', flush=True)

    cpu = GroundingPrelabeler('cpu')
    with Image.open(ROOT / 'demo-images' / 'image0.jpeg') as image:
        cpu_result = cpu.predict(image, cfg)
    assert cpu_result['boxes']
    print('CPU fallback with pinned GroundingDINO passed', flush=True)

    # Test ordinary Streamlit controls; the real canvas is covered in Edge.
    with patch.dict(os.environ, {'FIRE_DATASET_ROOT': str(dataset.root)}), patch('annotation_ui.box_editor', side_effect=lambda path, width, height, boxes, document_key: {'document_key': document_key, 'boxes': boxes}):
        app = run(AppTest.from_file(str(next((ROOT / 'pages').glob('6_*.py'))), default_timeout=120))
        widget(app, 'radio', '标注工作区').set_value('人工复核')
        run(app)
        assert any('当前画布' in row.value for row in app.caption)
        widget(app, 'radio', '标注工作区').set_value('YOLO导出')
        run(app)
        assert not widget(app, 'button', '导出已复核标签 ZIP').disabled
        assert len(app.get('download_button')) == 1
    browser_dataset, browser_ids, browser_version = seed(output / 'browser_catalog')
    AnnotationStore(browser_dataset)
    report = {'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'status': 'passed',
              'scope': 'Public/synthetic fixtures in isolated catalogs; automated review simulation is not human GT or accuracy evaluation.',
              'unit_tests_passed': result.testsRun, 'root': str(dataset.root.relative_to(ROOT)),
              'browser_root': str(browser_dataset.root.relative_to(ROOT)), 'browser_asset_ids': browser_ids, 'browser_split': browser_version,
              'model': model.metadata, 'effective_prompts': cfg.prompts(),
              'real_model_predictions': predicted, 'cpu_boxes': len(cpu_result['boxes']),
              'unreviewed_export_blocked': True, 'simulated_review_revisions_per_image': 2, 'explicit_negative_empty_label': True,
              'label_zip_sha256': file_sha(labels), 'training_yaml_accepted_by_ultralytics': True,
              'grouped_training_image_counts': {'train': 1, 'val': 1, 'test': 1},
              'ui_checks': ['review_page', 'reviewed_label_export_controls', 'training_export_controls']}
    (ROOT / 'reports' / 'annotation_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'passed', 'tests': result.testsRun, 'browser_root': str(browser_dataset.root)}), flush=True)


if __name__ == '__main__':
    main()
