"""Isolated synthetic/public fixtures; never populate the user's real dataset."""

import json
import os
import sys
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

import torch
from streamlit.testing.v1 import AppTest

from camera_source import CameraConfig, CameraSource
from data_collection import capture_clip, capture_snapshot, extract_frames, import_paths, probe_video
from dataset_store import DatasetStore, file_sha
from test_dataset import AdvancingCamera, image_bytes


def run(app):
    app.run()
    assert not app.exception, [item.message for item in app.exception]
    return app


def widget(app, kind, label):
    return next(item for item in getattr(app, kind) if item.label == label)


def main():
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    assert result.wasSuccessful() and not result.skipped
    output = ROOT / 'artifacts' / 'dataset_checks' / ('integration_' + uuid.uuid4().hex)
    source_dir = output / 'sources'
    source_dir.mkdir(parents=True)
    store = DatasetStore(output / 'catalog')
    for index in range(10):
        batch = store.create_batch(f'合成样例批次{index}', f'fixture_{index}', 'own', 'Synthetic verification fixture; not actual self-collected data', ['normal'] if index > 7 else ['fire'],
                                   {'setting': '室外', 'lighting': '白天'})
        path = source_dir / f'{index}.png'
        path.write_bytes(image_bytes(index))
        assert import_paths(store, [path], batch)[0]['asset_id']
        assert import_paths(store, [path], batch)[0]['结果'].startswith('重复')
    # Public demo never becomes an own-data count, even after frame extraction.
    demo = store.create_batch('公开演示视频', 'demo_video', 'demo', 'Upstream demo-videos/fire1.mov', ['smoke'])
    original = ROOT / 'demo-videos' / 'fire1.mov'
    source_hash = file_sha(original)
    video = store.import_video(original, demo)
    frames = extract_frames(store, video['asset_id'], demo, interval=2.0, maximum=10)
    assert frames['sampled'] == 5
    assert file_sha(original) == source_hash
    snapshot = capture_snapshot(store, demo, source_factory=AdvancingCamera)
    clip = capture_clip(store, demo, 1, source_factory=AdvancingCamera, sleep=lambda _: None)
    assert not AdvancingCamera.instances[-1].opened
    plan = store.plan_split(seed=42)
    assert plan['image_counts'] == {'train': 7, 'val': 2, 'test': 1}
    version = store.save_split(plan)
    stored = json.loads((store.root / 'splits' / version / 'manifest.json').read_text(encoding='utf-8'))
    assert len(stored['assets']) == 10 and all(row['eligible'] for row in stored['assets'])
    stats = store.statistics()
    assert stats['own_fire_images'] == 8 and stats['normal_images'] == 2 and stats['own_fire_videos'] == 0
    assert stats['excluded_demo_or_test'] >= 6
    print('Import, exact-pixel dedup, sampling, simulated USB collection and 7:2:1 split passed', flush=True)

    # Hardware smoke check discards the frame; no physical-camera data persisted.
    camera = CameraSource(CameraConfig())
    try:
        sample = camera.read_latest(0)
        assert sample is not None and sample.bgr.size
        physical = {'status': 'passed', 'width': sample.bgr.shape[1], 'height': sample.bgr.shape[0], 'personal_media_saved': False}
    finally:
        camera.release()
    assert not camera.isOpened()

    with patch.dict(os.environ, {'FIRE_DATASET_ROOT': str(store.root)}):
        app = run(AppTest.from_file(str(next((ROOT / 'pages').glob('5_*.py'))), default_timeout=120))
        assert any(item.value == '数据采集与管理' for item in app.title)
        widget(app, 'button', '生成划分预览').click()
        run(app)
        widget(app, 'button', '保存本次划分版本').click()
        run(app)
        assert any('已保存' in item.value for item in app.success)
        assert len(app.get('download_button')) >= 6
        widget(app, 'text_input', '批次名称').set_value('UI 导入测试')
        widget(app, 'text_input', '采集组编号').set_value('ui_group')
        widget(app, 'text_input', '来源说明').set_value('synthetic UI fixture')
        widget(app, 'selectbox', '素材来源').set_value('演示/测试')
        widget(app, 'button', '创建批次').click()
        run(app)
        widget(app, 'text_input', '素材文件夹绝对路径').set_value(str(source_dir))
        widget(app, 'button', '导入文件夹').click()
        run(app)
        assert store.statistics()['total_images'] == stats['total_images']
        assert widget(app, 'button', '保存本次划分版本').disabled
        assert any('已变化' in item.value for item in app.warning)
    # Use a separate empty catalog to verify the UI collection buttons too.
    ui_root = output / 'camera_ui'
    with patch.dict(os.environ, {'FIRE_DATASET_ROOT': str(ui_root)}), patch('data_collection.CameraSource', AdvancingCamera):
        app = run(AppTest.from_file(str(next((ROOT / 'pages').glob('5_*.py'))), default_timeout=120))
        for label, value in [('批次名称', '模拟采集UI'), ('采集组编号', 'fake_camera'), ('来源说明', 'synthetic public-safe frames')]:
            widget(app, 'text_input', label).set_value(value)
        widget(app, 'selectbox', '素材来源').set_value('演示/测试')
        widget(app, 'button', '创建批次').click()
        run(app)
        widget(app, 'button', '采集一张截图').click()
        run(app)
        assert not AdvancingCamera.instances[-1].opened
        widget(app, 'number_input', '短片采集时长（秒）').set_value(1.0)
        widget(app, 'button', '采集一段短片').click()
        run(app)
        assert not AdvancingCamera.instances[-1].opened
        camera_stats = DatasetStore(ui_root).statistics()
        assert (camera_stats['total_images'], camera_stats['total_videos'], camera_stats['eligible_assets']) == (1, 1, 0)

    report = {'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'status': 'passed',
              'scope': 'Synthetic/public fixtures in an isolated catalog; no test item counts as the user internship dataset.',
              'unit_tests_passed': result.testsRun, 'root': str(store.root.relative_to(ROOT)),
              'browser_root': str((output / 'browser_catalog').relative_to(ROOT)), 'physical_camera': physical,
              'public_video_sampled_frames': frames['sampled'], 'public_source_unchanged': True,
              'simulated_usb_snapshot_and_h264_clip': True, 'duplicate_import_does_not_inflate_unique_counts': True,
              'split_images': plan['image_counts'], 'version': version,
              'ui_checks': ['empty_page', 'create_batch', 'folder_import', 'statistics_and_downloads', 'save_split',
                            'changed_data_disables_old_preview', 'simulated_camera_snapshot_and_clip_release'],
              'stale_versions_preserved': all(not row['current'] for row in store.splits())}
    (ROOT / 'reports' / 'dataset_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'passed', 'root': str(store.root), 'tests': result.testsRun}), flush=True)


if __name__ == '__main__':
    main()
