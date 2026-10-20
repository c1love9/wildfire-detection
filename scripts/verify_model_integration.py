"""Real self-trained inference and existing alarm/media pipelines, isolated data."""

import os
os.environ.setdefault('YOLO_OFFLINE', 'true')
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))

import torch
import cv2
from dataclasses import replace
import json
import time
import unittest
from unittest.mock import patch
import uuid
from PIL import Image
from streamlit.testing.v1 import AppTest

from alarm import AlarmConfig
from alarm_log import AlarmLog, alarm_media_root, events_csv
from camera_session import CameraSession
from camera_source import CameraConfig
from dataset_store import now
from dataset_store import DatasetStore
from inference_ui import load_detector
from model_catalog import fire_models
from recording import RecordingConfig
from training import TrainConfig, TrainingStore, write_json
from video_detection import VideoSession
from verify_training import seed_fixture, wait
from verify_recording import inspect_media
from verify_camera import hardware_check
from test_camera import FakeCamera


def run(app):
    app.run()
    errors = [item.value for item in app.error if not item.value.startswith('当前报警等级：')]
    assert not app.exception and not errors, [item.message for item in app.exception] + errors
    return app


def widget(app, kind, label):
    return next(item for item in getattr(app, kind) if item.label == label)


def main():
    tests = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_*.py'))
    assert tests.wasSuccessful() and not tests.skipped
    torch.set_num_threads(min(8, torch.get_num_threads()))
    if os.environ.get('MODEL_VERIFY_ROOT'):
        output = Path(os.environ['MODEL_VERIFY_ROOT']).resolve()
        assert output.is_relative_to(ROOT / 'artifacts/model_checks')
        dataset = DatasetStore(output / 'catalog')
        training = TrainingStore(dataset)
        identifier = next(row['id'] for row in training.jobs() if row['kind'] == 'train' and row['status'] == 'completed')
        trained = training.result(identifier)
    else:
        output = ROOT / 'artifacts/model_checks' / ('integration_' + uuid.uuid4().hex)
        dataset, exported = seed_fixture(output / 'catalog')
        training = TrainingStore(dataset)
        identifier = training.start_train(exported, TrainConfig(epochs=40, imgsz=160, batch=2, device='0' if torch.cuda.is_available() else 'cpu', patience=0),
                                          'Automated model integration', 'Synthetic shape detector for functional integration only; not real fire training')
        trained = wait(training, identifier, timeout=240)
    options = fire_models(ROOT, dataset.root)
    trained_key = 'trained:' + identifier + ':best'
    choice = options[trained_key]
    positive = next(path for path in (training.directory(identifier) / 'input/labels/train').glob('*.txt') if path.read_text().strip())
    image_path = training.directory(identifier) / 'input/images/train' / (positive.stem + '.png')
    bgr = cv2.imread(str(image_path))
    devices = ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else [])
    image_checks = []
    for device in devices:
        detector = load_detector(choice, device)
        rgb, detections, counts, _ = detector.predict_image(Image.open(image_path), 0.05, 0.5)
        assert counts.get('fire', 0) and counts.get('smoke', 0), counts
        assert detector.metadata['classes'] == {0: 'fire', 1: 'smoke'}
        Image.fromarray(rgb).save(output / ('image_' + device.replace(':', '_') + '.png'))
        image_checks.append({'device': device, 'counts': counts, 'model_sha256': detector.metadata['sha256']})
    detector = load_detector(choice, devices[-1])
    upstream = load_detector(options['fire_n'], devices[-1])
    assert upstream.metadata['classes'] == {0: 'smoke', 1: 'fire'}
    video = output / 'synthetic_scene.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (bgr.shape[1], bgr.shape[0]))
    assert writer.isOpened()
    for _ in range(40): writer.write(bgr)
    writer.release()
    config = replace(AlarmConfig(), confirm_frames=2, attention_seconds=0.1, warning_seconds=0.3, alarm_seconds=0.5,
                     recovery_seconds=0.2, fire_confidence=0.05, smoke_confidence=0.05)
    recording = replace(RecordingConfig(), pre_seconds=0.2, post_seconds=0.4, output_fps=10)
    log = output / ('alarms_' + uuid.uuid4().hex + '.sqlite3')
    session = VideoSession(video, detector, conf=0.05, alarm_config=config, log_path=log,
                           recording_config=recording, source_name='synthetic self-trained video')
    while session.status == 'running': session.step()
    session.wait_for_media()
    assert session.status == 'completed' and session.processed_frames == 40
    assert session.alarms.snapshot.level == 3
    store = AlarmLog(log)
    event = store.events(source='synthetic self-trained video')[0]
    assert json.loads(event['metadata_json'])['model']['training_id'] == identifier
    video_media = inspect_media(store, event, alarm_media_root(log))
    assert video_media['status'] == 'ready'
    print('Self-trained image CPU/GPU, video alarms and H.264 media passed', flush=True)

    def factory(config): return FakeCamera(config, bgr=bgr)
    camera = CameraSession(CameraConfig(), detector, conf=0.05, alarm_config=config, log_path=log,
                           recording_config=recording, source_factory=factory, source_name='controlled self-trained camera')
    for _ in range(15): camera.step(); camera.capture.emit()
    assert camera.alarms.snapshot.level == 3
    first_run = camera.alarms.run_id
    camera.pause()
    camera.resume()
    assert camera.alarms.run_id != first_run and camera.alarms.event_count == 0
    for _ in range(15): camera.step(); camera.capture.emit()
    camera.stop(); camera.wait_for_media()
    camera_events = store.events(source='controlled self-trained camera')
    assert len(camera_events) == 2
    camera_media = [inspect_media(store, item, alarm_media_root(log)) for item in camera_events]
    assert all(json.loads(item['metadata_json'])['model']['sha256'] == choice.sha256 for item in camera_events)
    hardware = hardware_check(detector)
    print('Self-trained physical camera inference/release/resume and controlled alarm media passed', flush=True)

    # Real pages use the same catalog/cache. Controlled camera pixels are synthetic.
    class AdvancingCamera(FakeCamera):
        def __init__(self, cfg): super().__init__(cfg, stamp=time.monotonic(), bgr=bgr)
        def read_latest(self, after_sequence):
            if after_sequence: self.emit()
            return super().read_latest(after_sequence)

    with patch.dict(os.environ, {'FIRE_DATASET_ROOT': str(dataset.root), 'FIRE_ALARM_LOG_PATH': str(log)}):
        home = run(AppTest.from_file(str(next(ROOT.glob('1_*_Home.py'))), default_timeout=120))
        widget(home, 'selectbox', 'Select Model Size').set_value(trained_key)
        run(home)
        assert any(identifier in item.value for item in home.caption)
        widget(home, 'selectbox', 'Inference Device').set_value('CPU'); run(home)
        widget(home, 'radio', 'Select Model Type').set_value('General'); run(home)
        widget(home, 'radio', 'Select Model Type').set_value('Fire Detection'); run(home)
        with patch('camera_session.CameraSource', AdvancingCamera):
            page = run(AppTest.from_file(str(next((ROOT / 'pages').glob('3_*.py'))), default_timeout=120))
            widget(page, 'radio', '视频来源').set_value('摄像头')
            widget(page, 'selectbox', '检测模型').set_value(trained_key)
            widget(page, 'slider', '置信度阈值').set_value(0.05)
            for label, value in [('火焰报警置信度', 0.05), ('烟雾报警置信度', 0.05), ('关注持续时间（秒）', 0.0),
                                 ('预警持续时间（秒）', 0.1), ('报警持续时间（秒）', 0.2), ('确认前保存（秒）', 0.2), ('确认后保存（秒）', 0.4)]:
                widget(page, 'number_input', label).set_value(value)
            run(page)
            widget(page, 'button', '开始检测').click(); run(page)
            for _ in range(10): run(page)
            active = page.session_state['video_detection_session']
            assert active.alarms.snapshot.level == 3 and widget(page, 'selectbox', '检测模型').disabled
            widget(page, 'button', '暂停检测').click(); run(page)
            assert widget(page, 'selectbox', '检测模型').disabled and not active.is_open
            widget(page, 'button', '继续检测').click(); run(page)
            assert active.segment_number == 2 and active.alarms.snapshot.level == 0
            widget(page, 'button', '停止检测').click(); run(page); active.wait_for_media()
            widget(page, 'selectbox', '检测模型').set_value('fire_n'); run(page)
            widget(page, 'button', '开始检测').click(); run(page)
            switched = page.session_state['video_detection_session']
            assert switched.detector.metadata['origin'] == 'upstream' and switched.alarms.run_id != active.alarms.run_id
            widget(page, 'button', '停止检测').click(); run(page); switched.wait_for_media()
        history = run(AppTest.from_file(str(next((ROOT / 'pages').glob('4_*.py'))), default_timeout=120))
        assert any('本次检测模型' in item.value for item in history.caption)
    assert choice.sha256 in events_csv(store.events()).decode('utf-8-sig')
    report = {'generated_at': now(), 'status': 'passed', 'scope': 'Self-trained synthetic shape model; functional integration, not real fire accuracy. Physical camera frames discarded.',
              'unit_tests_passed': tests.testsRun, 'root': str(output.relative_to(ROOT)), 'dataset_root': str(dataset.root.relative_to(ROOT)),
              'training_id': identifier, 'model_key': trained_key, 'model_sha256': choice.sha256, 'training_epochs': trained['completed_epochs'],
              'image_checks': image_checks, 'upstream_class_mapping': upstream.metadata['classes'], 'trained_class_mapping': detector.metadata['classes'],
              'video_frames': 40, 'video_peak_level': event['peak_level'], 'video_media': video_media,
              'controlled_camera_segments': len(camera_events), 'camera_media': camera_media, 'physical_camera': hardware,
              'image_model_switch_and_general_regression': True, 'camera_model_locked_while_running_and_paused': True,
              'stopping_then_switching_creates_new_run': True, 'history_preserves_model_and_csv_hash': True,
              'browser_video': str(video.relative_to(ROOT)), 'browser_image': str(image_path.relative_to(ROOT)), 'alarm_log': str(log.relative_to(ROOT))}
    write_json(ROOT / 'reports/model_integration.json', report)
    print(json.dumps({'status': 'passed', 'tests': tests.testsRun, 'training_id': identifier}), flush=True)


if __name__ == '__main__': main()
