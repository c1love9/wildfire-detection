"""Real-model media and UI checks for archived simulated camera locations."""

import json
import os
import sys
import time
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

import cv2
import torch
from streamlit.testing.v1 import AppTest

from alarm import AlarmConfig
from alarm_log import AlarmLog, alarm_media_root, events_csv
from camera_session import CameraSession
from camera_source import CameraConfig
from monitoring import load_monitoring_points, point_from_metadata
from recording import RecordingConfig
from test_camera import FakeCamera
from verify_recording import inspect_media
from video_detection import FireSmokeDetector, VideoSession


def run(app):
    app.run()
    assert not app.exception, [item.message for item in app.exception]
    return app


def widget(app, kind, label):
    return next(item for item in getattr(app, kind) if item.label == label)


def main():
    output = ROOT / 'artifacts' / 'monitoring_checks' / ('integration_' + uuid.uuid4().hex)
    output.mkdir(parents=True)
    database = output / 'events.sqlite3'
    points = load_monitoring_points(ROOT / 'configs' / 'monitoring_points.json')
    alarm_config = AlarmConfig.load(ROOT / 'configs' / 'alarm.json')
    recording = RecordingConfig.load(ROOT / 'configs' / 'recording.json')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(min(8, torch.get_num_threads()))
    cv2.setNumThreads(2)
    detector = FireSmokeDetector(ROOT / 'fire-models' / 'fire_n.pt', device)
    video = VideoSession(ROOT / 'demo-videos' / 'fire1.mov', detector, alarm_config=alarm_config, log_path=database,
                         source_name='公开样例 · 监控点 A', recording_config=recording, monitoring_point=points[0])
    try:
        while video.status == 'running':
            video.step()
        assert video.status == 'completed' and video.processed_frames == 294
    finally:
        video.stop()
        video.wait_for_media()
    event_a = AlarmLog(database).events()[0]
    assert point_from_metadata(event_a['metadata_json']) == points[0]
    media_a = inspect_media(AlarmLog(database), event_a, alarm_media_root(database))
    print('Real video: 294 frames, archived point A and H.264 media passed', flush=True)

    capture = cv2.VideoCapture(str(ROOT / 'demo-videos' / 'fire1.mov'))
    ok, public_bgr = capture.read()
    capture.release()
    assert ok
    factor = min(640 / public_bgr.shape[1], 480 / public_bgr.shape[0], 1)
    public_bgr = cv2.resize(public_bgr, (round(public_bgr.shape[1] * factor), round(public_bgr.shape[0] * factor)))
    camera = CameraSession(CameraConfig(), detector, alarm_config=alarm_config, log_path=database,
                           source_name='模拟摄像头 · 监控点 B', recording_config=recording, monitoring_point=points[1],
                           source_factory=lambda config: FakeCamera(config, bgr=public_bgr))
    try:
        for _ in range(170):
            camera.step()
            camera.capture.emit()
        assert camera.alarms.snapshot.level == 3
    finally:
        camera.stop()
        camera.wait_for_media()
    event_b = AlarmLog(database).events(source='模拟摄像头 · 监控点 B')[0]
    assert point_from_metadata(event_b['metadata_json']) == points[1]
    media_b = inspect_media(AlarmLog(database), event_b, alarm_media_root(database))
    assert len(AlarmLog(database).events()) == 2
    assert all(point_from_metadata(item['metadata_json']) in points for item in AlarmLog(database).events())
    assert '模拟（WGS84）' in events_csv(AlarmLog(database).events()).decode('utf-8-sig')
    print('Controlled camera: real YOLO, archived point B and media passed', flush=True)

    # One short physical-camera smoke check, with recording and event DB disabled.
    hardware = CameraSession(CameraConfig(), detector, monitoring_point=points[0])
    try:
        deadline = time.monotonic() + 10
        while hardware.processed_frames < 10 and hardware.status == 'running' and time.monotonic() < deadline:
            hardware.step()
            time.sleep(0.08)
        assert hardware.status == 'running' and hardware.processed_frames == 10, hardware.error
        hardware.pause()
        assert not hardware.is_open
    finally:
        hardware.stop()
    print('Physical camera: 10 frames and release; no personal images/logs saved', flush=True)

    with patch.dict(os.environ, {'FIRE_ALARM_LOG_PATH': str(output / 'ui.sqlite3')}):
        app = run(AppTest.from_file(str(next((ROOT / 'pages').glob('3_*.py'))), default_timeout=120))
        widget(app, 'number_input', '模拟水平角（度）').set_value(90.0)
        widget(app, 'number_input', '模拟俯仰角（度）').set_value(-30.0)
        run(app)
        widget(app, 'button', '开始检测').click()
        run(app)
        active = app.session_state['video_detection_session']
        assert (active.monitoring_point.heading, active.monitoring_point.pitch) == (90, -30)
        assert widget(app, 'number_input', '模拟水平角（度）').disabled
        widget(app, 'button', '停止检测').click()
        run(app)
        active.wait_for_media()
        widget(app, 'selectbox', '位置记录方式').set_value('不记录位置')
        run(app)
        assert any('未记录位置' in item.value for item in app.info)

    with patch.dict(os.environ, {'FIRE_ALARM_LOG_PATH': str(database)}):
        history = run(AppTest.from_file(str(next((ROOT / 'pages').glob('4_*.py'))), default_timeout=30))
        for source, point in (('公开样例 · 监控点 A', points[0]), ('模拟摄像头 · 监控点 B', points[1])):
            widget(history, 'selectbox', '输入来源筛选').set_value(source)
            run(history)
            assert any(f'经度 {point.longitude:.6f}' in item.value for item in history.caption)
            assert len(history.get('deck_gl_json_chart')) == 1
            assert len(history.get('imgs')) == 1 and len(history.get('video')) == 1
            assert history.get('imgs')[0].proto.imgs[0].url.startswith('data:image/jpeg;base64,')
            widget(history, 'checkbox', '显示在线地图底图').uncheck()
            run(history)
            assert len(history.get('imgs')) == 1 and len(history.get('video')) == 1

    report = {
        'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'status': 'passed',
        'scope': 'Functional verification of simulated WGS84 locations and manually configured pose; no real GPS sensor or fire geolocation.',
        'device': device, 'database': str(database.relative_to(ROOT)),
        'real_model_file_frames': video.processed_frames, 'file_media': media_a,
        'controlled_camera_frames': camera.processed_frames, 'controlled_camera_media': media_b,
        'controlled_camera_material': 'Repeated public frame from demo-videos/fire1.mov with controlled 10 Hz timestamps',
        'physical_camera': {'frames': 10, 'released': not hardware.is_open, 'personal_images_or_logs_saved': False},
        'ui_checks': ['manual_pose_edit', 'pose_locked_while_running', 'disable_location', 'file_history_map_and_media',
                      'camera_history_map_and_media', 'inline_snapshots_and_isolated_history_map_controls'],
        'archived_locations_preserved_after_reopen': True, 'csv_includes_simulation_label_and_coordinates': True,
    }
    (ROOT / 'reports' / 'monitoring_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'passed', 'database': str(database), 'ui_checks': len(report['ui_checks'])}), flush=True)


if __name__ == '__main__':
    main()
