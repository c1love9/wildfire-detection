"""Camera hardware, controlled camera alarms/media, UI and file regressions."""

import argparse
import json
import os
import sys
import time
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from streamlit.testing.v1 import AppTest

from alarm import AlarmConfig
from alarm_log import AlarmLog, alarm_media_root
from camera_source import CameraConfig, CapturedFrame
from camera_session import CameraSession
from recording import RecordingConfig
from video_detection import FireSmokeDetector, VideoSession
from verify_recording import inspect_media


def run_page(app):
    app.run()
    assert not app.exception, [item.message for item in app.exception]
    return app


def hardware_check(detector):
    # Deliberately do not persist or screenshot any physical-camera image.
    camera = CameraSession(CameraConfig.load(ROOT / 'configs' / 'camera.json'), detector)
    try:
        deadline = time.monotonic() + 10
        while camera.processed_frames < 25 and camera.status == 'running' and time.monotonic() < deadline:
            camera.step()
            time.sleep(0.08)
        assert camera.status == 'running' and camera.processed_frames == 25, camera.error
        before = camera.latest.media_seconds
        first_run = camera.alarms.run_id
        result = dict(frames=25, observed_seconds=before, delivered_width=camera.info.width,
                      delivered_height=camera.info.height, reported_fps=camera.info.fps,
                      backend=camera.capture.info['backend'], skipped_capture_frames=camera.dropped_frames,
                      personal_images_saved=False)
        camera.pause()
        assert not camera.is_open and camera.step() is None
        time.sleep(0.2)
        camera.resume()
        assert camera.alarms.run_id != first_run
        deadline = time.monotonic() + 5
        while camera.processed_frames < 8 and camera.status == 'running' and time.monotonic() < deadline:
            camera.step()
            time.sleep(0.08)
        assert camera.status == 'running' and camera.processed_frames == 8, camera.error
        assert camera.latest.media_seconds < before
        result.update(released_on_pause=True, resumed_frames=8, resumed_in_new_observation=True)
        # Simulate capture-process loss while the actual device is open.
        camera.capture._process.kill()
        deadline = time.monotonic() + 5
        while camera.status == 'running' and time.monotonic() < deadline:
            time.sleep(0.02)
        assert camera.status == 'disconnected' and not camera.is_open
        result['capture_process_loss_detected_without_ui_poll'] = True
    finally:
        camera.stop()
    # Reopen after the forced reader exit, proving device ownership is released.
    reopened = CameraSession(CameraConfig(), detector)
    try:
        assert reopened.step() is not None
    finally:
        reopened.stop()
    result['reopened_after_process_loss'] = True
    result['released_at_end'] = not reopened.is_open
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--skip-hardware', action='store_true')
    args = parser.parse_args()
    output = args.output_dir or ROOT / 'artifacts' / 'camera_checks' / ('integration_' + uuid.uuid4().hex)
    output.mkdir(parents=True, exist_ok=True)
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    assert result.wasSuccessful() and not result.skipped
    from test_camera import FakeCamera
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(min(8, torch.get_num_threads()))
    cv2.setNumThreads(2)
    detector = FireSmokeDetector(ROOT / 'fire-models' / 'fire_n.pt', device)
    report = dict(generated_at=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
                  scope='Functional camera verification; simulated scene timing is not physical fire testing or accuracy data.',
                  unit_tests_passed=result.testsRun, device=device, file_regression=[], ui_checks=[])
    if args.skip_hardware:
        report['physical_camera'] = {'status': 'explicitly_skipped', 'personal_images_saved': False}
    else:
        report['physical_camera'] = {'status': 'passed', **hardware_check(detector)}
        print('Physical camera: inference, release, resume, reader interruption and reopen passed', flush=True)

    alarm_cfg = AlarmConfig.load(ROOT / 'configs' / 'alarm.json')
    recording_cfg = RecordingConfig.load(ROOT / 'configs' / 'recording.json')
    capture = cv2.VideoCapture(str(ROOT / 'demo-videos' / 'fire1.mov'))
    ok, public_bgr = capture.read()
    capture.release()
    assert ok
    factor = min(640 / public_bgr.shape[1], 480 / public_bgr.shape[0], 1)
    public_bgr = cv2.resize(public_bgr, (round(public_bgr.shape[1] * factor), round(public_bgr.shape[0] * factor)))
    database = output / 'camera.sqlite3'
    sources = []
    def factory(config):
        source = FakeCamera(config, bgr=np.zeros_like(public_bgr))
        sources.append(source)
        return source
    camera = CameraSession(CameraConfig(), detector, alarm_config=alarm_cfg, recording_config=recording_cfg,
                           log_path=database, source_name='模拟摄像头（公开样例验证）', source_factory=factory)
    try:
        for index in range(190):
            source = camera.capture
            source.latest = CapturedFrame(source.latest.sequence, source.latest.captured_at,
                                          np.zeros_like(public_bgr) if index < 70 else public_bgr)
            camera.step()
            source.emit()
        assert camera.alarms.snapshot.level == 3
    finally:
        camera.stop()
        camera.wait_for_media()
    event = AlarmLog(database).events()[0]
    controlled_media = inspect_media(AlarmLog(database), event, alarm_media_root(database))
    assert controlled_media['status'] == 'ready' and abs(controlled_media['encoded_duration'] - 15) < 0.04
    report['controlled_camera'] = dict(source_material='Black frames then a repeated public frame from demo-videos/fire1.mov',
                                     timestamps='controlled monotonic at 10 Hz', frames=190, real_model=True,
                                     highest_level=event['peak_level'], event_count=camera.alarms.event_count,
                                     media=controlled_media, camera_session_pipeline=True)
    print('Controlled camera: full 5+10 second H.264 clip and snapshot verified', flush=True)

    for source in sorted((ROOT / 'demo-videos').glob('*')):
        session = VideoSession(source, detector, alarm_config=alarm_cfg, log_path=output / 'files.sqlite3', recording_config=recording_cfg)
        while session.status == 'running':
            session.step()
        session.wait_for_media()
        assert session.status == 'completed' and session.processed_frames == session.info.frame_count
        store = session.alarms.store
        events = store.events(source=source.name)
        media = [inspect_media(store, row, alarm_media_root(store.path)) for row in events]
        assert not session.recorder.last_error
        report['file_regression'].append(dict(source=source.name, frames=session.processed_frames, events=len(events), media=len(media)))
        print(f'File regression {source.name}: {session.processed_frames} frames', flush=True)

    class AdvancingCamera(FakeCamera):
        def __init__(self, config):
            super().__init__(config, stamp=time.monotonic(), bgr=public_bgr)
        def read_latest(self, after_sequence):
            if after_sequence:
                self.emit()
            return super().read_latest(after_sequence)

    with patch.dict(os.environ, {'FIRE_ALARM_LOG_PATH': str(output / 'ui.sqlite3')}), patch('camera_session.CameraSource', AdvancingCamera):
        app = run_page(AppTest.from_file(str(next((ROOT / 'pages').glob('3_*.py'))), default_timeout=120))
        next(widget for widget in app.radio if widget.label == '视频来源').set_value('摄像头')
        run_page(app)
        for label, value in (('关注持续时间（秒）', 0.0), ('预警持续时间（秒）', 0.1), ('报警持续时间（秒）', 0.2),
                             ('确认前保存（秒）', 0.2), ('确认后保存（秒）', 0.3)):
            next(widget for widget in app.number_input if widget.label == label).set_value(value)
        run_page(app)
        next(widget for widget in app.button if widget.label == '开始检测').click()
        run_page(app)
        for _ in range(10):
            run_page(app)
        active = app.session_state['video_detection_session']
        assert active.input_type == 'camera' and active.alarms.snapshot.level == 3
        next(widget for widget in app.button if widget.label == '暂停检测').click()
        run_page(app)
        assert not active.is_open and active.status == 'paused'
        active.wait_for_media()
        next(widget for widget in app.button if widget.label == '继续检测').click()
        run_page(app)
        assert active.segment_number == 2 and active.alarms.snapshot.level == 0
        next(widget for widget in app.button if widget.label == '停止检测').click()
        run_page(app)
        active.wait_for_media()
        report['ui_checks'].append('camera_alarm_pause_release_resume_new_segment_and_stop')
        history = run_page(AppTest.from_file(str(next((ROOT / 'pages').glob('4_*.py'))), default_timeout=30))
        assert len(history.get('imgs')) == 1 and len(history.get('video')) == 1
        assert any('摄像头时间' in item.value for item in history.caption)
        report['ui_checks'].append('camera_history_snapshot_video_and_observation_time_label')
        next(widget for widget in app.radio if widget.label == '视频来源').set_value('内置样例')
        run_page(app)
        next(widget for widget in app.button if widget.label == '开始检测').click()
        run_page(app)
        file_session = app.session_state['video_detection_session']
        assert file_session.input_type == 'video' and file_session.latest.media_seconds == 0
        next(widget for widget in app.button if widget.label == '停止检测').click()
        run_page(app)
        file_session.wait_for_media()
        report['ui_checks'].append('switch_back_to_file_preserves_original_timestamp_semantics')
    report['status'] = 'passed' if not args.skip_hardware else 'passed_without_hardware'
    (ROOT / 'reports' / 'camera_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'unit_tests': result.testsRun,
                      'file_regression_frames': sum(row['frames'] for row in report['file_regression']),
                      'ui_checks': len(report['ui_checks'])}), flush=True)


if __name__ == '__main__':
    main()
