"""Real event media, controlled boundary cases, codecs and history UI checks."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import unittest
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import torch
from streamlit.testing.v1 import AppTest

from alarm import AlarmConfig
from alarm_log import AlarmLog, alarm_media_root
from recording import RecordingConfig, resolve_media_asset
from video_detection import FireSmokeDetector, VideoSession


class ControlledDetector:
    device = 'controlled_test'

    def __init__(self, from_frame=1, fail_at=None):
        self.calls = 0
        self.from_frame, self.fail_at = from_frame, fail_at

    def predict(self, bgr, conf, iou):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError('Intentional test failure')
        rgb = cv2.cvtColor(cv2.resize(bgr, (320, 240)), cv2.COLOR_BGR2RGB)
        positive = self.calls >= self.from_frame
        boxes = [{'class': 'fire', 'confidence': 0.9, 'xyxy': [0, 0, 50, 50]}] if positive else []
        return rgb, boxes, {'fire': int(positive), 'smoke': 0}, 0.0


def inspect_media(store, event, root):
    media = store.media(event['id'])
    assert media is not None and media['status'] in {'ready', 'partial'}, media
    snapshot = resolve_media_asset(root, media['snapshot_path'], '.jpg')
    video = resolve_media_asset(root, media['clip_path'], '.mp4')
    assert snapshot is not None and video is not None
    assert cv2.imread(str(snapshot)) is not None
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                                               '-show_entries', 'stream=codec_name,pix_fmt,nb_frames,duration,width,height',
                                               '-of', 'json', str(video)]))['streams'][0]
    assert probe['codec_name'] == 'h264' and probe['pix_fmt'] == 'yuv420p', probe
    details = media['details']
    assert abs(float(probe['duration']) - details['duration_seconds']) <= 1 / details['config']['output_fps'] + 0.01
    assert details['start_frame'] <= details['trigger_frame'] <= details['end_frame']
    assert details['start_seconds'] <= details['trigger_seconds'] < details['end_seconds']
    return dict(status=media['status'], trigger_seconds=details['trigger_seconds'], start_seconds=details['start_seconds'],
                end_seconds=details['end_seconds'], encoded_duration=float(probe['duration']), encoded_frames=int(probe['nb_frames']),
                pre_truncated=details['pre_truncated'], post_truncated=details['post_truncated'],
                completion_reason=details['completion_reason'], codec=probe['codec_name'],
                snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                clip_sha256=hashlib.sha256(video.read_bytes()).hexdigest())


def run_page(app):
    app.run()
    assert not app.exception, [item.message for item in app.exception]
    return app


def main():
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        raise SystemExit('This verification requires FFmpeg with libx264 and ffprobe on PATH.')
    output = ROOT / 'artifacts' / 'recording_checks' / ('integration_' + uuid.uuid4().hex)
    output.mkdir(parents=True, exist_ok=True)
    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_*.py'))
    assert result.wasSuccessful() and not result.skipped
    cfg = RecordingConfig.load(ROOT / 'configs' / 'recording.json')
    alarm_cfg = AlarmConfig.load(ROOT / 'configs' / 'alarm.json')
    report = dict(generated_at=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
                  scope='Functional recording verification; controlled outputs are not real-world accuracy data.',
                  unit_tests_passed=result.testsRun, recording_config=asdict(cfg), real_videos=[], controlled_cases=[], ui_checks=[])
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    cv2.setNumThreads(2)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    report['device'] = device
    report['ffmpeg'] = subprocess.check_output(['ffmpeg', '-version'], creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0).decode().splitlines()[0]
    detector = FireSmokeDetector(ROOT / 'fire-models' / 'fire_n.pt', device)
    database = output / 'real.sqlite3'
    for source in sorted((ROOT / 'demo-videos').glob('*')):
        session = VideoSession(source, detector, alarm_config=alarm_cfg, log_path=database, recording_config=cfg)
        while session.status == 'running':
            session.step()
        session.wait_for_media()
        assert session.status == 'completed' and not session.is_open
        assert session.processed_frames == session.info.frame_count
        assert not session.recorder.last_error and not session.alarms.log_error
        store = AlarmLog(database)
        events = store.events(source=source.name)
        assets = [inspect_media(store, event, alarm_media_root(database)) for event in events]
        assert len(events) == session.alarms.event_count
        assert session.recorder.max_buffer_bytes_seen <= cfg.max_buffer_mb * 1024**2
        assert session.recorder.max_buffer_frames_seen <= cfg.max_buffer_frames
        assert session.recorder.buffer_bytes == 0
        report['real_videos'].append(dict(source=source.name, frames=session.processed_frames, events=len(events), media=assets,
                                         peak_buffer_bytes=session.recorder.max_buffer_bytes_seen,
                                         peak_buffer_frames=session.recorder.max_buffer_frames_seen))
        print(f'{source.name}: {session.processed_frames} detected frames; {len(assets)} screenshots/clips verified', flush=True)

    # Full 5+10 second window and late onset on a real file with controlled outputs.
    controlled_database = output / 'controlled.sqlite3'
    for name, onset, stop_at, failure in (('full_5_plus_10', 211, None, None), ('near_eof', 925, None, None),
                                        ('manual_stop', 1, 40, None), ('detector_error', 1, None, 41)):
        session = VideoSession(ROOT / 'demo-videos' / 'fire2.mov', ControlledDetector(onset, failure), alarm_config=alarm_cfg,
                               log_path=controlled_database, source_name=name, recording_config=cfg)
        while session.status == 'running':
            try:
                session.step()
            except RuntimeError:
                assert failure is not None
            if session.processed_frames == 30 and session.status == 'running':
                session.pause()
                prior = session.recorder._last_input
                assert session.step() is None and session.recorder._last_input == prior
                session.resume()
            if stop_at and session.processed_frames >= stop_at:
                session.stop()
        session.wait_for_media()
        assert not session.is_open and not session.recorder.last_error
        store = AlarmLog(controlled_database)
        events = store.events(source=name)
        assert len(events) == 1
        media = inspect_media(store, events[0], alarm_media_root(controlled_database))
        if name == 'full_5_plus_10':
            assert media['status'] == 'ready' and abs(media['encoded_duration'] - 15) < 0.04
        else:
            assert media['post_truncated'] and media['status'] == 'partial'
        assert events[0]['end_reason'] == session.status
        report['controlled_cases'].append(dict(name=name, observation_end=session.status, media=media))
        print(f'{name}: {media["status"]}, {media["encoded_duration"]:.3f}s', flush=True)

    ui_database = output / 'ui.sqlite3'
    with patch.dict(os.environ, {'FIRE_ALARM_LOG_PATH': str(ui_database)}):
        app = run_page(AppTest.from_file(str(next((ROOT / 'pages').glob('3_*.py'))), default_timeout=120))
        for label, value in (('关注持续时间（秒）', 0.0), ('预警持续时间（秒）', 0.1), ('报警持续时间（秒）', 0.2),
                             ('确认前保存（秒）', 0.2), ('确认后保存（秒）', 0.3)):
            next(widget for widget in app.number_input if widget.label == label).set_value(value)
        run_page(app)
        next(widget for widget in app.button if widget.label == '开始检测').click()
        run_page(app)
        active = app.session_state['video_detection_session']
        for _ in range(15):
            run_page(app)
        next(widget for widget in app.button if widget.label == '停止检测').click()
        run_page(app)
        active.wait_for_media()
        assert not active.recorder.last_error
        event = AlarmLog(ui_database).events()[0]
        inspect_media(AlarmLog(ui_database), event, alarm_media_root(ui_database))
        report['ui_checks'].append('video_controls_create_media_for_confirmed_event')
        history = run_page(AppTest.from_file(str(next((ROOT / 'pages').glob('4_*.py'))), default_timeout=30))
        assert not history.error
        assert len(history.get('imgs')) == 1 and len(history.get('video')) == 1
        report['ui_checks'].append('history_renders_snapshot_and_video')
        media = AlarmLog(ui_database).media(event['id'])
        (alarm_media_root(ui_database) / media['clip_path']).rename(alarm_media_root(ui_database) / event['id'] / 'clip.missing-test.mp4')
        run_page(history)
        assert any('片段文件已不在' in item.value for item in history.warning)
        report['ui_checks'].append('missing_clip_shows_message_without_exception')
    report['status'] = 'passed'
    (ROOT / 'reports' / 'recording_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'passed', 'unit_tests': result.testsRun,
                      'real_model_video_frames': sum(row['frames'] for row in report['real_videos']),
                      'controlled_cases': len(report['controlled_cases']), 'ui_checks': len(report['ui_checks'])}), flush=True)


if __name__ == '__main__':
    main()
