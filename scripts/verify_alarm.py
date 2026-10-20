"""Verify temporal rules, real video integration, persistence and alarm UI."""

from __future__ import annotations

import json
import os
import sys
import unittest
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from streamlit.testing.v1 import AppTest

from alarm import AlarmConfig
from alarm_log import AlarmLog
from video_detection import FireSmokeDetector, VideoSession


def button(app, label):
    return next(widget for widget in app.button if widget.label == label)


def run_page(app):
    app.run()
    assert not app.exception, [item.message for item in app.exception]
    return app


class ControlledDetector:
    """Simulated outputs for lifecycle tests, not for real-world accuracy."""
    device = 'controlled_test'

    def __init__(self, fail_at=None):
        self.calls = 0
        self.fail_at = fail_at

    def predict(self, bgr, conf, iou):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError('Intentional detector failure')
        return np.zeros((8, 8, 3), dtype=np.uint8), [{'class': 'fire', 'confidence': 0.9, 'xyxy': [0, 0, 50, 50]}], {'fire': 1, 'smoke': 0}, 0.0


def main():
    output = ROOT / 'artifacts' / 'alarm_checks' / ('integration_' + uuid.uuid4().hex)
    output.mkdir(parents=True, exist_ok=True)
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_alarm.py')
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    cfg = AlarmConfig.load(ROOT / 'configs' / 'alarm.json')
    report = {'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
              'scope': 'Functional tests; controlled outputs demonstrate suppression, not measured real-world accuracy.',
              'config': asdict(cfg), 'unit_tests_passed': result.testsRun,
              'controlled_spikes': {'input_positive_frames': 50, 'confirmed_events': 0, 'source': 'simulated detector outputs'},
              'full_videos': [], 'lifecycle': [], 'ui_checks': []}
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    cv2.setNumThreads(2)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    report['device'] = device
    detector = FireSmokeDetector(ROOT / 'fire-models' / 'fire_n.pt', device)
    database = output / 'videos.sqlite3'
    for source in sorted((ROOT / 'demo-videos').glob('*')):
        session = VideoSession(source, detector, alarm_config=cfg, log_path=database)
        raw_positive, temporal_positive, raw_episodes, prior_positive = 0, 0, 0, False
        while session.status == 'running':
            frame = session.step()
            if frame is None:
                break
            positive = any(box['confidence'] >= (cfg.fire_confidence if box['class'] == 'fire' else cfg.smoke_confidence) for box in frame.detections)
            raw_positive += positive
            raw_episodes += positive and not prior_positive
            prior_positive = positive
            temporal_positive += session.alarms.snapshot.level > 0
        assert session.status == 'completed' and not session.is_open, session.error
        assert session.processed_frames == session.info.frame_count
        events = AlarmLog(database).events(source=source.name)
        assert len(events) == session.alarms.event_count
        assert all(event['end_reason'] in {'recovered', 'completed'} for event in events)
        assert not session.alarms.log_error
        report['full_videos'].append(dict(source=source.name, frames=session.processed_frames,
                                         raw_positive_frames=raw_positive, raw_positive_episodes=raw_episodes,
                                         confirmed_state_frames=temporal_positive, events=len(events),
                                         peak_level=max((event['peak_level'] for event in events), default=0),
                                         transitions=sum(len(AlarmLog(database).transitions(event['id'])) for event in events)))
        print(f'{source.name}: {session.processed_frames} frames, {len(events)} events', flush=True)

    controlled_db = output / 'lifecycle.sqlite3'
    for fail_at in (None, 163):
        session = VideoSession(ROOT / 'demo-videos' / 'fire1.mov', ControlledDetector(fail_at), log_path=controlled_db,
                               source_name='controlled_stop' if fail_at is None else 'controlled_error')
        for _ in range(161):
            session.step()
        assert session.alarms.snapshot.level == 3
        session.pause()
        before = (session.processed_frames, session.alarms.snapshot, len(session.alarms.recent_transitions))
        assert session.step() is None
        assert before == (session.processed_frames, session.alarms.snapshot, len(session.alarms.recent_transitions))
        session.resume()
        session.step()
        if fail_at:
            try:
                session.step()
            except RuntimeError:
                pass
            else:
                raise AssertionError('Intentional inference error did not propagate')
            assert session.status == 'error'
        else:
            session.stop()
        assert not session.is_open
        assert session.alarms.snapshot.level == 3
        assert session.step() is None
        event = AlarmLog(controlled_db).events(source=session.alarms.source)[0]
        assert event['last_level'] == 3 and event['end_reason'] == session.status
        report['lifecycle'].append({'reason': session.status, 'pause_freezes_state': True, 'last_level_retained': 3, 'released': True})

    # Isolate all UI-test data from the user's actual alarm history.
    ui_database = output / 'ui.sqlite3'
    with patch.dict(os.environ, {'FIRE_ALARM_LOG_PATH': str(ui_database)}):
        app = run_page(AppTest.from_file(str(next((ROOT / 'pages').glob('3_*.py'))), default_timeout=120))
        for label, value in (('关注持续时间（秒）', 0.0), ('预警持续时间（秒）', 0.1), ('报警持续时间（秒）', 0.2)):
            next(widget for widget in app.number_input if widget.label == label).set_value(value)
        run_page(app)
        button(app, '开始检测').click()
        run_page(app)
        active = app.session_state['video_detection_session']
        for _ in range(20):
            if active.alarms.snapshot.level == 3:
                break
            run_page(app)
        assert active.alarms.snapshot.level == 3
        assert any('当前报警等级：报警' in item.value for item in app.error)
        report['ui_checks'].append('real_model_alarm_banner')
        button(app, '暂停检测').click()
        run_page(app)
        before = (active.processed_frames, len(active.alarms.recent_transitions))
        run_page(app)
        assert before == (active.processed_frames, len(active.alarms.recent_transitions))
        report['ui_checks'].append('paused_alarm_and_log_stable')
        button(app, '继续检测').click()
        run_page(app)
        assert active.processed_frames > before[0]
        button(app, '停止检测').click()
        run_page(app)
        assert active.alarms.snapshot.level == 3 and not active.is_open
        first_run = active.alarms.run_id
        next(widget for widget in app.selectbox if widget.label == '选择样例视频').select('fire3.mp4')
        run_page(app)
        button(app, '开始检测').click()
        run_page(app)
        other = app.session_state['video_detection_session']
        assert other.alarms.run_id != first_run and other.alarms.snapshot.level == 0
        assert other.alarms.event_count == 0
        button(app, '停止检测').click()
        run_page(app)
        report['ui_checks'].append('source_switch_resets_evidence_preserves_history')
        next(widget for widget in app.number_input if widget.label == '报警持续时间（秒）').set_value(0.1)
        run_page(app)
        assert button(app, '开始检测').disabled
        report['ui_checks'].append('invalid_rule_disables_start')
        history = run_page(AppTest.from_file(str(next((ROOT / 'pages').glob('4_*.py'))), default_timeout=30))
        assert not history.error
        next(widget for widget in history.selectbox if widget.label == '最高等级筛选').select('报警')
        run_page(history)
        assert len(history.dataframe[0].value) == 1
        assert AlarmLog(ui_database).events()[0]['end_reason'] == 'stopped'
        report['ui_checks'].append('persistent_history_filter_and_transition_display')
    report['status'] = 'passed'
    (ROOT / 'reports' / 'alarm_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'passed', 'unit_tests': result.testsRun,
                      'full_video_frames': sum(row['frames'] for row in report['full_videos']),
                      'ui_checks': len(report['ui_checks'])}), flush=True)


if __name__ == '__main__':
    main()
