"""Controlled model-output cases, not a labelled real-world accuracy dataset."""

import csv
import io
import json
import sqlite3
import unittest
import uuid
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from alarm import AlarmConfig, AlarmEngine
from alarm_log import AlarmLog, AlarmMonitor, events_csv


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'artifacts' / 'alarm_checks' / ('unit_' + uuid.uuid4().hex)


def box(label='fire', confidence=0.9, xyxy=None):
    return {'class': label, 'confidence': confidence, 'xyxy': xyxy or [0, 0, 50, 50]}


def feed(target, start, stop, detections, fps=10):
    return [target.update(index + 1, index / fps, detections) for index in range(start, stop)]


class AlarmRuleTests(unittest.TestCase):
    def test_config_validation(self):
        self.assertEqual(AlarmConfig.load(ROOT / 'configs' / 'alarm.json'), AlarmConfig())
        for overrides in ({'confirm_frames': 1}, {'confirm_frames': True}, {'alarm_seconds': 1},
                          {'warning_seconds': float('nan')}, {'fire_confidence': 0}, {'match_iou': 0},
                          {'recovery_seconds': 0}, {'max_frame_gap_seconds': -1}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                replace(AlarmConfig(), **overrides)

    def test_single_frame_spikes_never_confirm(self):
        engine = AlarmEngine()
        raw_positive_frames = 0
        for index in range(100):
            detections = [box()] if index % 2 == 0 else []
            raw_positive_frames += bool(detections)
            self.assertEqual(engine.update(index + 1, index / 10, detections).level, 0)
        self.assertEqual(raw_positive_frames, 50)

    def test_duplicate_boxes_are_not_multiple_frames(self):
        engine = AlarmEngine(replace(AlarmConfig(), attention_seconds=0))
        state = engine.update(1, 0, [box()] * 30)
        self.assertEqual((state.level, state.pending_frames), (0, 1))

    def test_low_confidence_and_invalid_geometry_ignored(self):
        engine = AlarmEngine()
        invalid = [box(confidence=0.49), box(confidence=float('nan')), box(xyxy=[10, 10, 1, 1]),
                   box(xyxy=[0, 0, float('inf'), 50]), box(label='person')]
        self.assertTrue(all(state.level == 0 for state in feed(engine, 0, 100, invalid)))

    def test_all_levels_and_exact_duration_boundaries(self):
        states = feed(AlarmEngine(), 0, 61, [box()])
        first = {level: next(index / 10 for index, state in enumerate(states) if state.level == level) for level in (1, 2, 3)}
        self.assertEqual(first, {1: 0.5, 2: 2.0, 3: 5.0})

    def test_smoke_independently_reaches_alarm(self):
        state = feed(AlarmEngine(), 0, 61, [box('smoke')])[-1]
        self.assertEqual((state.level, state.classes), (3, ('smoke',)))

    def test_threshold_is_inclusive(self):
        self.assertEqual(feed(AlarmEngine(), 0, 61, [box(confidence=0.5)])[-1].level, 3)

    def test_spatial_jumps_do_not_accumulate(self):
        engine = AlarmEngine()
        for index in range(100):
            xyxy = [0, 0, 50, 50] if index % 2 else [100, 100, 150, 150]
            self.assertEqual(engine.update(index + 1, index / 10, [box(xyxy=xyxy)]).level, 0)

    def test_classes_do_not_share_a_streak(self):
        engine = AlarmEngine()
        for index in range(100):
            self.assertEqual(engine.update(index + 1, index / 10, [box('fire' if index % 2 else 'smoke')]).level, 0)

    def test_stable_region_survives_distractors_and_box_order(self):
        engine = AlarmEngine()
        for index in range(61):
            detections = [box(xyxy=[index % 3, 0, 50 + index % 3, 50]),
                          box(confidence=0.99, xyxy=[200 + 100 * index, 0, 250 + 100 * index, 50])]
            state = engine.update(index + 1, index / 10, detections[::1 if index % 2 else -1])
        self.assertEqual(state.level, 3)

    def test_missing_before_confirmation_restarts_streak(self):
        engine = AlarmEngine()
        feed(engine, 0, 5, [box()])
        engine.update(6, 0.5, [])
        self.assertEqual(feed(engine, 6, 11, [box()])[-1].level, 0)
        self.assertEqual(engine.update(12, 1.1, [box()]).level, 1)

    def test_short_miss_holds_but_does_not_escalate(self):
        engine = AlarmEngine()
        feed(engine, 0, 20, [box()])  # last supporting time is 1.9 seconds
        state = engine.update(21, 2.0, [])
        self.assertEqual((state.level, state.pending_frames), (1, 0))
        self.assertEqual(engine.update(22, 2.1, [box()]).level, 1)

    def test_recovery_wait_and_downgrade(self):
        engine = AlarmEngine()
        feed(engine, 0, 61, [box()])
        self.assertEqual(feed(engine, 61, 80, [])[-1].level, 3)
        self.assertEqual(engine.update(81, 8.0, []).level, 0)
        engine = AlarmEngine()
        feed(engine, 0, 61, [box()])
        # Smoke starts while fire disappears; expiry lowers the existing event.
        state = feed(engine, 61, 81, [box('smoke')])[-1]
        self.assertEqual((state.level, state.previous_level, state.classes), (1, 3, ('smoke',)))

    def test_large_timestamp_gap_restarts_confirmation(self):
        engine = AlarmEngine()
        engine.update(1, 0, [box()])
        state = engine.update(2, 10, [box()])
        self.assertEqual((state.level, state.pending_frames), (0, 1))

    def test_frame_rate_and_processing_speed_do_not_change_source_thresholds(self):
        for fps in (4, 10, 30):
            engine = AlarmEngine()
            states = feed(engine, 0, fps * 6, [box()], fps)
            for level, expected in ((2, 2.0), (3, 5.0)):
                first = next(index / fps for index, state in enumerate(states) if state.level == level)
                self.assertLessEqual(abs(first - expected), 1 / fps)

    def test_repeated_or_reversed_input_is_rejected(self):
        engine = AlarmEngine()
        engine.update(1, 0, [])
        for number, seconds in ((1, 0.1), (3, 0.1), (2, 0), (2, float('nan'))):
            with self.subTest(number=number, seconds=seconds), self.assertRaises(ValueError):
                engine.update(number, seconds, [])
        self.assertEqual(engine.update(2, 0.1, []).level, 0)


class AlarmHistoryTests(unittest.TestCase):
    def setUp(self):
        OUTPUT.mkdir(parents=True, exist_ok=True)
        self.path = OUTPUT / (uuid.uuid4().hex + '.sqlite3')

    def monitor(self, source='test.mp4'):
        return AlarmMonitor(AlarmConfig(), source, self.path, metadata={'test': True})

    def test_one_event_for_sustained_alarm_persists_after_reopen(self):
        monitor = self.monitor()
        feed(monitor, 0, 61, [box()])
        feed(monitor, 61, 82, [])
        monitor.finish('completed')
        store = AlarmLog(self.path)
        events = store.events()
        self.assertEqual((len(events), monitor.event_count), (1, 1))
        self.assertEqual((events[0]['peak_level'], events[0]['last_level'], events[0]['end_reason']), (3, 0, 'recovered'))
        transitions = store.transitions(events[0]['id'])
        self.assertEqual([row['to_level'] for row in transitions], [1, 2, 3, 0])
        self.assertEqual(len({row['event_id'] for row in transitions}), 1)
        with sqlite3.connect(self.path) as connection:
            config = json.loads(connection.execute('SELECT config_json FROM runs').fetchone()[0])
        self.assertEqual(config['confirm_frames'], 3)

    def test_stop_eof_and_error_are_not_recovery(self):
        for reason in ('stopped', 'completed', 'error'):
            with self.subTest(reason=reason):
                monitor = self.monitor(reason + '.mp4')
                feed(monitor, 0, 61, [box('smoke')])
                monitor.finish(reason)
                monitor.finish(reason)
                event = AlarmLog(self.path).events(source=reason + '.mp4')[0]
                self.assertEqual((event['last_level'], event['end_reason']), (3, reason))
                self.assertEqual([row['to_level'] for row in AlarmLog(self.path).transitions(event['id'])], [1, 2, 3, 3])
                with self.assertRaises(RuntimeError):
                    monitor.update(62, 6.1, [])

    def test_filters_csv_and_new_session_are_independent(self):
        for name, frames in (('first.mp4', 10), ('=second.mp4', 61)):
            monitor = self.monitor(name)
            feed(monitor, 0, frames, [box()])
            monitor.finish('stopped')
            self.assertEqual(monitor.event_count, 1)
        store = AlarmLog(self.path)
        self.assertEqual(len(store.events()), 2)
        self.assertEqual(len(store.events(peak_level=3)), 1)
        self.assertEqual(len(store.events(source='first.mp4', peak_level=3)), 0)
        exported = list(csv.DictReader(io.StringIO(events_csv(store.events(peak_level=3)).decode('utf-8-sig'))))
        self.assertEqual((len(exported), exported[0]['视频'], exported[0]['最高等级']), (1, "'=second.mp4", '报警'))
        self.assertEqual(len({event['run_id'] for event in store.events()}), 2)

    def test_recurrence_after_recovery_creates_new_event(self):
        monitor = self.monitor()
        feed(monitor, 0, 10, [box()])
        feed(monitor, 10, 31, [])
        feed(monitor, 31, 41, [box()])
        monitor.finish('completed')
        self.assertEqual(monitor.event_count, 2)
        self.assertEqual(len(AlarmLog(self.path).events()), 2)

    def test_single_frame_spikes_create_no_log_events(self):
        monitor = self.monitor()
        for index in range(100):
            monitor.update(index + 1, index / 10, [box()] if index % 2 else [])
        monitor.finish('completed')
        self.assertEqual((monitor.event_count, len(AlarmLog(self.path).events())), (0, 0))

    def test_logging_failure_is_visible_and_detection_continues(self):
        monitor = self.monitor()
        with patch.object(monitor.store, 'save_event', side_effect=sqlite3.OperationalError('disk full')):
            state = feed(monitor, 0, 61, [box()])[-1]
            self.assertEqual(state.level, 3)
            self.assertIn('写入失败', monitor.log_error)
        monitor.finish('stopped')

    def test_unwritable_database_fails_clearly_at_start(self):
        self.path.mkdir()
        with self.assertRaisesRegex(RuntimeError, '无法创建报警日志'):
            self.monitor()


if __name__ == '__main__':
    unittest.main()
