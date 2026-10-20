"""Controlled live frames and camera transport processes; no personal images."""

import json
import math
import subprocess
import sys
import time
import unittest
import uuid
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from alarm import AlarmConfig
from alarm_log import AlarmLog, alarm_media_root
from camera_capture import HEADER, FRAME
from camera_source import CameraConfig, CameraReadError, CameraSource, CapturedFrame
from camera_session import CameraSession
from recording import RecordingConfig


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'artifacts' / 'camera_checks' / ('unit_' + uuid.uuid4().hex)
ORIGINAL_POPEN = subprocess.Popen


class FakeCamera:
    def __init__(self, config=None, stamp=100.0, bgr=None):
        self.info = {'fps': 10.0, 'backend': 'controlled'}
        self.error = None
        self.opened = True
        self.latest = CapturedFrame(1, stamp, np.zeros((48, 64, 3), np.uint8) if bgr is None else bgr)

    def emit(self, delta=0.1, sequence_jump=1):
        self.latest = CapturedFrame(self.latest.sequence + sequence_jump, self.latest.captured_at + delta, self.latest.bgr)

    def read_latest(self, after_sequence):
        if self.error:
            raise CameraReadError(self.error)
        if not self.opened:
            raise CameraReadError('Controlled source closed')
        return self.latest if self.latest.sequence > after_sequence else None

    def isOpened(self):
        return self.opened

    def release(self):
        self.opened = False

    def metadata(self):
        return {'input_type': 'camera', 'clock': 'controlled_monotonic', 'test_source': True}


class Detector:
    device = 'controlled'

    def __init__(self, label='fire', fail_at=None):
        self.label, self.fail_at, self.calls = label, fail_at, 0

    def predict(self, bgr, conf, iou):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError('Controlled inference failure')
        boxes = [{'class': self.label, 'confidence': 0.9, 'xyxy': [0, 0, 30, 30]}]
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), boxes, {'fire': int(self.label == 'fire'), 'smoke': int(self.label == 'smoke')}, 0.0


class CameraSessionTests(unittest.TestCase):
    def setUp(self):
        self.root = OUTPUT / uuid.uuid4().hex
        self.root.mkdir(parents=True)
        self.cfg = replace(AlarmConfig(), confirm_frames=2, attention_seconds=0.1, warning_seconds=0.3,
                           alarm_seconds=0.5, recovery_seconds=0.2)
        self.recording = replace(RecordingConfig(), pre_seconds=0.2, post_seconds=2, output_fps=10, min_free_mb=0)
        self.session = None
        self.sources = []

    def factory(self, config):
        source = FakeCamera(config, stamp=100 + 100 * len(self.sources))
        self.sources.append(source)
        return source

    def create(self, detector=None, **kwargs):
        self.session = CameraSession(CameraConfig(), detector or Detector(), alarm_config=self.cfg,
                                     log_path=self.root / 'events.sqlite3', recording_config=self.recording,
                                     source_factory=self.factory, **kwargs)
        return self.session

    def feed(self, count, jump=1):
        for _ in range(count):
            self.session.step()
            self.session.capture.emit(sequence_jump=jump)

    def tearDown(self):
        if self.session:
            self.session.stop()
            self.session.wait_for_media()

    def test_defaults_and_invalid_camera_settings(self):
        self.assertEqual(CameraConfig.load(ROOT / 'configs' / 'camera.json'), CameraConfig())
        for overrides in ({'index': -1}, {'index': True}, {'backend': 'invalid'}, {'width': 0},
                          {'fps': 0}, {'read_timeout_seconds': math.nan}, {'idle_timeout_seconds': 1}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                replace(CameraConfig(), **overrides)

    def test_live_pipeline_reaches_alarm_and_saves_snapshot_and_clip(self):
        session = self.create()
        self.feed(11)
        self.assertEqual(session.alarms.snapshot.level, 3)
        session.stop()
        session.wait_for_media()
        store = AlarmLog(self.root / 'events.sqlite3')
        event = store.events()[0]
        self.assertEqual((event['peak_level'], event['last_level'], event['end_reason']), (3, 3, 'stopped'))
        self.assertEqual(json.loads(event['metadata_json'])['input_type'], 'camera')
        media = store.media(event['id'])
        self.assertEqual(media['status'], 'partial', media)
        self.assertEqual(media['details']['input_type'], 'camera')
        self.assertTrue((alarm_media_root(store.path) / media['snapshot_path']).is_file())
        capture = cv2.VideoCapture(str(alarm_media_root(store.path) / media['clip_path']))
        self.assertTrue(capture.read()[0])
        capture.release()
        self.assertFalse(session.is_open)

    def test_smoke_can_trigger_without_fire(self):
        self.create(Detector('smoke'))
        self.feed(7)
        self.assertEqual((self.session.alarms.snapshot.level, self.session.alarms.snapshot.classes), (3, ('smoke',)))

    def test_repeated_polls_do_not_count_the_same_capture_twice(self):
        self.create()
        self.session.step()
        for _ in range(20):
            self.assertIsNone(self.session.step())
        self.assertEqual((self.session.processed_frames, self.session.alarms.event_count), (1, 0))

    def test_skipped_capture_frames_use_real_timestamps(self):
        self.create()
        self.feed(7, jump=5)
        self.assertEqual(self.session.processed_frames, 7)
        self.assertEqual(self.session.dropped_frames, 24)
        self.assertAlmostEqual(self.session.latest.media_seconds, 0.6)
        self.assertEqual(self.session.alarms.snapshot.level, 3)

    def test_long_unobserved_gap_is_not_recovery(self):
        session = self.create()
        self.feed(7)
        session.capture.emit(delta=2)
        self.assertIsNone(session.step())
        session.wait_for_media()
        self.assertEqual((session.status, session.alarms.snapshot.level), ('disconnected', 3))
        event = session.alarms.store.events()[0]
        self.assertEqual(event['end_reason'], 'disconnected')
        self.assertNotIn(0, [row['to_level'] for row in session.alarms.store.transitions(event['id'])])
        self.assertTrue(session.alarms.store.media(event['id'])['details']['post_truncated'])

    def test_disconnect_closes_event_even_without_another_ui_poll(self):
        session = self.create()
        self.feed(7)
        session.capture.error = 'Controlled unplug'
        deadline = time.monotonic() + 2
        while session.status == 'running' and time.monotonic() < deadline:
            time.sleep(0.02)
        session.wait_for_media()
        self.assertEqual((session.status, session.alarms.snapshot.level), ('disconnected', 3))
        self.assertFalse(session.is_open)
        self.assertEqual(session.alarms.store.events()[0]['end_reason'], 'disconnected')

    def test_pause_closes_device_and_resume_starts_new_observation(self):
        session = self.create()
        self.feed(7)
        old_run, source = session.alarms.run_id, session.capture
        session.pause()
        self.assertFalse(source.isOpened())
        self.assertIsNone(session.step())
        self.assertEqual(session.alarms.snapshot.level, 3)
        session.resume()
        self.assertNotEqual(session.alarms.run_id, old_run)
        self.assertEqual((session.alarms.snapshot.level, session.segment_number), (0, 2))
        self.assertEqual(session.step().media_seconds, 0)
        session.stop()
        session.wait_for_media()
        event = session.alarms.store.events()[0]
        self.assertEqual((event['end_reason'], event['last_level']), ('paused', 3))

    def test_failed_reopen_keeps_paused_state_and_last_alarm(self):
        session = self.create()
        self.feed(7)
        session.pause()
        with patch.object(session, '_source_factory', side_effect=ValueError('Controlled open failure')):
            with self.assertRaises(ValueError):
                session.resume()
        self.assertEqual((session.status, session.alarms.snapshot.level, session.segment_number), ('paused', 3, 1))
        self.assertFalse(session.is_open)

    def test_inference_failure_releases_device_without_recovery(self):
        session = self.create(Detector(fail_at=7))
        self.feed(6)
        with self.assertRaises(RuntimeError):
            session.step()
        session.wait_for_media()
        self.assertEqual((session.status, session.alarms.snapshot.level), ('error', 3))
        self.assertFalse(session.is_open)
        self.assertEqual(session.alarms.store.events()[0]['end_reason'], 'error')

    def test_reversed_camera_clock_is_not_accepted(self):
        session = self.create()
        self.feed(3)
        session.capture.emit(delta=-2)
        session.step()
        self.assertEqual(session.status, 'disconnected')


class CameraTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OUTPUT.mkdir(parents=True, exist_ok=True)
        cls.worker = OUTPUT / 'controlled_worker.py'
        ok, jpeg = cv2.imencode('.jpg', np.zeros((48, 64, 3), np.uint8))
        assert ok
        cls.worker.write_text('''import sys,time,struct,json
header=struct.Struct('!cI'); frame=struct.Struct('!Qd')
def send(kind, body):
    sys.stdout.buffer.write(header.pack(kind,len(body))+body); sys.stdout.buffer.flush()
mode=sys.argv[1]
if mode=='silent': time.sleep(60)
elif mode=='oversize':
    sys.stdout.buffer.write(header.pack(b'F', 17*1024**2));sys.stdout.buffer.flush();time.sleep(60)
else:
    send(b'I',json.dumps({'fps':50,'backend':'controlled'}).encode())
    jpeg=''' + repr(jpeg.tobytes()) + '''
    for index in range(1000):
        send(b'F',frame.pack(index+1,time.monotonic())+jpeg)
        if mode=='stall':time.sleep(60)
        time.sleep(0.02)
''', encoding='utf-8')

    def launch(self, mode):
        return patch('camera_source.subprocess.Popen', side_effect=lambda command, **kwargs:
                     ORIGINAL_POPEN([sys.executable, '-u', str(self.worker), mode], **kwargs))

    def config(self, **overrides):
        return replace(CameraConfig(), open_timeout_seconds=1, read_timeout_seconds=0.3, idle_timeout_seconds=0.7, **overrides)

    def test_latest_frame_buffer_and_lease_release(self):
        with self.launch('stream'):
            source = CameraSource(self.config())
            try:
                first = source.read_latest(0)
                with self.assertRaisesRegex(ValueError, '其他会话'):
                    CameraSource(self.config())
                time.sleep(0.15)
                latest = source.read_latest(first.sequence)
                self.assertGreater(latest.sequence, first.sequence + 2)
                self.assertEqual(latest.bgr.shape, (48, 64, 3))
            finally:
                source.release()
            self.assertFalse(source.isOpened())
            reopened = CameraSource(self.config())
            reopened.release()

    def test_stalled_native_reader_is_killed_and_lease_released(self):
        with self.launch('stall'):
            source = CameraSource(self.config())
            self.assertTrue(source._done.wait(2))
            self.assertIn('超时', source.error)
            self.assertFalse(source.isOpened())
            self.assertNotIn(source.config.index, CameraSource._leases)
            source.release()

    def test_idle_page_closes_device_without_polls(self):
        with self.launch('stream'):
            source = CameraSource(self.config())
            self.assertTrue(source._done.wait(2))
            self.assertIn('未刷新', source.error)
            self.assertFalse(source.isOpened())

    def test_startup_timeout_releases_lease(self):
        with self.launch('silent'):
            with self.assertRaisesRegex(ValueError, '首次取帧超时'):
                CameraSource(self.config())
        self.assertNotIn(0, CameraSource._leases)

    def test_oversized_packet_rejected_without_allocation(self):
        with self.launch('oversize'):
            with self.assertRaisesRegex(ValueError, '数据包超过上限'):
                CameraSource(self.config())
        self.assertNotIn(0, CameraSource._leases)


if __name__ == '__main__':
    unittest.main()
