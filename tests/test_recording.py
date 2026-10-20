"""Real H.264 encoding with controlled frames and isolated event databases."""

import json
import shutil
import sqlite3
import subprocess
import unittest
import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from alarm import AlarmConfig
from alarm_log import AlarmLog, AlarmMonitor
from recording import EventRecorder, RecordingConfig, resolve_media_asset


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'artifacts' / 'recording_checks' / ('unit_' + uuid.uuid4().hex)


def detection():
    return [{'class': 'fire', 'confidence': 0.9, 'xyxy': [0, 0, 50, 50]}]


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.root = OUTPUT / uuid.uuid4().hex
        self.root.mkdir(parents=True)
        self.monitor = AlarmMonitor(AlarmConfig(), 'controlled.mp4', self.root / 'events.sqlite3')
        self.cfg = replace(RecordingConfig(), pre_seconds=1, post_seconds=1, output_fps=10, min_free_mb=0)
        self.recorder = None

    def tearDown(self):
        if self.recorder:
            self.recorder.finish('stopped')
            self.recorder.wait()

    def recorder_for(self, **overrides):
        self.recorder = EventRecorder(replace(self.cfg, **overrides), self.root / 'media', self.monitor.store, 10)
        return self.recorder

    def frame(self, index, seconds=None, positive=True, rgb=None):
        seconds = index / 10 if seconds is None else seconds
        self.monitor.update(index + 1, seconds, detection() if positive else [])
        rgb = np.full((32, 48, 3), [min(index * 5, 235), 10, 20], dtype=np.uint8) if rgb is None else rgb
        frame = SimpleNamespace(number=index + 1, media_seconds=seconds, rgb=rgb, time_estimated=False)
        event = self.monitor.current_event
        self.recorder.add_frame(frame, event['id'] if event else None)

    def finish(self, reason='completed'):
        self.monitor.finish(reason)
        self.recorder.finish(reason)
        self.recorder.wait()
        return [self.monitor.store.media(event['id']) for event in self.monitor.store.events()]

    def test_config_validation(self):
        self.assertEqual(RecordingConfig.load(ROOT / 'configs' / 'recording.json'), RecordingConfig())
        for overrides in ({'pre_seconds': -1}, {'post_seconds': 0}, {'pre_seconds': 30, 'post_seconds': 40},
                          {'post_seconds': float('nan')}, {'enabled': 1}, {'output_fps': 0},
                          {'max_buffer_mb': 0}, {'max_pending_clips': 9}, {'jpeg_quality': True}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                replace(self.cfg, **overrides)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg and ffprobe required')
    def test_full_window_h264_snapshot_frame_order_and_duration(self):
        recorder = self.recorder_for()
        for index in range(40):
            self.frame(index, positive=index >= 10)
        media = self.finish()[0]
        self.assertEqual(media['status'], 'ready', media)
        details = media['details']
        self.assertEqual((details['trigger_frame'], details['start_frame'], details['end_frame']), (16, 6, 25))
        self.assertAlmostEqual(details['duration_seconds'], 2)
        self.assertFalse(details['pre_truncated'] or details['post_truncated'])
        image = Image.open(recorder.root / media['snapshot_path'])
        self.assertLess(abs(np.asarray(image)[0, 0, 0].item() - 75), 5)
        video = recorder.root / media['clip_path']
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                                                   '-show_entries', 'stream=codec_name,pix_fmt,nb_frames,duration', '-of', 'json', str(video)]))['streams'][0]
        self.assertEqual((probe['codec_name'], probe['pix_fmt'], int(probe['nb_frames'])), ('h264', 'yuv420p', 20))
        self.assertAlmostEqual(float(probe['duration']), 2, places=2)
        capture = cv2.VideoCapture(str(video))
        colors = []
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            colors.append(int(bgr[0, 0, 2]))
        capture.release()
        self.assertEqual(len(colors), 20)
        self.assertLess(abs(colors[0] - 25), 8)
        self.assertLess(abs(colors[-1] - 120), 8)
        self.assertTrue(all(right >= left - 3 for left, right in zip(colors, colors[1:])))
        self.assertFalse(any(recorder.root.rglob('frames.ffconcat')))
        self.assertFalse(any(recorder.root.rglob('frames')))
        self.assertEqual(len(list(recorder.root.rglob('snapshot.jpg'))), 1)

    def test_start_end_and_stop_partial_windows(self):
        for reason in ('completed', 'stopped', 'error'):
            with self.subTest(reason=reason):
                # Each isolated recorder uses a separate monitor and directory.
                self.monitor = AlarmMonitor(AlarmConfig(), reason, self.root / (reason + '.sqlite3'))
                self.recorder = EventRecorder(self.cfg, self.root / reason, self.monitor.store, 10)
                for index in range(10):
                    self.frame(index)
                media = self.finish(reason)[0]
                self.assertEqual(media['status'], 'partial', media)
                self.assertTrue(media['details']['pre_truncated'] and media['details']['post_truncated'])
                self.assertEqual(media['details']['completion_reason'], reason)
                self.assertAlmostEqual(media['details']['end_seconds'], 1)
                self.assertEqual(self.recorder.buffer_bytes, 0)

    def test_variable_timestamps_preserve_media_duration(self):
        recorder = self.recorder_for(pre_seconds=0.4, post_seconds=0.8)
        seconds = 0
        for index in range(40):
            self.frame(index, seconds=seconds)
            seconds += (0.06, 0.12, 0.04, 0.09)[index % 4]
        media = self.finish()[0]
        self.assertEqual(media['status'], 'ready', media)
        details = media['details']
        capture = cv2.VideoCapture(str(recorder.root / media['clip_path']))
        duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / capture.get(cv2.CAP_PROP_FPS)
        capture.release()
        self.assertLessEqual(abs(duration - details['duration_seconds']), 0.101)
        self.assertAlmostEqual(details['end_seconds'], details['target_end_seconds'])

    def test_bounded_buffer_and_no_event_no_files(self):
        recorder = self.recorder_for(max_buffer_mb=0.02, max_buffer_frames=4)
        random = np.random.default_rng(4)
        for index in range(200):
            self.frame(index, positive=False, rgb=random.integers(0, 256, (96, 128, 3), dtype=np.uint8))
            self.assertLessEqual(len(recorder.buffer), 4)
            self.assertLessEqual(recorder.buffer_bytes, int(0.02 * 1024**2))
        self.assertEqual(self.finish(), [])
        self.assertFalse(recorder.root.exists())

    def test_buffer_limit_is_reported_as_short_pre_window(self):
        self.recorder_for(max_buffer_frames=2)
        for index in range(40):
            self.frame(index, positive=index >= 15)
        media = self.finish()[0]
        self.assertEqual(media['status'], 'partial', media)
        self.assertTrue(media['details']['pre_truncated'])
        self.assertFalse(media['details']['post_truncated'])

    def test_overlapping_events_are_separate_and_capacity_keeps_screenshot(self):
        self.monitor = AlarmMonitor(replace(AlarmConfig(), confirm_frames=2, attention_seconds=0, recovery_seconds=0.1), 'overlap', self.root / 'overlap.sqlite3')
        self.recorder_for(max_pending_clips=1)
        for index in range(25):
            self.frame(index, positive=index in {0, 1, 5, 6})
        media = self.finish()
        self.assertEqual(len(media), 2)
        self.assertEqual({row['status'] for row in media}, {'partial', 'failed'})
        failed = next(row for row in media if row['status'] == 'failed')
        self.assertIn('上限', failed['error'])
        self.assertTrue((self.recorder.root / failed['snapshot_path']).exists())

    def test_missing_encoder_keeps_snapshot_and_reports_error(self):
        recorder = self.recorder_for(ffmpeg_binary='no-such-ffmpeg-for-test')
        for index in range(10):
            self.frame(index)
        media = self.finish()[0]
        self.assertEqual(media['status'], 'failed')
        self.assertTrue((recorder.root / media['snapshot_path']).exists())
        self.assertIsNone(media['clip_path'])
        self.assertIn('FFmpeg', media['error'])

    def test_encoder_failure_is_not_a_ready_clip(self):
        import sys
        self.recorder_for(ffmpeg_binary=sys.executable)
        for index in range(10):
            self.frame(index)
        media = self.finish()[0]
        self.assertEqual(media['status'], 'failed')
        self.assertIsNone(media['clip_path'])
        self.assertTrue(media['snapshot_path'])

    def test_low_disk_space_does_not_stop_alarm(self):
        self.recorder_for(min_free_mb=1)
        with patch('recording.shutil.disk_usage', return_value=SimpleNamespace(free=0)):
            for index in range(61):
                self.frame(index)
        media = self.finish()[0]
        self.assertEqual(media['status'], 'failed')
        self.assertIn('空间', media['error'])
        self.assertEqual(self.monitor.snapshot.level, 3)

    def test_disabled_recording_allocates_no_buffer_or_media(self):
        recorder = self.recorder_for(enabled=False)
        for index in range(10):
            self.frame(index)
        self.finish()
        self.assertEqual((len(recorder.buffer), recorder.buffer_bytes), (0, 0))
        self.assertIsNone(self.monitor.store.media(self.monitor.store.events()[0]['id']))

    def test_legacy_database_upgrade_retains_events(self):
        for index in range(10):
            self.monitor.update(index + 1, index / 10, detection())
        self.monitor.finish('stopped')
        with sqlite3.connect(self.monitor.store.path) as connection:
            connection.execute('DROP TABLE event_media')  # isolated legacy-schema fixture
        reopened = AlarmLog(self.monitor.store.path)
        event = reopened.events()[0]
        self.assertEqual(event['end_reason'], 'stopped')
        self.assertIsNone(reopened.media(event['id']))

    def test_asset_paths_stay_inside_media_directory(self):
        root = self.root / 'assets'
        root.mkdir()
        (self.root / 'outside.mp4').write_bytes(b'test')
        (root / 'inside.mp4').write_bytes(b'test')
        self.assertIsNone(resolve_media_asset(root, '../outside.mp4', '.mp4'))
        self.assertIsNone(resolve_media_asset(root, 'missing.mp4', '.mp4'))
        self.assertIsNone(resolve_media_asset(root, 'inside.mp4', '.jpg'))
        self.assertEqual(resolve_media_asset(root, 'inside.mp4', '.mp4'), root / 'inside.mp4')


if __name__ == '__main__':
    unittest.main()
