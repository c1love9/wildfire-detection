"""Geographic orientation, archived locations and source lifecycle integration."""

import csv
import base64
import io
import json
import math
import unittest
import uuid
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

from alarm import AlarmConfig
from alarm_log import AlarmLog, events_csv
from camera_session import CameraSession
from camera_source import CameraConfig
from monitoring import MonitoringPoint, destination, load_monitoring_points, point_from_metadata, view_sector
from monitoring_ui import location_deck
from test_camera import Detector, FakeCamera
from video_detection import VideoSession


ROOT = Path(__file__).resolve().parents[1]


class MonitoringTests(unittest.TestCase):
    def setUp(self):
        self.points = load_monitoring_points(ROOT / 'configs' / 'monitoring_points.json')
        self.point = self.points[0]
        self.root = ROOT / 'artifacts' / 'monitoring_checks' / ('unit_' + uuid.uuid4().hex)
        self.path = self.root / 'events.sqlite3'
        self.cfg = replace(AlarmConfig(), confirm_frames=2, attention_seconds=0.1, warning_seconds=0.3, alarm_seconds=0.5)

    def test_presets_round_trip_and_cannot_claim_hardware_gps(self):
        self.assertEqual(len({point.id for point in self.points}), len(self.points))
        self.assertEqual(point_from_metadata({'monitoring_point': self.point.metadata()}), self.point)
        self.assertTrue(all(point.simulated for point in self.points))
        for change in ({'simulated': False}, {'coordinate_system': 'GCJ02'}):
            with self.assertRaises(ValueError):
                replace(self.point, **change)

    def test_nonfinite_and_out_of_range_coordinates_or_angles_are_rejected(self):
        for change in ({'latitude': 91}, {'longitude': -181}, {'longitude': math.nan}, {'latitude': True},
                       {'heading': -1}, {'heading': math.inf}, {'pitch': -91}, {'horizontal_fov': 0}, {'name': ''}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(self.point, **change)

    def test_direction_uses_clockwise_bearings_and_lon_lat_order(self):
        north = destination(self.point, 0)
        east = destination(self.point, 90)
        self.assertAlmostEqual(north[0], self.point.longitude)
        self.assertGreater(north[1], self.point.latitude)
        self.assertGreater(east[0], self.point.longitude)
        self.assertAlmostEqual(east[1], self.point.latitude, places=6)
        sector = view_sector(self.point)
        self.assertEqual(sector[0], [self.point.longitude, self.point.latitude])
        self.assertEqual(sector[-1], sector[0])
        self.assertEqual(sector[13], destination(self.point, self.point.heading))

    def test_old_or_malformed_metadata_never_inherits_current_location(self):
        for metadata in ('{}', '{broken', 'null', '[]', {'monitoring_point': None},
                         {'monitoring_point': {'latitude': 31}}, {'monitoring_point': {**self.point.metadata(), 'latitude': math.nan}}):
            self.assertIsNone(point_from_metadata(metadata))

    def test_file_event_archives_original_location_and_csv(self):
        session = VideoSession(ROOT / 'demo-videos' / 'fire1.mov', Detector(), alarm_config=self.cfg,
                               log_path=self.path, monitoring_point=self.point)
        try:
            for _ in range(20):
                session.step()
        finally:
            session.stop()
        changed_point = replace(self.point, longitude=self.point.longitude + 1, heading=180)
        with self.assertRaises(FrozenInstanceError):
            self.point.heading = 180
        events = AlarmLog(self.path).events()
        self.assertEqual(point_from_metadata(events[0]['metadata_json']), self.point)
        self.assertNotEqual(point_from_metadata(events[0]['metadata_json']), changed_point)
        row = next(csv.DictReader(io.StringIO(events_csv(events).decode('utf-8-sig'))))
        self.assertEqual(row['位置来源'], '模拟（WGS84）')
        self.assertEqual(float(row['经度']), self.point.longitude)
        self.assertEqual(float(row['水平角（度）']), self.point.heading)

    def test_camera_new_observation_keeps_location_without_merging_events(self):
        camera = CameraSession(CameraConfig(), Detector(), alarm_config=self.cfg, log_path=self.path,
                               source_factory=FakeCamera, monitoring_point=self.point)
        try:
            for _ in range(2):
                for _ in range(8):
                    camera.step()
                    camera.capture.emit()
                camera.pause()
                if camera.segment_number == 1:
                    camera.resume()
        finally:
            camera.stop()
        events = AlarmLog(self.path).events()
        self.assertEqual(len(events), 2)
        self.assertEqual(len({row['run_id'] for row in events}), 2)
        self.assertTrue(all(point_from_metadata(row['metadata_json']) == self.point for row in events))

    def test_legacy_event_exports_as_missing_not_zero_coordinates(self):
        camera = CameraSession(CameraConfig(), Detector(), alarm_config=self.cfg, log_path=self.path, source_factory=FakeCamera)
        try:
            for _ in range(8):
                camera.step()
                camera.capture.emit()
        finally:
            camera.stop()
        row = next(csv.DictReader(io.StringIO(events_csv(AlarmLog(self.path).events()).decode('utf-8-sig'))))
        self.assertEqual((row['监控点'], row['经度'], row['纬度']), ('未记录', '', ''))

    def test_history_map_uses_event_snapshots_and_offline_style_has_no_network_sources(self):
        events = [dict(id=str(index), metadata_json=json.dumps({'monitoring_point': point.metadata()}),
                       peak_level=3, started_at='2026-10-20T00:00:00') for index, point in enumerate(self.points)]
        events.append(dict(id='legacy', metadata_json='{}'))
        deck = json.loads(location_deck(self.point, level=3, events=events, selected_id='0', basemap=False).to_json())
        markers = deck['layers'][2]['data']
        self.assertEqual([row['position'] for row in markers], [[point.longitude, point.latitude] for point in self.points])
        self.assertEqual(markers[0]['color'], [220, 45, 45, 230])
        self.assertTrue(deck['mapStyle'].startswith('data:application/json;base64,'))
        style = json.loads(base64.b64decode(deck['mapStyle'].split(',', 1)[1]))
        self.assertEqual(style['version'], 8)
        self.assertEqual(style['sources'], {})
        self.assertNotIn('glyphs', style)
        self.assertNotIn('sprite', style)


if __name__ == '__main__':
    unittest.main()
