"""Counts, provenance, collection cleanup, anti-leakage and version integrity."""

import csv
import io
import json
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from data_collection import capture_clip, capture_snapshot, extract_frames, import_paths, probe_video
from dataset_store import DatasetStore, csv_bytes
from test_camera import FakeCamera


ROOT = Path(__file__).resolve().parents[1]


def image_bytes(index=0, format='PNG'):
    image = Image.new('RGB', (64, 48), (index % 256, (index * 23) % 256, (index * 71) % 256))
    output = io.BytesIO()
    image.save(output, format=format)
    return output.getvalue()


def test_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (64, 48))
    assert writer.isOpened()
    for index in range(20):
        writer.write(np.full((48, 64, 3), index * 10, dtype=np.uint8))
    writer.release()
    return path


class AdvancingCamera(FakeCamera):
    instances = []
    def __init__(self, config=None):
        super().__init__(config, bgr=np.full((48, 64, 3), 90, np.uint8))
        self.instances.append(self)

    def read_latest(self, after_sequence):
        if after_sequence:
            self.emit()
        return super().read_latest(after_sequence)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / 'artifacts' / 'dataset_checks' / ('unit_' + uuid.uuid4().hex)
        self.store = DatasetStore(self.root)

    def batch(self, group='group_a', origin='own', content=('fire',), **kwargs):
        return self.store.create_batch(group, group, origin, 'synthetic unit fixture, not actual collected data', content, **kwargs)

    def seed_groups(self, count=10):
        for index in range(count):
            batch = self.batch('group_' + str(index))
            self.store.import_image(image_bytes(index), batch, 'source.png')

    def test_batch_validation_and_empty_progress(self):
        self.assertEqual(self.store.statistics()['total_images'], 0)
        for values in [('', 'g', 'own', 's', ['fire']), ('a', 'g', 'unknown', 's', ['fire']),
                       ('a', 'g', 'own', '', ['fire']), ('a', 'g', 'own', 's', ['normal', 'fire'])]:
            with self.assertRaises(ValueError):
                self.store.create_batch(*values)

    def test_pixel_duplicate_and_reimport_do_not_inflate_count(self):
        one, two = self.batch('a'), self.batch('b')
        first = self.store.import_image(image_bytes(1), one, 'one.png')
        duplicate = self.store.import_image(image_bytes(1, 'BMP'), two, 'renamed.bmp')
        self.assertTrue(first['added'])
        self.assertFalse(duplicate['added'])
        self.assertEqual(first['asset_id'], duplicate['asset_id'])
        self.store.import_image(image_bytes(1), one, 'again.png')
        self.assertEqual(self.store.statistics()['total_images'], 1)
        self.assertEqual(len(self.store.snapshot()['occurrences']), 2)

    def test_external_video_frames_are_counted_without_inventing_original_videos(self):
        batch = self.batch(origin='public')
        self.store.import_image(image_bytes(1), batch, frame_number=30, media_seconds=1.,
                                metadata={'source_video': 'public/raw.mp4', 'source_sha256': 'a' * 64})
        self.store.import_image(image_bytes(2), batch, frame_number=60, media_seconds=2.,
                                metadata={'source_video': 'unverified/path.mp4'})
        stats = self.store.statistics()
        self.assertEqual(stats['derived_images'], 1)
        self.assertEqual(stats['total_videos'], 0)
        self.assertEqual(stats['own_assets'], 0)

    def test_demo_and_source_conflicts_are_not_claimed_as_own(self):
        self.store.import_image(image_bytes(1), self.batch(origin='demo'))
        self.assertEqual(self.store.statistics()['eligible_assets'], 0)
        self.store.import_image(image_bytes(2), self.batch('b', origin='public'))
        self.store.import_image(image_bytes(2), self.batch('c', origin='own'))
        stats = self.store.statistics()
        self.assertEqual((stats['own_fire_images'], stats['provenance_conflicts'], stats['eligible_assets']), (0, 1, 1))

    def test_scene_conflict_is_not_counted_as_normal_or_fire(self):
        self.store.import_image(image_bytes(1), self.batch(content=('normal',)))
        self.store.import_image(image_bytes(1), self.batch('b', content=('fire',)))
        stats = self.store.statistics()
        self.assertEqual((stats['normal_images'], stats['own_fire_images'], stats['scene_conflicts']), (0, 0, 1))

    def test_corrupt_image_and_video_are_rejected_without_assets(self):
        batch = self.batch()
        with self.assertRaises(ValueError):
            self.store.import_image(b'not image', batch)
        path = self.root / 'bad.mp4'
        path.write_bytes(b'not video')
        self.assertFalse(import_paths(self.store, [path], batch)[0]['asset_id'])
        self.assertEqual(self.store.assets(), [])

    def test_video_counts_duration_once_and_frames_keep_parent_and_batch(self):
        batch = self.batch()
        path = test_video(self.root / 'input.mp4')
        result = self.store.import_video(path, batch)
        self.store.import_video(path, batch)
        stats = self.store.statistics()
        self.assertEqual(stats['total_videos'], 1)
        self.assertAlmostEqual(stats['own_fire_video_seconds'], 2, places=2)
        outcome = extract_frames(self.store, result['asset_id'], batch, interval=0.5, maximum=3)
        self.assertEqual((outcome['sampled'], outcome['added']), (3, 3))
        again = extract_frames(self.store, result['asset_id'], batch, interval=0.5, maximum=3)
        self.assertEqual(again['added'], 0)
        frames = [row for row in self.store.snapshot()['occurrences'] if row['parent_id']]
        self.assertEqual(len(frames), 3)
        self.assertEqual([row['media_seconds'] for row in sorted(frames, key=lambda row: row['frame_number'])], [0, 0.5, 1])
        frame_asset = next(row for row in self.store.assets() if row['id'] == frames[0]['asset_id'])
        self.store.import_image(self.store.asset_path(frame_asset), batch)
        self.assertEqual(self.store.statistics()['derived_images'], 3)
        with self.assertRaises(ValueError):
            extract_frames(self.store, result['asset_id'], self.batch('other'))

    def test_invalid_sampling_and_split_parameters_fail_clearly(self):
        for interval in (0, float('nan'), True):
            with self.assertRaises(ValueError):
                extract_frames(self.store, 'missing', 'missing', interval)
        for ratios in ((0.7, 0.2, 0), (0.7, 0.2, 0.2), (float('nan'), 0.2, 0.1)):
            with self.assertRaises(ValueError):
                self.store.plan_split(ratios)

    def test_tampered_video_cannot_generate_frames_with_old_provenance(self):
        batch = self.batch()
        result = self.store.import_video(test_video(self.root / 'input.mp4'), batch)
        asset = next(row for row in self.store.assets() if row['id'] == result['asset_id'])
        self.store.asset_path(asset).write_bytes(b'changed')
        with self.assertRaisesRegex(OSError, '内容已改变'):
            extract_frames(self.store, asset['id'], batch)
        self.assertEqual(self.store.statistics()['total_images'], 0)

    def test_grouped_split_is_reproducible_and_lists_are_disjoint(self):
        self.seed_groups()
        first, second = self.store.plan_split(seed=18), self.store.plan_split(seed=18)
        self.assertEqual(first['assets'], second['assets'])
        self.assertEqual(first['image_counts'], {'train': 7, 'val': 2, 'test': 1})
        version = self.store.save_split(first)
        lists = [(self.root / 'splits' / version / (name + '.txt')).read_text().splitlines() for name in ('train', 'val', 'test')]
        self.assertEqual(len(set().union(*map(set, lists))), 10)
        self.assertTrue(all(not set(lists[a]) & set(lists[b]) for a, b in ((0, 1), (0, 2), (1, 2))))

    def test_duplicates_link_distinct_declared_groups_before_split(self):
        self.seed_groups(4)
        original = self.store.assets()[0]
        linked = {row['batch_id'] for row in self.store.snapshot()['occurrences'] if row['asset_id'] == original['id']}
        other_batch = next(row for row in self.store.batches() if row['id'] not in linked)
        self.store.import_image(self.store.asset_path(original), other_batch['id'])
        plan = self.store.plan_split()
        self.assertEqual(plan['group_count'], 3)
        group_splits = {}
        for row in plan['assets']:
            for group in row['groups']:
                group_splits.setdefault(group, set()).add(row['split'])
        self.assertTrue(all(len(value) == 1 for value in group_splits.values()))

    def test_parent_video_and_all_sampled_frames_stay_together(self):
        self.seed_groups(3)
        batch = self.batch('video_group')
        video = self.store.import_video(test_video(self.root / 'parent.mp4'), batch)
        extract_frames(self.store, video['asset_id'], batch, interval=0.5, maximum=3)
        plan = self.store.plan_split()
        assignments = {row['id']: row['split'] for row in plan['assets']}
        for occurrence in self.store.snapshot()['occurrences']:
            if occurrence['parent_id']:
                self.assertEqual(assignments[occurrence['asset_id']], assignments[occurrence['parent_id']])

    def test_one_large_group_cannot_be_split_to_fill_ratios(self):
        batch = self.batch()
        for index in range(10):
            self.store.import_image(image_bytes(index), batch)
        with self.assertRaisesRegex(ValueError, '至少需要3个'):
            self.store.plan_split()

    def test_custom_ratios_place_largest_group_in_largest_requested_set(self):
        for index, count in enumerate((3, 1, 1)):
            batch = self.batch('group_' + str(index))
            for item in range(count):
                self.store.import_image(image_bytes(index * 10 + item), batch)
        self.assertEqual(self.store.plan_split((0.1, 0.8, 0.1))['image_counts'], {'train': 1, 'val': 3, 'test': 1})

    def test_saved_version_is_immutable_and_changed_data_marks_it_historical(self):
        self.seed_groups(3)
        plan = self.store.plan_split()
        version = self.store.save_split(plan)
        path = self.root / 'splits' / version / 'manifest.json'
        before = path.read_bytes()
        self.assertTrue(self.store.splits()[0]['current'])
        self.store.import_image(image_bytes(20), self.batch('new'))
        self.assertFalse(self.store.splits()[0]['current'])
        self.assertEqual(path.read_bytes(), before)
        with self.assertRaisesRegex(ValueError, '已变化'):
            self.store.save_split(plan)

    def test_missing_or_modified_file_blocks_export(self):
        self.seed_groups(3)
        plan = self.store.plan_split()
        self.store.asset_path(plan['assets'][0]).write_bytes(b'changed')
        with self.assertRaises(OSError):
            self.store.save_split(plan)
        self.assertEqual(self.store.splits(), [])

    def test_csv_preserves_source_hash_and_neutralizes_formulas(self):
        batch = self.store.create_batch('=batch', 'g', 'own', '=source', ['normal'])
        self.store.import_image(image_bytes(2), batch, 'sample.png')
        row = next(csv.DictReader(io.StringIO(csv_bytes(self.store.manifest_rows()).decode('utf-8-sig'))))
        self.assertEqual(row['batch'], "'=batch")
        self.assertEqual(len(row['source_sha256']), 64)

    def test_concurrent_import_keeps_one_asset_and_both_origins(self):
        batches = [self.batch('a'), self.batch('b')]
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda batch: self.store.import_image(image_bytes(1), batch), batches))
        self.assertEqual(sum(row['added'] for row in results), 1)
        self.assertEqual(len(self.store.snapshot()['occurrences']), 2)

    def test_camera_collection_uses_clean_frames_and_releases_source(self):
        batch = self.batch(origin='demo')
        image = capture_snapshot(self.store, batch, source_factory=AdvancingCamera)
        self.assertFalse(AdvancingCamera.instances[-1].opened)
        duplicate = capture_snapshot(self.store, batch, source_factory=AdvancingCamera)
        self.assertEqual(image['asset_id'], duplicate['asset_id'])
        self.assertFalse(duplicate['added'])
        video = capture_clip(self.store, batch, 1.0, source_factory=AdvancingCamera, sleep=lambda _: None)
        self.assertFalse(AdvancingCamera.instances[-1].opened)
        asset = next(row for row in self.store.assets() if row['id'] == video['asset_id'])
        self.assertAlmostEqual(probe_video(self.store.asset_path(asset))['duration'], 1, places=2)
        self.assertFalse(list(self.root.glob('capture_*')))
        self.assertEqual(self.store.statistics()['eligible_assets'], 0)
        receipts = [json.loads(row['metadata_json']) for row in self.store.snapshot()['occurrences']]
        self.assertEqual({row['type'] for row in receipts}, {'usb_snapshot', 'usb_clip'})
        self.assertEqual(len({row['capture_id'] for row in receipts}), 3)

    def test_capture_failure_releases_source_and_never_registers_short_clip(self):
        class BrokenCamera(AdvancingCamera):
            def read_latest(self, sequence):
                if sequence:
                    raise RuntimeError('controlled camera failure')
                return super().read_latest(sequence)
        with self.assertRaises(RuntimeError):
            capture_clip(self.store, self.batch(), 1, source_factory=BrokenCamera, sleep=lambda _: None)
        self.assertFalse(BrokenCamera.instances[-1].opened)
        self.assertFalse(list(self.root.glob('capture_*')))
        self.assertEqual(self.store.assets(), [])


if __name__ == '__main__':
    unittest.main()
