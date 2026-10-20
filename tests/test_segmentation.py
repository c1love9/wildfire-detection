import torch
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
from inference_result import FrameResult, merge_masks
from segmentation import SegmentedDetector, overlay_masks, verify_model


class SegmentationTests(unittest.TestCase):
    def test_union_and_overlap(self):
        first = np.zeros((10, 10), bool); first[:5] = True
        second = np.zeros((10, 10), bool); second[:, :5] = True
        detections = [{'class': 'fire'}, {'class': 'fire'}, {'class': 'smoke'}]
        masks = merge_masks([first, first, second], detections, (10, 10))
        result = FrameResult(np.zeros((10, 10, 3), np.uint8), detections, {}, 10, 10, masks=masks)
        self.assertEqual(result.areas(), {'fire': .5, 'smoke': .5, 'combined': .75})
        self.assertEqual(detections[0]['area_ratio'], .5)

    def test_unknown_is_not_zero(self):
        result = FrameResult(np.zeros((10, 10, 3), np.uint8), [], {}, 10, 10)
        self.assertIsNone(result.areas()['combined'])
        result.masks = merge_masks([], [], (10, 10))
        self.assertEqual(result.areas()['combined'], 0)

    def test_bad_shape_and_count(self):
        with self.assertRaises(ValueError): merge_masks([], [{'class': 'fire'}], (10, 10))
        with self.assertRaises(ValueError): merge_masks([np.zeros((5, 5))], [{'class': 'fire'}], (10, 10))
        with self.assertRaises(ValueError): merge_masks([np.full((10, 10), np.nan)], [{'class': 'fire'}], (10, 10))

    def test_overlay_does_not_mutate_original(self):
        original = np.zeros((10, 10, 3), np.uint8)
        masks = {'fire': np.ones((10, 10), bool), 'smoke': np.zeros((10, 10), bool)}
        output = overlay_masks(original, masks)
        self.assertGreater(output.sum(), 0)
        self.assertEqual(original.sum(), 0)

    def test_failure_does_not_reuse_previous_mask(self):
        class Detector:
            device, model, metadata = 'cpu', object(), {}
            def predict_frame(self, bgr, conf, iou):
                return FrameResult(bgr.copy(), [{'class': 'fire', 'confidence': .9, 'xyxy': [0, 0, 10, 10]}], {'fire': 1}, 10, 10)
        class Segmenter:
            metadata = {}
            def predict(self, image, boxes): return [np.ones((10, 10), bool)]
        segmenter = Segmenter(); detector = SegmentedDetector(Detector(), segmenter)
        self.assertEqual(detector.predict_frame(np.zeros((10, 10, 3), np.uint8), .2, .5).areas()['fire'], 1)
        with patch.object(segmenter, 'predict', side_effect=RuntimeError('controlled failure')):
            failed = detector.predict_frame(np.zeros((10, 10, 3), np.uint8), .2, .5)
        self.assertIsNone(failed.areas()['fire'])
        self.assertNotIn('area_ratio', failed.detections[0])

    def test_missing_model_is_explicit(self):
        with self.assertRaises(OSError): verify_model(Path(__file__).parent / 'not-a-model')


if __name__ == '__main__': unittest.main()
