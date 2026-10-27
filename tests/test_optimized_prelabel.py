"""Known-answer token scoring, original coordinates and fusion invariants."""
from dataclasses import replace
import json
from pathlib import Path
import re
import unittest
import uuid

import torch

from optimized_prelabel import (OptimizedPrelabelConfig, OptimizedPrelabeler,
                               query_tokens, joint_boxes, merge_candidates)


class Tokenizer:
    cls_token_id = 101
    sep_token_id = 102

    def encode(self, text, add_special_tokens=True):
        vocab = {'fire': 1, 'smoke': 2, 'plume': 3, 'flame': 4, '.': 5}
        values = [vocab[w] for w in re.findall(r'[a-z]+|\.', text)]
        return [101, *values, 102] if add_special_tokens else values


class OptimizedPrelabelTests(unittest.TestCase):
    def setUp(self):
        self.config = OptimizedPrelabelConfig()

    def boxes(self, probabilities, bounds=None, spans=None, groups=None, config=None):
        probs = torch.tensor(probabilities)
        coords = torch.tensor(bounds if bounds is not None else [[.5, .5, .4, .2]] * len(probs))
        return joint_boxes(torch.logit(probs), coords, spans or [[[1]], [[3]]],
                           groups or [['fire'], ['smoke']], 200, 100, config or self.config)

    def test_query_spans_exclude_special_tokens_and_punctuation(self):
        text, spans, groups = query_tokens(Tokenizer(), self.config)
        self.assertEqual(text, 'fire. smoke. ')
        self.assertEqual(spans, [[[1]], [[3]]])
        self.assertEqual(groups, [['fire'], ['smoke']])

    def test_overlapping_fire_smoke_are_not_forced_into_one_class(self):
        boxes = self.boxes([[.99, .8, .99, .7, .99, .99], [.99, .1, .99, .1, .99, .99]])
        self.assertEqual({b['class'] for b in boxes}, {'fire', 'smoke'})
        self.assertEqual(len({b['id'] for b in boxes}), 2)
        for b in boxes:
            for actual, expected in zip(b['xyxy'], [60, 40, 140, 60]):
                self.assertAlmostEqual(actual, expected, places=4)

    def test_special_token_high_scores_do_not_generate_false_boxes(self):
        self.assertEqual(self.boxes([[.99, .1, .99, .1, .99, .99]]), [])

    def test_nms_suppresses_duplicates_only_within_class(self):
        boxes = self.boxes([[.5, .8, .5, .7, .5], [.5, .7, .5, .6, .5]])
        self.assertEqual(len(boxes), 2)
        self.assertEqual({b['class'] for b in boxes}, {'fire', 'smoke'})

    def test_compound_phrase_uses_mean_instead_of_strongest_word(self):
        config = replace(self.config, smoke_prompt='smoke plume.', smoke_threshold=.6)
        boxes = self.boxes([[.5, .1, .5, .9, .1, .5]], spans=[[[1]], [[3, 4]]],
                           groups=[['fire'], ['smoke plume']], config=config)
        self.assertEqual(boxes, [])

    def test_nonfinite_outputs_are_failure_not_negative_image(self):
        with self.assertRaises(ValueError):
            joint_boxes(torch.tensor([[float('nan'), .5]]), torch.tensor([[.5, .5, .4, .2]]),
                        [[[0]], [[1]]], [['fire'], ['smoke']], 200, 100, self.config)

    def test_masked_padding_negative_infinity_is_valid(self):
        logits = torch.tensor([[float('-inf'), 2., float('-inf'), 1., float('-inf')]])
        boxes = joint_boxes(logits, torch.tensor([[.5, .5, .4, .2]]), [[[1]], [[3]]],
                            [['fire'], ['smoke']], 200, 100, self.config)
        self.assertEqual({b['class'] for b in boxes}, {'fire', 'smoke'})

    def test_fusion_preserves_dino_geometry_and_cross_class_overlap(self):
        def b(key, label, bounds, score):
            return {'id': key, 'class': label, 'xyxy': bounds, 'confidence': score}
        dino = b('dino', 'fire', [10, 10, 30, 30], .3)
        aux = [b('duplicate', 'fire', [11, 11, 31, 31], .99),
               b('smoke', 'smoke', [10, 10, 30, 30], .8),
               b('other', 'fire', [60, 60, 80, 80], .9)]
        merged = merge_candidates([dino], aux, self.config)
        self.assertEqual({b['id'] for b in merged}, {'dino', 'smoke', 'other'})
        self.assertEqual(next(b for b in merged if b['id'] == 'dino')['xyxy'], [10, 10, 30, 30])

    def test_bad_config_and_cross_class_duplicate_phrase_are_rejected(self):
        for values in ({'fire_threshold': float('nan')}, {'smoke_threshold': True},
                       {'shortest_edge': 900}, {'fire_prompt': ''}, {'smoke_prompt': 'fire.'},
                       {'auxiliary_sha256': 'wrong'}, {'revision': 'main'}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                replace(self.config, **values)

    def test_untrusted_receipt_rejected_before_loading(self):
        root = Path(__file__).resolve().parents[1] / 'artifacts/optimized_prelabel_checks' / uuid.uuid4().hex
        root.mkdir(parents=True)
        with self.assertRaises(OSError):
            OptimizedPrelabeler(directory=root)
        (root / 'model_receipt.json').write_text(json.dumps({'model_id': 'other'}), encoding='utf-8')
        with self.assertRaises(ValueError):
            OptimizedPrelabeler(directory=root)


if __name__ == '__main__':
    unittest.main()
