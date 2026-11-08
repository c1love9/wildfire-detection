"""Model selection must be frozen on validation, never independent test scores."""
import sys
from pathlib import Path
import unittest
from unittest.mock import Mock,patch
import threading
import tempfile
import json
import os
from contextlib import closing
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from complete_reviewed_workflow import select_by_validation
from audit_requirements import remaining_gates, requirements, load_worktime_evidence
from summarize_worktime import summarize
from model_runtime import YoloDetector
from restore_final_delivery import restore, sha256, safe_path


class FormalWorkflowTests(unittest.TestCase):
    def test_reported_worktime_does_not_close_segmentation_reference_gap(self):
        facts = {'images': 700, 'videos': 300, 'video_seconds': 900, 'proposals': 1600, 'review_versions': 701,
                 'approved_positive_images': 507, 'approved_negative_images': 193, 'latest_approved_images': 700,
                 'reviewed_exports': 1, 'completed_training_jobs': 2, 'completed_comparisons': 1,
                 'independently_double_submitted_images': 200, 'reference_mask_records': 0, 'worktime_records': 0}
        estimate = {'assisted': {'seconds_per_image': 5}, 'manual_mean_seconds_per_image': 10,
                    'target_images': 700, 'estimated_saved_seconds': 3500, 'estimated_saved_fraction': .5}
        rows = {row['id']: row for row in requirements(facts, worktime=estimate)}
        self.assertIn('已有', rows['T2-06']['status'])
        self.assertIn('待参考', rows['T4-06']['status'])
        gates = remaining_gates(facts, {'status': 'completed'}, estimate)
        self.assertFalse(any('工时估算尚缺' in gate for gate in gates))
        self.assertTrue(any('分割IoU' in gate for gate in gates))
        self.assertEqual(facts['worktime_records'], 0)

    def test_worktime_estimate_rejects_later_modified_manual_timers(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as temp:
            catalog=Path(temp)/'catalog.sqlite3';path=Path(temp)/'estimate.json'
            with closing(sqlite3.connect(catalog)) as db, db:
                db.executescript('CREATE TABLE quality_submissions(task_id TEXT,asset_id TEXT,role TEXT,seconds REAL);'
                                 'CREATE TABLE annotation_reviews(seq INTEGER,asset_id TEXT,decision TEXT);')
                db.executemany('INSERT INTO quality_submissions VALUES(?,?,?,?)',[('test','image','A',8),('test','image','B',10)])
                db.execute("INSERT INTO annotation_reviews VALUES(1,'image','approved')")
            estimate=summarize(catalog,5,'Synthetic unit-test input, not real measurement.',task_id='test')
            path.write_text(json.dumps(estimate),encoding='utf-8')
            self.assertIsNotNone(load_worktime_evidence(path,catalog))
            with closing(sqlite3.connect(catalog)) as db, db:db.execute("UPDATE quality_submissions SET seconds=80 WHERE role='A'")
            with self.assertRaises(ValueError):load_worktime_evidence(path,catalog)

    def test_choose_validation_winner_and_tie_baseline(self):
        def result(value): return {'metrics': {'overall': {'map50_95': value}}, 'test_map': 1-value}
        jobs = {'baseline': 'a', 'multiscale': 'b'}
        selection = select_by_validation(jobs, {'baseline': result(.4), 'multiscale': result(.5)})
        self.assertEqual(selection['training_id'], 'b')
        self.assertFalse(selection['test_used_for_selection'])
        self.assertEqual(select_by_validation(jobs, {'baseline': result(.5), 'multiscale': result(.5)})['training_id'], 'a')

    def test_missing_validation_cannot_be_replaced_by_test(self):
        with self.assertRaises(ValueError):
            select_by_validation({'baseline': 'a', 'multiscale': 'b'},
                                 {'baseline': {'metrics': {'overall': {'map50_95': None}}},
                                  'multiscale': {'metrics': {'overall': {'map50_95': .3}}}})

    def test_completed_reviews_do_not_hide_negative_shortfall(self):
        facts = {'images': 700, 'videos': 300, 'video_seconds': 900, 'proposals': 1600, 'review_versions': 701,
                 'approved_positive_images': 507, 'approved_negative_images': 193, 'latest_approved_images': 700,
                 'reviewed_exports': 1, 'completed_training_jobs': 2, 'completed_comparisons': 1,
                 'independently_double_submitted_images': 200, 'reference_mask_records': 0, 'worktime_records': 0}
        rows = {row['id']: row for row in requirements(facts)}
        self.assertEqual(rows['T2-03']['status'], '全部图像复核已保存')
        self.assertEqual(rows['T1-03']['status'], '数量待补')
        self.assertIn('7幅', rows['T1-03']['remaining'])
        gates = remaining_gates(facts, {'status': 'completed'})
        self.assertTrue(any('7幅' in gate for gate in gates))
        self.assertFalse(any('剩余图像真实人工复核' in gate for gate in gates))
        self.assertTrue(any('人工参考掩码' in gate for gate in gates))

    def test_metric_inference_does_not_draw_or_resize_ui_image(self):
        detector = object.__new__(YoloDetector)
        detector._lock = threading.Lock()
        boxes = [{'class': 'fire', 'confidence': .8, 'xyxy': [1, 2, 3, 4]}]
        detector._infer = Mock(return_value=(None, boxes, {'fire': 1}))
        detector.predict_image = Mock(side_effect=AssertionError('UI drawing entered benchmark'))
        self.assertEqual(detector.predict_boxes(Image.new('L', (12, 10))), boxes)
        args = detector._infer.call_args.args
        self.assertEqual(args[0].mode, 'RGB')
        self.assertEqual(args[1:], (.001, .5))

    def test_restore_validates_all_inputs_before_any_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); package=root/'package'; project=root/'new_project'
            package.mkdir(); (package/'08_验收与校验').mkdir()
            (package/'one').write_bytes(b'first'); (package/'two').write_bytes(b'second')
            files={name:{'bytes':(package/name).stat().st_size,'sha256':sha256(package/name)} for name in ('one','two')}
            (package/'manifest.json').write_text(json.dumps({'release_id':'test','files':files}))
            mapping={'files':[{'source':n,'target':'data/'+n} for n in files]}
            (package/'08_验收与校验/restore_map.json').write_text(json.dumps(mapping))
            (package/'two').write_bytes(b'changed')
            with self.assertRaises(OSError):restore(package,project)
            self.assertFalse(project.exists())
            (package/'two').write_bytes(b'second')
            self.assertEqual(restore(package,project)['restored_files'],2)
            self.assertEqual(restore(package,project)['restored_files'],0)
            (project/'data/two').write_bytes(b'my existing work')
            with self.assertRaises(FileExistsError):restore(package,project)
            self.assertEqual((project/'data/two').read_bytes(),b'my existing work')

    def test_restore_path_cannot_escape_selected_root(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):safe_path(Path(temp),'../outside')

    def test_version_two_restores_custom_layout_and_checks_mapping_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);package=root/'package';project=root/'project'
            package.mkdir();(package/'payload').write_bytes(b'verified model')
            map_path=package/'new_layout/map.json';map_path.parent.mkdir()
            map_path.write_text(json.dumps({'files':[{'source':'payload','target':'models/model.pt'}]}))
            files={name:{'bytes':(package/name).stat().st_size,'sha256':sha256(package/name)}
                   for name in ('payload','new_layout/map.json')}
            manifest={'schema_version':2,'release_id':'test','restore_map':'new_layout/map.json',
                      'offline_models_included':True,'files':files}
            (package/'manifest.json').write_text(json.dumps(manifest))
            original=map_path.read_bytes()
            map_path.write_text(json.dumps({'files':[{'source':'payload','target':'different.pt'}]}))
            with self.assertRaises(OSError):restore(package,project)
            self.assertFalse(project.exists())
            map_path.write_bytes(original)
            result=restore(package,project)
            self.assertTrue(result['offline_models_included'])
            self.assertEqual((project/'models/model.pt').read_bytes(),b'verified model')

    def test_version_two_rejects_unverified_or_outside_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);package=root/'package';package.mkdir()
            manifest={'schema_version':2,'release_id':'test','restore_map':'unlisted.json','files':{}}
            (package/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):restore(package,root/'project')
            manifest['restore_map']='../outside.json'
            (package/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):restore(package,root/'project')

    def test_cpu_prediction_cannot_hide_gpu_for_later_first_load(self):
        detector=object.__new__(YoloDetector);detector.device='cpu';detector.imgsz=640
        detector.model=Mock()
        def predict(*args,**kwargs):
            self.assertEqual(str(kwargs['device']),'cpu')
            self.assertNotIsInstance(kwargs['device'],str)
            os.environ['CUDA_VISIBLE_DEVICES']='-1'
            result=Mock();result.names={0:'fire',1:'smoke'};result.boxes.data.cpu.return_value.tolist.return_value=[]
            return [result]
        detector.model.predict.side_effect=predict
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}):
            detector._infer(Image.new('RGB',(10,10)),.25,.5)
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'0')
        with patch.dict(os.environ):
            os.environ.pop('CUDA_VISIBLE_DEVICES',None)
            detector._infer(Image.new('RGB',(10,10)),.25,.5)
            self.assertNotIn('CUDA_VISIBLE_DEVICES',os.environ)

    def test_prediction_error_also_restores_gpu_visibility(self):
        detector=object.__new__(YoloDetector);detector.device='cpu';detector.imgsz=640;detector.model=Mock()
        def fail(*args,**kwargs):
            os.environ['CUDA_VISIBLE_DEVICES']='-1'
            raise RuntimeError('simulated third-party error')
        detector.model.predict.side_effect=fail
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}):
            with self.assertRaises(RuntimeError):detector._infer(None,.25,.5)
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'0')


if __name__ == '__main__':
    unittest.main()
