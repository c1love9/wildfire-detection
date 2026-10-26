import torch
import unittest
import uuid
import json
from pathlib import Path
import numpy as np
from test_training import seed
from quality import QualityStore,WorkTimer,mask_metrics,pause_work_timers,preannotation_metrics
from test_annotations import Model
from grounding_prelabel import PrelabelConfig


class QualityTests(unittest.TestCase):
    def store(self,proposals=None):
        root=Path(__file__).resolve().parents[1]/'artifacts/completion_checks'/('quality_'+uuid.uuid4().hex)
        dataset,_,_=seed(root);store=QualityStore(dataset);key=store.annotations.catalog()[0]['id']
        if proposals is not None:store.annotations.prelabel([key],Model(proposals),PrelabelConfig(),'Synthetic candidate fixture')
        _,tokens=store.create([key],'Simulated A','Simulated B','Automated simulation')
        return store,key,tokens

    def test_access_separation_and_lock(self):
        store,key,tokens=self.store()
        store.submit(tokens['A'],key,[],1,True)
        self.assertIsNone(store.own_submission(tokens['B'],key))
        with self.assertRaises(ValueError):store.submissions(tokens['A'])
        with self.assertRaises(ValueError):store.submissions(tokens['admin'])
        with self.assertRaises(ValueError):store.submit(tokens['A'],key,[],2,True)
        store.submit(tokens['B'],key,[],1,True)
        report=store.report(tokens['admin'])
        self.assertEqual(report['images'],1);self.assertFalse(report['at_least_200_images'])
        self.assertEqual(report['preannotation_quality'],{'A':None,'B':None})
        self.assertIsNone(report['agreement_summary']['mean_iou_of_matched'])
        self.assertEqual(report['agreement_summary']['both_empty_images'],1)

    def test_empty_submission_requires_confirmation(self):
        store,key,tokens=self.store()
        with self.assertRaises(ValueError):store.submit(tokens['A'],key,[],1)

    def test_timed_submit_pauses_active_interval_and_keeps_elapsed_seconds(self):
        store,key,tokens=self.store();clock=[0.];timer=WorkTimer(lambda:clock[0])
        timer.start();clock[0]=4.;timer.activity();clock[0]=7.
        store.submit_timed(tokens['A'],key,[{'class':'fire','xyxy':[1,1,8,8]}],timer)
        self.assertIsNone(timer.last)
        self.assertEqual(store.own_submission(tokens['A'],key)['seconds'],7.)
        clock[0]=12.;self.assertEqual(timer.seconds(),7.)
        with self.assertRaises(ValueError):store.submit_timed(tokens['A'],key,[],timer,True)

    def test_timed_submit_still_requires_measured_work_and_empty_confirmation(self):
        store,key,tokens=self.store();clock=[0.];timer=WorkTimer(lambda:clock[0])
        with self.assertRaises(ValueError):store.submit_timed(tokens['A'],key,[],timer,True)
        timer.start();clock[0]=3.
        with self.assertRaises(ValueError):store.submit_timed(tokens['A'],key,[],timer)
        self.assertIsNone(store.own_submission(tokens['A'],key))
        store.submit_timed(tokens['A'],key,[],timer,True)
        self.assertEqual(store.own_submission(tokens['A'],key)['seconds'],3.)

    def test_progress_exposes_only_own_answers_and_admin_counts(self):
        store,key,tokens=self.store()
        store.submit(tokens['A'],key,[],1,True)
        self.assertEqual(store.progress(tokens['A'])['remaining_asset_ids'],[])
        self.assertEqual(store.progress(tokens['B'])['submitted'],0)
        self.assertEqual(store.progress(tokens['B'])['remaining_asset_ids'],[key])
        self.assertEqual(store.progress(tokens['admin']),{'total':1,'A':1,'B':0})
        with self.assertRaises(ValueError):store.report(tokens['admin'])

    def test_candidate_refreeze_rejects_any_existing_human_submission(self):
        store,key,tokens=self.store()
        original=store.access(tokens['admin'])['snapshot_json']
        store.submit(tokens['A'],key,[],1,True)
        with self.assertRaises(ValueError):store.refresh_proposals_before_start(tokens['admin'],'Synthetic attempt after work started')
        self.assertEqual(store.access(tokens['admin'])['snapshot_json'],original)

    def test_report_compares_frozen_candidates_to_A_and_B_separately(self):
        predicted=[{'class':'fire','xyxy':[1,1,8,8],'confidence':.05},
                   {'class':'smoke','xyxy':[2,2,9,9],'confidence':.8}]
        store,key,tokens=self.store(predicted)
        a=[{'class':'fire','xyxy':[1,1,8,8]}]
        b=[{'class':'smoke','xyxy':[2,2,9,9]}]
        store.submit(tokens['A'],key,a,3)
        store.submit(tokens['B'],key,b,4)
        originals=store.submissions(tokens['admin']);latest=store.annotations.latest(key)
        # A later candidate run must not silently replace frozen predictions.
        store.annotations.prelabel([key],Model([]),PrelabelConfig(),'Synthetic later empty candidate')
        report=store.report(tokens['admin'])
        for role in ('A','B'):
            self.assertEqual(report['preannotation_quality'][role]['overall']['tp'],1)
            self.assertEqual(report['preannotation_quality'][role]['overall']['fp'],1)
            self.assertEqual(report['preannotation_quality'][role]['overall']['recall'],1)
        self.assertEqual(report['preannotation_quality']['A']['per_class'][0]['tp'],1)
        self.assertEqual(report['preannotation_quality']['B']['per_class'][1]['tp'],1)
        self.assertEqual(report['agreement_summary']['unmatched_A'],1)
        self.assertEqual(report['agreement_summary']['unmatched_B'],1)
        self.assertEqual(report['manual_work_seconds'],{'A':3,'B':4})
        self.assertEqual(store.submissions(tokens['admin']),originals)
        self.assertEqual(store.annotations.latest(key),latest)
        with store.dataset.connection() as db:self.assertEqual(db.execute('SELECT COUNT(*) FROM quality_references').fetchone()[0],0)

    def test_existing_legacy_references_are_preserved_but_do_not_control_report(self):
        store,key,tokens=self.store()
        for role in ('A','B'):store.submit(tokens[role],key,[],2,True)
        task=store.access(tokens['admin'])['task_id']
        with store.dataset.connection() as db:
            db.execute('INSERT INTO quality_references(task_id,asset_id,boxes_json,actor,note,created_at) VALUES(?,?,?,?,?,?)',
                       (task,key,json.dumps([{'class':'fire','xyxy':[1,1,8,8]}]),'Legacy simulation','Historical fixture','2026-01-01'))
        report=store.report(tokens['admin'])
        self.assertEqual(report['agreement_summary']['both_empty_images'],1)
        with store.dataset.connection() as db:self.assertEqual(db.execute('SELECT COUNT(*) FROM quality_references').fetchone()[0],1)

    def test_prelabel_matching_handles_duplicates_wrong_classes_missing_and_empty(self):
        fire={'class':'fire','xyxy':[1,1,8,8],'confidence':.8}
        smoke={**fire,'class':'smoke'}
        duplicated=preannotation_metrics({'x':[fire]},{'x':[fire,fire]})['overall']
        self.assertEqual((duplicated['tp'],duplicated['fp'],duplicated['fn']),(1,1,0))
        wrong=preannotation_metrics({'x':[fire]},{'x':[smoke]})['overall']
        self.assertEqual((wrong['tp'],wrong['fp'],wrong['fn']),(0,1,1))
        empty=preannotation_metrics({'x':[]},{'x':[]})
        self.assertIsNone(empty['overall']['precision']);self.assertIsNone(empty['overall']['recall'])
        missing=preannotation_metrics({'x':[fire]},{'x':[]})['overall']
        self.assertEqual(missing['precision'],0);self.assertEqual(missing['recall'],0)

    def test_completion_checks_image_role_pairs_not_just_total_count(self):
        store,key,tokens=self.store();store.submit(tokens['A'],key,[],1,True)
        task=store.access(tokens['admin'])['task_id']
        with store.dataset.connection() as db:
            db.execute('INSERT INTO quality_submissions VALUES(?,?,?,?,?,?)',(task,'unexpected','B','[]',1,'2026-01-01'))
        self.assertFalse(store.completed(store.access(tokens['admin'])))
        with self.assertRaises(ValueError):store.report(tokens['admin'])

    def test_timer_pauses_and_expires(self):
        clock=[0.];timer=WorkTimer(lambda:clock[0]);timer.start();clock[0]=4;timer.activity();clock[0]=6;timer.pause()
        self.assertEqual(timer.seconds(),6)
        clock[0]=100;self.assertEqual(timer.seconds(),6)
        timer.start();clock[0]=200;timer.expire();self.assertIsNone(timer.last);self.assertEqual(timer.seconds(),66)

    def test_switching_image_or_identity_pauses_previous_work(self):
        clock=[0.];first=WorkTimer(lambda:clock[0]);second=WorkTimer(lambda:clock[0])
        first.start();clock[0]=7
        pause_work_timers([{'timer':first},{'timer':second},{'boxes':[]}])
        second.start();clock[0]=12;second.pause()
        self.assertIsNone(first.last)
        self.assertEqual(first.seconds(),7)
        self.assertEqual(second.seconds(),5)

    def test_invalid_access_and_same_person(self):
        store,key,tokens=self.store()
        with self.assertRaises(ValueError):store.access('not-the-token')
        with self.assertRaises(ValueError):store.create([key],'Same','Same','Admin')

    def test_known_mask_error(self):
        a=np.zeros((10,10),np.uint8);a[:5]=255;b=a.copy();b[5]=255
        values=mask_metrics(a,b)
        self.assertAlmostEqual(values['iou'],5/6)
        self.assertAlmostEqual(values['absolute_area_error_percentage_points'],10)

    def test_empty_and_invalid_masks(self):
        empty=np.zeros((10,10),np.uint8)
        self.assertEqual(mask_metrics(empty,empty)['iou'],1)
        self.assertIsNone(mask_metrics(empty,empty)['relative_area_error'])
        with self.assertRaises(ValueError):mask_metrics(empty,np.zeros((5,5)))
        with self.assertRaises(ValueError):mask_metrics(empty,np.full((10,10),128))

    def test_independent_reference_required(self):
        store,key,_=self.store();asset=store.annotations.image(key);mask=np.zeros((asset['height'],asset['width']),np.uint8)
        with self.assertRaises(ValueError):store.compare_masks(key,mask,mask,'tester','simulation')
        record=store.compare_masks(key,mask,mask,'tester','Explicit synthetic mask calculation',True)
        self.assertEqual(record['metrics']['dice'],1)

    def test_timing_pair_requires_same_actor(self):
        store,_,_=self.store();store.worktime('group','A','manual',10,'simulated control');store.worktime('group','B','assisted',5,'simulated control')
        self.assertEqual(store.worktime_report()['paired_estimates'],[])
        store.worktime('group','A','assisted',12,'simulated slower assisted work')
        self.assertAlmostEqual(store.worktime_report()['paired_estimates'][0]['estimated_saved_fraction'],-.2)


if __name__=='__main__':unittest.main()
