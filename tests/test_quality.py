import torch
import unittest
import uuid
from pathlib import Path
import numpy as np
from test_training import seed
from quality import QualityStore,WorkTimer,mask_metrics,pause_work_timers


class QualityTests(unittest.TestCase):
    def store(self):
        root=Path(__file__).resolve().parents[1]/'artifacts/completion_checks'/('quality_'+uuid.uuid4().hex)
        dataset,_,_=seed(root);store=QualityStore(dataset);key=store.annotations.catalog()[0]['id']
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
        store.arbitrate(tokens['admin'],key,[],'Synthetic empty annotation exercise')
        report=store.report(tokens['admin'])
        self.assertEqual(report['images'],1);self.assertFalse(report['at_least_200_images'])

    def test_empty_submission_requires_confirmation(self):
        store,key,tokens=self.store()
        with self.assertRaises(ValueError):store.submit(tokens['A'],key,[],1)

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
