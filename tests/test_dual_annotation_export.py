"""Submission packaging must retain real versions and reject unfinished work."""
import json
import sys
import unittest
import uuid
from pathlib import Path
from zipfile import ZipFile
import torch
from test_training import seed
from quality import QualityStore

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from export_dual_annotation import export_task
from prepare_dual_annotation import has_grounding_parameters
from grounding_prelabel import PrelabelConfig
from dataclasses import asdict


class DualExportTests(unittest.TestCase):
    def test_source_empty_labels_are_not_counted_as_actual_grounding_inference(self):
        self.assertFalse(has_grounding_parameters(None))
        self.assertFalse(has_grounding_parameters({'metadata':{'method':'source empty labels','runtime':None}}))
        parameters=asdict(PrelabelConfig())
        self.assertTrue(has_grounding_parameters({'metadata':{'parameters':parameters}}))
        self.assertTrue(has_grounding_parameters({'metadata':{'runtime':{'parameters':parameters}}}))
        parameters['revision']='another-version'
        self.assertFalse(has_grounding_parameters({'metadata':{'parameters':parameters}}))

    def task(self):
        root=ROOT/'artifacts/dual_annotation_checks'/uuid.uuid4().hex
        dataset,_,_=seed(root);store=QualityStore(dataset);key=store.annotations.catalog()[0]['id']
        task,tokens=store.create([key],'Synthetic A','Synthetic B','Synthetic arbitration')
        private=root/'task_private.json';private.write_text(json.dumps({'task_id':task,'tokens':tokens,'dataset_root':str(root)}),encoding='utf-8')
        return root,store,key,tokens,private

    def test_unfinished_task_creates_no_submission_package(self):
        root,store,key,tokens,private=self.task();target=root/'deliveries'
        with self.assertRaises(ValueError):export_task(private,target,minimum_images=1)
        self.assertFalse(target.exists())
        for role in ['A','B']:store.submit(tokens[role],key,[],1,True)
        with self.assertRaises(ValueError):export_task(private,target,minimum_images=1)
        self.assertFalse(target.exists())

    def test_export_preserves_independent_and_adjudicated_labels_and_excludes_codes(self):
        root,store,key,tokens,private=self.task()
        store.submit(tokens['A'],key,[{'class':'fire','xyxy':[1,1,8,8]}],2)
        store.submit(tokens['B'],key,[],3,True)
        store.arbitrate(tokens['admin'],key,[{'class':'smoke','xyxy':[2,2,9,9]}],'Synthetic changed-class adjudication')
        reference=store.reference(tokens['admin'],key)
        with self.assertRaises(ValueError):export_task(private,root/'deliveries',minimum_images=1)
        store.approve_reference(tokens['admin'],key,reference['seq'],store.annotations.latest(key)['id'])
        result=export_task(private,root/'deliveries',minimum_images=1)
        with ZipFile(result['zip']) as package:
            self.assertTrue(package.read(f'labels_A/{key}.txt').startswith(b'0 '))
            self.assertEqual(package.read(f'labels_B/{key}.txt'),b'')
            self.assertTrue(package.read(f'labels_adjudicated/{key}.txt').startswith(b'1 '))
            self.assertFalse(json.loads(package.read('task_summary.json'))['at_least_200_images'])
            for name in package.namelist():
                if name.endswith(('.json','.csv','.txt')):
                    for token in tokens.values():self.assertNotIn(token.encode(),package.read(name))
            self.assertIsNone(package.testzip())


if __name__=='__main__':unittest.main()
