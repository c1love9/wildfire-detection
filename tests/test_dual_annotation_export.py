"""Submission packaging must retain real versions and reject unfinished work."""
import json
import sys
import unittest
import uuid
import hashlib
from pathlib import Path
from zipfile import ZipFile
import torch
from test_training import seed
from quality import QualityStore
from quality_export import quality_archive

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
        task,tokens=store.create([key],'Synthetic A','Synthetic B','Synthetic manager')
        private=root/'task_private.json';private.write_text(json.dumps({'task_id':task,'tokens':tokens,'dataset_root':str(root)}),encoding='utf-8')
        return root,store,key,tokens,private

    def test_unfinished_task_creates_no_submission_package(self):
        root,store,key,tokens,private=self.task();target=root/'deliveries'
        with self.assertRaises(ValueError):export_task(private,target,minimum_images=1)
        self.assertFalse(target.exists())
        store.submit(tokens['A'],key,[],1,True)
        with self.assertRaises(ValueError):export_task(private,target,minimum_images=1,selection='A')
        self.assertFalse(target.exists())

    def test_export_preserves_both_labels_without_references_or_review_approval(self):
        root,store,key,tokens,private=self.task()
        store.submit(tokens['A'],key,[{'class':'fire','xyxy':[1,1,8,8]}],2)
        store.submit(tokens['B'],key,[],3,True)
        with store.dataset.connection() as db:db.execute('DELETE FROM annotation_reviews')
        originals=store.submissions(tokens['admin'])
        result=export_task(private,root/'deliveries',minimum_images=1)
        with ZipFile(result['zip']) as package:
            self.assertTrue(package.read(f'labels_A/{key}.txt').startswith(b'0 '))
            self.assertEqual(package.read(f'labels_B/{key}.txt'),b'')
            self.assertFalse(any('adjudicated' in name for name in package.namelist()))
            self.assertIn('annotation_quality.json',package.namelist())
            self.assertEqual(len(json.loads(package.read('annotation_quality.json'))['independent_submissions']),2)
            self.assertFalse(json.loads(package.read('task_summary.json'))['at_least_200_images'])
            for name in package.namelist():
                if name.endswith(('.json','.csv','.txt')):
                    for token in tokens.values():self.assertNotIn(token.encode(),package.read(name))
            self.assertIsNone(package.testzip())
            checks=json.loads(package.read('checksums.json'))
            self.assertEqual(set(checks),set(package.namelist())-{'checksums.json'})
            for name,record in checks.items():self.assertEqual(hashlib.sha256(package.read(name)).hexdigest(),record['sha256'])
        self.assertEqual(store.submissions(tokens['admin']),originals)
        with store.dataset.connection() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM annotation_reviews').fetchone()[0],0)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM quality_references').fetchone()[0],0)

    def test_single_role_exports_exclude_other_role_and_quality_report(self):
        root,store,key,tokens,private=self.task()
        store.submit(tokens['A'],key,[{'class':'fire','xyxy':[1,1,8,8]}],2)
        store.submit(tokens['B'],key,[],3,True)
        for role in ('A','B'):
            result=export_task(private,root/'deliveries',minimum_images=1,selection=role)
            with ZipFile(result['zip']) as package:
                names=package.namelist();other='B' if role=='A' else 'A'
                self.assertIn(f'labels_{role}/{key}.txt',names)
                self.assertNotIn(f'independent_{other}.json',names)
                self.assertNotIn('annotation_quality.json',names)
                self.assertFalse(any(name.startswith(f'labels_{other}/') for name in names))
                summary=json.loads(package.read('task_summary.json'))
                self.assertEqual(summary['selection'],role)
                self.assertEqual(summary['independent_submissions'],1)
                self.assertFalse(summary['training_labels_approved_by_export'])

    def test_export_retains_historical_reference_only_as_optional_legacy_file(self):
        root,store,key,tokens,private=self.task()
        for role in ('A','B'):store.submit(tokens[role],key,[],1,True)
        with store.dataset.connection() as db:
            db.execute('INSERT INTO quality_references(task_id,asset_id,boxes_json,actor,note,created_at) VALUES(?,?,?,?,?,?)',
                       (store.access(tokens['admin'])['task_id'],key,'[]','legacy fixture','Old saved record','2026-01-01'))
        result=export_task(private,root/'deliveries',minimum_images=1)
        with ZipFile(result['zip']) as package:
            self.assertEqual(len(json.loads(package.read('legacy_reference_history.json'))),1)
            self.assertEqual(json.loads(package.read('annotation_quality.json'))['agreement_summary']['both_empty_images'],1)

    def test_export_rejects_role_credentials_invalid_selection_and_changed_images(self):
        root,store,key,tokens,private=self.task()
        for role in ('A','B'):store.submit(tokens[role],key,[],1,True)
        with self.assertRaises(ValueError):quality_archive(store,tokens['A'],'A')
        with self.assertRaises(ValueError):quality_archive(store,tokens['admin'],'all')
        with self.assertRaises(ValueError):export_task(private,root/'deliveries')
        store.dataset.asset_path(store.annotations.image(key)).write_bytes(b'changed fixture')
        with self.assertRaises(OSError):export_task(private,root/'deliveries',minimum_images=1)
        self.assertFalse((root/'deliveries').exists())


if __name__=='__main__':unittest.main()
