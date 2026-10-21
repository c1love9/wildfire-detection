import torch
import unittest
import tempfile
from pathlib import Path
from dataset_store import DatasetStore, file_sha
from comparison import ComparisonStore
from training import write_json
from PIL import Image
from comparison import evaluate_predictions,letterbox,transform_boxes


class ComparisonTests(unittest.TestCase):
    def box(self,label='fire',confidence=.9):return {'class':label,'confidence':confidence,'xyxy':[10,10,30,30]}

    def test_perfect_and_absent_class(self):
        result=evaluate_predictions({'a':[self.box()]},{'a':[self.box()]})
        self.assertEqual(result['overall']['precision'],1)
        self.assertGreater(result['overall']['map50'],.99)
        self.assertIsNone(result['per_class'][1]['map50'])

    def test_duplicates_are_false_positives(self):
        row=evaluate_predictions({'a':[self.box()]},{'a':[self.box(),self.box()]})['per_class'][0]
        self.assertEqual((row['tp'],row['fp'],row['fn']),(1,1,0))

    def test_wrong_class_and_empty_prediction(self):
        result=evaluate_predictions({'a':[self.box()]},{'a':[self.box('smoke')]})
        self.assertEqual(result['overall']['recall'],0)
        self.assertEqual(result['per_class'][1]['fp'],1)
        self.assertEqual(evaluate_predictions({'a':[self.box()]},{'a':[]})['overall']['map50'],0)

    def test_fixed_confidence_and_negative_image(self):
        result=evaluate_predictions({'a':[self.box()],'b':[]},{'a':[self.box(confidence=.1)],'b':[self.box()]})
        self.assertEqual(result['overall']['recall'],0)
        self.assertEqual(result['negative_image_false_positive_rate'],1)

    def test_letterbox_coordinate_rounding(self):
        image,transform=letterbox(Image.new('RGB',(1000,333)))
        self.assertEqual(image.size,(640,640))
        box=transform_boxes([{'class':'fire','xyxy':[0,0,1000,333]}],transform)[0]['xyxy']
        self.assertEqual(box[0],0);self.assertEqual(box[2],640)
        self.assertEqual(box[3]-box[1],213)

    def test_history_rejects_changed_frozen_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset=DatasetStore(Path(temporary));store=ComparisonStore(dataset)
            identifier='compare_test';directory=store.root/identifier;directory.mkdir()
            image=directory/'input.png';image.write_bytes(b'original')
            write_json(directory/'request.json',{'id':identifier})
            write_json(directory/'result.json',{'inventory':{'input.png':file_sha(image)}})
            with dataset.connection() as db:
                db.execute('INSERT INTO comparisons VALUES(?,?,?,?,?)',(identifier,'2026-10-21',file_sha(directory/'request.json'),file_sha(directory/'result.json'),'completed'))
            self.assertEqual(store.result(identifier)['inventory']['input.png'],file_sha(image))
            image.write_bytes(b'changed')
            with self.assertRaises(OSError):store.result(identifier)


if __name__=='__main__':unittest.main()
