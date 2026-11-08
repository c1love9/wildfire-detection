"""Guard independent-test figures against split leakage and metric mislabelling."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from plot_test_performance import audit_split,check_test_ids,presence_confusion


def split_fixture():
    return [{'kind':'image','id':s,'sha256':'file_'+s,'component':'component_'+s,
             'groups':['group_'+s],'split':s} for s in ('train','val','test')]


class PerformanceFigureTests(unittest.TestCase):
    def test_split_audit_checks_all_images_and_excludes_video_from_image_counts(self):
        rows=split_fixture()+[{'kind':'video','id':'clip','split':'train'}]
        report=audit_split(rows,{'train':1,'val':1,'test':1})
        self.assertEqual(report['images'],3)
        self.assertEqual(report['excluded_videos'],1)
        self.assertEqual(report['known_groups_per_split'],{'train':1,'val':1,'test':1})
        self.assertTrue(all(not any(row.values()) for row in report['cross_split'].values()))

    def test_known_source_leakage_is_rejected_even_with_distinct_image_ids(self):
        rows=split_fixture();rows[2]['groups']=rows[0]['groups']
        with self.assertRaisesRegex(ValueError,'crosses splits'):
            audit_split(rows,{'train':1,'val':1,'test':1})

    def test_validation_predictions_cannot_be_presented_as_test(self):
        comparison={'references':{'val':[]},'predictions':{},'timings_ms':{}}
        with self.assertRaisesRegex(ValueError,'frozen test IDs'):
            check_test_ids(comparison,split_fixture())

    def test_image_presence_accuracy_does_not_claim_box_localization_or_class_correctness(self):
        references={'positive':[{'class':'fire','xyxy':[0,0,2,2]}],'negative':[]}
        # Intentionally wrong class/location still means predicted image presence.
        predictions={'positive':[{'class':'smoke','xyxy':[100,100,102,102],'confidence':.3}],
                     'negative':[{'class':'fire','xyxy':[0,0,2,2],'confidence':.8}]}
        result=presence_confusion(references,predictions,.25)
        self.assertEqual((result['tp'],result['fp'],result['tn'],result['fn']),(1,1,0,0))
        self.assertEqual(result['accuracy'],.5)


if __name__=='__main__':unittest.main()
