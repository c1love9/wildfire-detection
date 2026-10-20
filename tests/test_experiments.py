import torch
import unittest
from pathlib import Path
import uuid
from test_training import seed
from experiments import ExperimentStore,alarm_ablation
from training import TrainConfig


class ExperimentTests(unittest.TestCase):
    def test_single_factor_pair_and_manifest_tamper(self):
        root=Path(__file__).resolve().parents[1]/'artifacts/completion_checks'/('experiment_'+uuid.uuid4().hex)
        dataset,_,export=seed(root);store=ExperimentStore(dataset)
        key=store.create(export,TrainConfig(epochs=1,imgsz=160,batch=2,device='cpu'),'unit simulation','Synthetic test only')
        manifest=store.read(key);a,b=manifest['configs'].values()
        self.assertEqual([k for k in a if a[k]!=b[k]],['multi_scale'])
        job=store.start(key,'baseline',launch=False)
        with self.assertRaises(ValueError):store.start(key,'baseline',launch=False)
        path=store.root/key/'manifest.json';path.write_text('{}')
        with self.assertRaises(OSError):store.read(key)

    def test_single_frame_spike_vs_confirmation(self):
        box={'class':'fire','confidence':.9,'xyxy':[0,0,10,10]}
        frames=[(i*.1,[box] if i in (2,5,8) else []) for i in range(20)]
        result=alarm_ablation(frames)
        self.assertEqual(result['single_frame_trigger_episodes'],3)
        self.assertEqual(result['continuous_confirmed_events'],0)

    def test_multiscale_type_validation(self):
        with self.assertRaises(ValueError):TrainConfig(multi_scale=1)


if __name__=='__main__':unittest.main()
