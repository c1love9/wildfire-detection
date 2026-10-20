import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from alarm import AlarmConfig
from alarm_log import AlarmLog
from inference_result import FrameResult
from trends import TrendRecorder, trend_rows


class TrendTests(unittest.TestCase):
    def store(self):
        path = Path(__file__).resolve().parents[1] / 'artifacts/completion_checks' / (uuid.uuid4().hex + '.sqlite3')
        store = AlarmLog(path); store.start_run('test', 'fixture', AlarmConfig(), {})
        return store

    def frame(self, number, seconds, result=None):
        return SimpleNamespace(number=number, media_seconds=seconds, result=result, inference_ms=1.)

    def test_sample_cadence_and_flush(self):
        store = self.store(); recorder = TrendRecorder(store, 'test')
        for index in range(15): recorder.add(self.frame(index+1,index/10),0)
        self.assertEqual(len(store.samples('test')),2)
        recorder.flush(); recorder.flush()
        self.assertEqual(len(AlarmLog(store.path).samples('test')),3)
        self.assertIsNone(store.samples('test')[0]['combined'])

    def test_level_changes_are_sampled(self):
        recorder=TrendRecorder(self.store(),'test')
        recorder.add(self.frame(1,0),0);recorder.add(self.frame(2,.1),1)
        self.assertEqual(len(recorder.recent),2)

    def test_rate_and_unknown(self):
        rows=[dict(frame=i+1,media_seconds=i,fire=x,smoke=0,combined=x,level=0,status='ready',elapsed_ms=1) for i,x in enumerate([.1,.2,None,.3])]
        data=trend_rows(rows)
        self.assertAlmostEqual(data[1]['画面占比变化率（百分点/秒）'],10)
        self.assertIsNone(data[3]['画面占比变化率（百分点/秒）'])

    def test_live_window_is_bounded(self):
        recorder=TrendRecorder(None,'test')
        for index in range(350): recorder.add(self.frame(index+1,index),0)
        self.assertEqual(len(recorder.recent),300)


if __name__ == '__main__': unittest.main()
