import unittest
import uuid
import json
from pathlib import Path
from channels import load_channels, save_channels, ROOT
from alarm_log import AlarmLog, AlarmMonitor, event_rows
from alarm import AlarmConfig


class ChannelTests(unittest.TestCase):
    def test_default_bindings(self):
        rows=load_channels()
        self.assertEqual(len(rows),3)
        self.assertEqual(rows[0]['point_id'],'demo_a')
        self.assertEqual(rows[1]['point_id'],'demo_b')

    def test_validation_keeps_original(self):
        path=ROOT/'artifacts/completion_checks'/('channels_'+uuid.uuid4().hex+'.json')
        rows=load_channels();save_channels(rows,path);before=path.read_bytes()
        rows[0]['source']='../outside.mp4'
        with self.assertRaises(ValueError):save_channels(rows,path)
        self.assertEqual(path.read_bytes(),before)

    def test_channel_snapshot_and_filter(self):
        path=ROOT/'artifacts/completion_checks'/(uuid.uuid4().hex+'.sqlite3')
        channel=load_channels()[0]
        monitor=AlarmMonitor(AlarmConfig(),'fixture',path,{'channel':channel})
        for i in range(10):monitor.update(i+1,i*.1,[{'class':'fire','confidence':.9,'xyxy':[0,0,10,10]}])
        monitor.finish('stopped')
        channel['name']='changed now'
        store=AlarmLog(path);events=store.events(channel_id='channel_a')
        self.assertEqual(len(events),1)
        self.assertEqual(event_rows(events)[0]['通道名称'],'监控通道 A')
        self.assertEqual(store.events(channel_id='channel_b'),[])


if __name__=='__main__':unittest.main()
