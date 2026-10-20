from dataclasses import replace
import unittest
import io
import wave
from alarm import AlarmEngine, AlarmConfig
from alerts import notification_key, tone


class AreaAlarmTests(unittest.TestCase):
    def boxes(self, area):
        return [{'class':'fire','confidence':.9,'xyxy':[0,0,10,10],'area_ratio':area}]

    def test_area_escalates_after_confirmation(self):
        engine=AlarmEngine()
        for i in range(21): state=engine.update(i+1,i*.1,self.boxes(.15))
        self.assertEqual(state.level,3)

    def test_single_frame_does_not_escalate(self):
        engine=AlarmEngine()
        self.assertEqual(engine.update(1,0,self.boxes(.9)).level,0)
        self.assertEqual(engine.update(2,.1,[]).level,0)

    def test_missing_mask_breaks_area_confirmation(self):
        engine=AlarmEngine(replace(AlarmConfig(),warning_seconds=10,alarm_seconds=20))
        for i in range(20): state=engine.update(i+1,i*.1,self.boxes(.5 if i%2==0 else None))
        self.assertEqual(state.level,1)

    def test_small_and_unknown_still_use_duration(self):
        for area in [None, .001]:
            engine=AlarmEngine()
            for i in range(51): state=engine.update(i+1,i*.1,self.boxes(area))
            self.assertEqual(state.level,3)

    def test_region_change_does_not_share_area_timer(self):
        engine=AlarmEngine(replace(AlarmConfig(),warning_seconds=10,alarm_seconds=20))
        for i in range(20):
            boxes=self.boxes(.5);boxes[0]['xyxy']=[i*100,0,i*100+10,10]
            state=engine.update(i+1,i*.1,boxes)
        self.assertEqual(state.level,0)

    def test_sound_and_notification_identity(self):
        self.assertNotEqual(notification_key('r','e',1),notification_key('r','e',2))
        with wave.open(io.BytesIO(tone())) as data:
            self.assertEqual(data.getnframes(),6400)


if __name__=='__main__':unittest.main()
