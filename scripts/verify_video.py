"""Verify real video inference, media timestamps, controls, EOF and the video page."""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import torch
from PIL import Image
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from video_detection import FireSmokeDetector, VideoSession


def button(app, label):
    return next(widget for widget in app.button if widget.label == label)


def assert_page(app):
    assert not app.exception, [item.message for item in app.exception]
    assert not app.error, [item.value for item in app.error]


def main():
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    cv2.setNumThreads(2)
    output = ROOT / 'artifacts' / 'video_checks'
    output.mkdir(parents=True, exist_ok=True)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    detector = FireSmokeDetector(ROOT / 'fire-models' / 'fire_n.pt', device)
    report = {
        'generated_at': datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
        'scope': 'Functional video tests with upstream pretrained weights; no accuracy claims.',
        'device': device, 'full_videos': [], 'controls': [], 'ui_checks': [],
    }
    seen_classes = set()
    for source in sorted((ROOT / 'demo-videos').glob('*')):
        started = time.perf_counter()
        session = VideoSession(source, detector)
        previous = 0
        previous_time = -1.0
        fallback_frames = 0
        while session.status == 'running':
            frame = session.step()
            if frame is None:
                break
            assert frame.number == previous + 1, 'Frame skipped or repeated'
            if frame.time_estimated:
                fallback_frames += 1
            else:
                assert math.isclose(frame.media_seconds, session.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000, abs_tol=1e-6)
            assert frame.media_seconds > previous_time
            assert frame.rgb.ndim == 3 and frame.rgb.shape[2] == 3
            assert frame.rgb.shape[1] <= 960 and frame.rgb.shape[0] <= 720
            for box in frame.detections:
                assert box['class'] in {'fire', 'smoke'}
                assert 0 <= box['confidence'] <= 1
                x1, y1, x2, y2 = box['xyxy']
                assert 0 <= x1 <= x2 <= session.info.width + 1
                assert 0 <= y1 <= y2 <= session.info.height + 1
                if box['class'] not in seen_classes:
                    Image.fromarray(frame.rgb).save(output / f"video_{box['class']}.jpg")
                    seen_classes.add(box['class'])
            if frame.number == 5:
                session.pause()
                assert session.step() is None and session.processed_frames == 5
                assert session.latest.media_seconds == frame.media_seconds
                session.resume()
                report['controls'].append({'source': source.name, 'pause_resume_preserves_frame_and_media_time': True})
            previous = frame.number
            previous_time = frame.media_seconds
        assert session.status == 'completed', session.error
        assert not session.is_open, 'Video handle was not released at EOF'
        assert session.processed_frames == session.info.frame_count
        report['full_videos'].append({
            'source': source.relative_to(ROOT).as_posix(), 'decoded_and_detected_frames': session.processed_frames,
            'source_frame_count': session.info.frame_count, 'source_fps': session.info.fps,
            'fire_frames': session.fire_frames, 'smoke_frames': session.smoke_frames,
            'last_media_seconds': session.latest.media_seconds, 'elapsed_seconds': round(time.perf_counter() - started, 3),
            'sequential_frames': True, 'media_timestamps_valid': True, 'released_at_eof': True,
            'timestamp_fallback_frames': fallback_frames,
        })
        print(f'{source.name}: {session.processed_frames} frames passed', flush=True)
    assert seen_classes == {'fire', 'smoke'}, 'Videos did not exercise both output classes'
    cpu = FireSmokeDetector(ROOT / 'fire-models' / 'fire_n.pt', 'cpu')
    session = VideoSession(ROOT / 'demo-videos' / 'fire1.mov', cpu)
    for number in range(1, 4):
        assert session.step().number == number
    session.stop()
    assert not session.is_open and session.step() is None
    report['cpu_and_stop_check'] = {'frames': 3, 'released_on_stop': True, 'no_frames_after_stop': True}
    corrupt = output / 'invalid.mp4'
    corrupt.write_bytes(b'not a valid video')
    try:
        VideoSession(corrupt, detector)
    except ValueError:
        report['invalid_video_rejected'] = True
    else:
        raise AssertionError('Invalid video was accepted')
    app = AppTest.from_file(str(next((ROOT / 'pages').glob('3_*.py'))), default_timeout=120).run()
    assert_page(app)
    button(app, '开始检测').click().run()
    assert_page(app)
    active = app.session_state['video_detection_session']
    assert active.processed_frames >= 1
    report['ui_checks'].append('start_displays_real_frame_and_metrics')
    button(app, '暂停检测').click().run()
    assert_page(app)
    paused_frame = active.processed_frames
    app.run()
    assert active.status == 'paused' and active.processed_frames == paused_frame
    report['ui_checks'].append('pause_does_not_advance_video')
    button(app, '继续检测').click().run()
    assert_page(app)
    assert active.processed_frames > paused_frame
    report['ui_checks'].append('resume_continues_from_next_frame')
    button(app, '停止检测').click().run()
    assert_page(app)
    assert active.status == 'stopped' and not active.is_open
    report['ui_checks'].append('stop_releases_video')
    next(widget for widget in app.selectbox if widget.label == '选择样例视频').select('fire3.mp4').run()
    assert_page(app)
    button(app, '开始检测').click().run()
    assert_page(app)
    short = app.session_state['video_detection_session']
    for _ in range(70):
        if short.status != 'running':
            break
        app.run()
        assert_page(app)
    assert short.status == 'completed' and short.processed_frames == 64 and not short.is_open
    assert not button(app, '开始检测').disabled
    report['ui_checks'].append('full_video_eof_updates_controls_and_keeps_last_frame')
    report['status'] = 'passed'
    destination = ROOT / 'reports' / 'video_detection.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'passed', 'total_full_video_frames': sum(item['decoded_and_detected_frames'] for item in report['full_videos']), 'ui_checks': len(report['ui_checks'])}), flush=True)


if __name__ == '__main__':
    main()
