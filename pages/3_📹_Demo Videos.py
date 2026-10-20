from pathlib import Path
import base64
import hashlib
from io import BytesIO
import logging
from dataclasses import replace

import torch
from PIL import Image
import streamlit as st

from video_detection import VideoSession
from alarm import AlarmConfig, LEVEL_NAMES, CLASS_NAMES
from alarm_log import alarm_log_path
from recording import RecordingConfig
from camera_source import CameraConfig
from camera_session import CameraSession
from monitoring_ui import edit_monitoring_point, show_location
from dataset_store import dataset_root
from model_catalog import fire_models
from inference_ui import load_detector, show_model


ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = alarm_log_path(ROOT)
SESSION_KEY = 'video_detection_session'
STATUS = {'running': '正在检测', 'paused': '已暂停', 'completed': '检测完成', 'stopped': '已停止', 'error': '检测失败', 'disconnected': '摄像头观测中断'}


def save_upload(upload) -> Path:
    content = upload.getbuffer()
    digest = hashlib.sha256(content).hexdigest()
    suffix = Path(upload.name).suffix.lower()
    if suffix not in {'.mp4', '.mov', '.avi', '.mkv'}:
        raise ValueError('请选择MP4、MOV、AVI或MKV视频。')
    destination = ROOT / 'artifacts' / 'uploads' / (digest + suffix)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        destination.write_bytes(content)
    return destination


def format_time(seconds: float) -> str:
    minutes, seconds = divmod(seconds, 60)
    return f'{int(minutes):02d}:{seconds:05.2f}'


st.set_page_config(page_title='视频火烟检测', page_icon='📹', layout='wide')
st.title('视频火焰与烟雾检测')
st.caption('使用YOLOv8逐帧检测，画面中的fire代表火焰、smoke代表烟雾，框旁数字为置信度。')

session = st.session_state.get(SESSION_KEY)
locked = session is not None and session.status in {'running', 'paused'}
source_mode = st.radio('视频来源', ['内置样例', '上传视频', '摄像头'], horizontal=True, disabled=locked)
source_path = None
source_name = None
camera_config = None
if source_mode == '内置样例':
    samples = sorted((ROOT / 'demo-videos').glob('*'))
    if samples:
        sample_name = st.selectbox('选择样例视频', [path.name for path in samples], disabled=locked)
        source_path = ROOT / 'demo-videos' / sample_name
        source_name = sample_name
        st.caption('部分内置样例自带旧检测框，本次检测结果以当前帧数量和“当前帧检测明细”为准。')
    else:
        st.info('没有找到内置视频，可以切换到“上传视频”。')
elif source_mode == '上传视频':
    upload = st.file_uploader('上传待检测视频', type=['mp4', 'mov', 'avi', 'mkv'], disabled=locked)
    if upload is not None:
        source_path = save_upload(upload)
        source_name = upload.name
else:
    try:
        default_camera = CameraConfig.load(ROOT / 'configs' / 'camera.json')
        camera_col, backend_col = st.columns(2)
        camera_index = camera_col.number_input('摄像头编号', min_value=0, max_value=9, value=default_camera.index, step=1, disabled=locked)
        backend_names = {'AUTO': '自动', 'DSHOW': 'DirectShow', 'MSMF': 'Media Foundation'}
        backend = backend_col.selectbox('采集后端', list(backend_names), index=list(backend_names).index(default_camera.backend),
                                       format_func=backend_names.get, disabled=locked)
        camera_config = replace(default_camera, index=int(camera_index), backend=backend)
        source_path, source_name = camera_index, f'摄像头 {camera_index}'
        st.caption('使用运行本项目电脑连接的摄像头。默认编号0；点击开始后才打开设备。暂停时关闭设备，继续时创建新观测段并重新确认。')
        st.caption('实时输入只检测最新采样画面，避免积压；离开页面且长时间不刷新时会自动关闭设备。')
    except (OSError, ValueError, TypeError) as exc:
        st.error(f'摄像头配置无效：{exc}')

monitoring_point = None
monitoring_config_valid = True
try:
    monitoring_point = edit_monitoring_point(ROOT / 'configs' / 'monitoring_points.json', locked)
except (OSError, ValueError, TypeError, KeyError) as exc:
    monitoring_config_valid = False
    st.error(f'监控点配置无效：{exc}')
if monitoring_point:
    st.caption(f'当前关联：{monitoring_point.name}（模拟位置与视角，非设备实测）。')

model_col, device_col, conf_col = st.columns(3)
models = st.session_state.get('video_model_choices') if locked else fire_models(ROOT, dataset_root(ROOT))
models = models or fire_models(ROOT, dataset_root(ROOT))
with model_col:
    model_names = list(models)
    if not model_names:
        st.error('缺少火烟模型，请按运行说明取回fire_n.pt。')
        st.stop()
    selected_model = st.selectbox('检测模型', model_names, index=model_names.index('fire_n') if 'fire_n' in model_names else 0,
                                  format_func=lambda key: models[key].label, disabled=locked)
choice = models[selected_model]
with device_col:
    devices = ['CPU'] + (['GPU (CUDA)'] if torch.cuda.is_available() else [])
    device_label = st.selectbox('计算设备', devices, index=len(devices) - 1, disabled=locked)
    device = 'cuda:0' if device_label == 'GPU (CUDA)' else 'cpu'
with conf_col:
    confidence = st.slider('置信度阈值', 0.05, 0.95, 0.20, 0.05, disabled=locked)
show_model(choice)
st.caption('图片和视频页可分别选择模型；视频/摄像头运行或暂停时模型锁定，停止后可切换并开始新观测。')
with st.expander('检测设置'):
    sizes = sorted({160, 320, 640, 960, 1280, choice.imgsz})
    imgsz = st.selectbox('推理图像尺寸', sizes, index=sizes.index(choice.imgsz), disabled=locked)
    iou = st.slider('重叠框抑制阈值（NMS IoU）', 0.1, 0.9, 0.5, 0.05, disabled=locked)
    st.caption('视频文件逐帧检测，摄像头检测最新采样帧。持续时间使用素材时间或实时采集时间，不使用页面刷新次数。')

try:
    default_alarm = AlarmConfig.load(ROOT / 'configs' / 'alarm.json')
except (OSError, ValueError, TypeError) as exc:
    st.error(f'无法读取报警配置：{exc}')
    st.stop()
try:
    default_recording = RecordingConfig.load(ROOT / 'configs' / 'recording.json')
except (OSError, ValueError, TypeError) as exc:
    st.error(f'无法读取录制配置：{exc}')
    st.stop()
with st.expander('报警规则'):
    st.caption('同一类别、同一区域同时满足连续帧数和持续时间才确认；烟雾可独立触发。以下为课程原型调试参数。')
    frames_col, fire_threshold_col, smoke_threshold_col = st.columns(3)
    confirm_frames = frames_col.number_input('连续确认帧数', min_value=2, value=default_alarm.confirm_frames, step=1, disabled=locked)
    fire_threshold = fire_threshold_col.number_input('火焰报警置信度', min_value=0.0, max_value=1.0, value=float(default_alarm.fire_confidence), step=0.05, disabled=locked)
    smoke_threshold = smoke_threshold_col.number_input('烟雾报警置信度', min_value=0.0, max_value=1.0, value=float(default_alarm.smoke_confidence), step=0.05, disabled=locked)
    attention_col, warning_col, alarm_col, recovery_col = st.columns(4)
    attention_seconds = attention_col.number_input('关注持续时间（秒）', min_value=0.0, value=float(default_alarm.attention_seconds), step=0.5, disabled=locked)
    warning_seconds = warning_col.number_input('预警持续时间（秒）', min_value=0.0, value=float(default_alarm.warning_seconds), step=0.5, disabled=locked)
    alarm_seconds = alarm_col.number_input('报警持续时间（秒）', min_value=0.0, value=float(default_alarm.alarm_seconds), step=0.5, disabled=locked)
    recovery_seconds = recovery_col.number_input('恢复等待（秒）', min_value=0.0, value=float(default_alarm.recovery_seconds), step=0.5, disabled=locked)
    st.caption(f'区域匹配IoU ≥ {default_alarm.match_iou:g}，相邻检测帧时间间隔 ≤ {default_alarm.max_frame_gap_seconds:g} 秒。摄像头超过该间隔将结束观测，避免跨未观察区间确认或恢复。')
    st.caption('日志保存在本机，侧边栏“Alarm History”可查询和导出。')

with st.expander('报警截图与片段'):
    recording_enabled = st.checkbox('保存报警截图和片段', value=default_recording.enabled, disabled=locked)
    pre_col, post_col = st.columns(2)
    pre_seconds = pre_col.number_input('确认前保存（秒）', min_value=0.0, max_value=30.0, value=float(default_recording.pre_seconds), step=1.0, disabled=locked)
    post_seconds = post_col.number_input('确认后保存（秒）', min_value=0.0, max_value=60.0, value=float(default_recording.post_seconds), step=1.0, disabled=locked)
    st.caption('首次连续确认（关注级）时保存截图，每个事件生成一段带检测框的静音视频。开始或结束处时间不足、提前停止时会标明实际范围。')

alarm_config = None
recording_config = None
try:
    alarm_config = replace(default_alarm, confirm_frames=int(confirm_frames), fire_confidence=fire_threshold,
                           smoke_confidence=smoke_threshold, attention_seconds=attention_seconds,
                           warning_seconds=warning_seconds, alarm_seconds=alarm_seconds, recovery_seconds=recovery_seconds)
    recording_config = replace(default_recording, enabled=recording_enabled, pre_seconds=pre_seconds, post_seconds=post_seconds)
except ValueError as exc:
    st.warning(str(exc))
if alarm_config is not None and confidence > min(alarm_config.fire_confidence, alarm_config.smoke_confidence):
    st.warning('检测置信度高于部分报警阈值，低于检测阈值的框不会进入报警判定。')

signature = (source_mode, str(source_path), selected_model, choice.sha256, imgsz, device, confidence, iou, alarm_config, recording_config, camera_config, monitoring_point)
if st.session_state.get('video_input_signature') != signature:
    st.session_state['video_input_signature'] = signature
    st.session_state.pop('video_detection_error', None)
if session is not None and st.session_state.get('video_detection_signature') != signature:
    session.stop()
    st.session_state.pop(SESSION_KEY, None)
    session = None
    locked = False

start_col, pause_col, stop_col = st.columns(3)
with start_col:
    if st.button('开始检测', type='primary', disabled=locked or bool(choice.error) or source_path is None or alarm_config is None or recording_config is None or not monitoring_config_valid, width='stretch'):
        try:
            if session is not None:
                session.stop()
            with st.spinner('正在加载模型并打开输入…'):
                detector = load_detector(choice, device, imgsz)
                if source_mode == '摄像头':
                    st.session_state[SESSION_KEY] = CameraSession(camera_config, detector, confidence, iou,
                                                                  alarm_config=alarm_config, log_path=LOG_PATH,
                                                                  source_name=source_name, recording_config=recording_config,
                                                                  monitoring_point=monitoring_point)
                else:
                    st.session_state[SESSION_KEY] = VideoSession(source_path, detector, confidence, iou,
                                                                alarm_config=alarm_config, log_path=LOG_PATH,
                                                                source_name=source_name, recording_config=recording_config,
                                                                monitoring_point=monitoring_point)
            st.session_state['video_detection_signature'] = signature
            st.session_state['video_model_choices'] = models
            st.session_state['video_detection_name'] = source_name
            st.session_state.pop('video_detection_error', None)
            st.rerun()
        except (ValueError, OSError, RuntimeError) as exc:
            st.session_state['video_detection_error'] = str(exc)
with pause_col:
    pause_label = '继续检测' if session is not None and session.status == 'paused' else '暂停检测'
    if st.button(pause_label, disabled=not locked, width='stretch'):
        try:
            if session.status == 'paused':
                session.resume()
            else:
                session.pause()
            st.session_state.pop('video_detection_error', None)
            st.rerun()
        except (ValueError, OSError, RuntimeError) as exc:
            st.session_state['video_detection_error'] = str(exc)
            st.rerun()
with stop_col:
    if st.button('停止检测', disabled=not locked, width='stretch'):
        session.stop()
        st.rerun()

if st.session_state.get('video_detection_error'):
    st.error(st.session_state['video_detection_error'])


@st.fragment(run_every=0.1 if session is not None and session.status == 'running' else None)
def show_detection():
    active_session = st.session_state.get(SESSION_KEY)
    if active_session is None:
        st.info('选择视频或摄像头后点击“开始检测”，这里会显示带框检测画面。')
        return
    if active_session.status == 'running':
        try:
            active_session.step()
        except Exception:
            logging.getLogger(__name__).exception('Video inference failed')
    if active_session.status in {'completed', 'error', 'disconnected'} and locked:
        st.rerun()
    st.caption(f"{st.session_state['video_detection_name']} · {STATUS[active_session.status]}")
    identity = getattr(active_session.detector, 'metadata', {})
    st.caption(f"本次观测模型：{identity.get('label', selected_model)} · {identity.get('sha256', '')[:12]} · {identity.get('imgsz', imgsz)}像素")
    alarms = active_session.alarms
    alarm_state = alarms.snapshot
    labels = '、'.join(CLASS_NAMES[label] for label in alarm_state.classes)
    level_message = f'当前报警等级：{LEVEL_NAMES[alarm_state.level]}' + (f' · {labels}' if labels else '')
    if active_session.status != 'running':
        level_message += '（检测已暂停，保留最后判定）' if active_session.status == 'paused' else '（检测已结束，保留最后判定）'
    {0: st.success, 1: st.info, 2: st.warning, 3: st.error}[alarm_state.level](level_message)
    is_camera = active_session.input_type == 'camera'
    st.caption(f'当前区域连续命中 {alarm_state.pending_frames} 帧 · 持续 {alarm_state.pending_seconds:.2f} 秒 · 本段确认事件 {alarms.event_count} 个')
    if alarms.log_error:
        st.warning(alarms.log_error)
    if active_session.recorder and active_session.recorder.last_error:
        st.warning(active_session.recorder.last_error)
    latest = active_session.latest
    if latest is not None:
        frame_col, time_col, fire_col, smoke_col = st.columns(4)
        frame_col.metric('本段已检测帧' if is_camera else '当前帧', latest.number if is_camera else f'{latest.number} / {active_session.info.frame_count or "未知"}')
        time_col.metric('本段观测时间' if is_camera else '视频时间（估计）' if latest.time_estimated else '视频时间', format_time(latest.media_seconds))
        fire_col.metric('火焰', latest.counts.get('fire', 0))
        smoke_col.metric('烟雾', latest.counts.get('smoke', 0))
        # Keep each image in the same update as its metrics. Separate media URLs
        # can expire while an earlier browser request is still in flight.
        encoded = BytesIO()
        Image.fromarray(latest.rgb).save(encoded, format='JPEG', quality=85)
        image_url = 'data:image/jpeg;base64,' + base64.b64encode(encoded.getvalue()).decode('ascii')
        st.image(image_url, caption='逐帧检测结果', width='content')
        st.caption(f'处理速度 {active_session.processing_fps:.1f} 帧/秒 · 本帧推理 {latest.inference_ms:.1f} ms · {"设备报告" if is_camera else "源视频"} {active_session.info.fps:.2f} 帧/秒')
        if is_camera:
            st.caption(f'观测段 {active_session.segment_number} · 跳过已过时采集帧 {active_session.dropped_frames} 帧' +
                       (f' · 当前画面距采集 {active_session.frame_age_seconds:.2f} 秒' if active_session.status == 'running' else ' · 保留最后画面'))
        if active_session.info.frame_count:
            st.progress(min(1.0, latest.number / active_session.info.frame_count))
        with st.expander('当前帧检测明细'):
            rows = [{'类别': {'fire': '火焰', 'smoke': '烟雾'}.get(box['class'], box['class']), '置信度': round(box['confidence'], 3), '原图边界框（像素）': ', '.join(str(round(value)) for value in box['xyxy'])} for box in latest.detections]
            if rows:
                st.dataframe(rows, hide_index=True, width='stretch')
            else:
                st.write('当前帧未检出火焰或烟雾。')
    if active_session.info.fps_estimated:
        st.info('设备未报告有效帧率，显示值按请求帧率估计。' if is_camera else '未能读取源视频帧率，帧率显示值按30帧/秒估计。')
    if active_session.status == 'completed':
        st.success(f'检测完成：共处理{active_session.processed_frames}帧；含火焰{active_session.fire_frames}帧，含烟雾{active_session.smoke_frames}帧。')
    elif active_session.status in {'error', 'disconnected'}:
        st.error(active_session.error)
    with st.expander('本次报警变化（最近50条）'):
        changes = list(alarms.recent_transitions)
        if changes:
            st.dataframe([{'事件编号': item['event_id'][:12], '帧号': item['frame'],
                           '视频时间（秒）': round(item['media_seconds'], 3),
                           '等级变化': f"{LEVEL_NAMES[item['from_level']]} → {LEVEL_NAMES[item['to_level']]}",
                           '原因': {'confirmed': '连续确认', 'escalated': '升级', 'downgraded': '降级',
                                    'recovered': '持续消失后恢复', 'completed': '视频结束，未判定恢复',
                                    'stopped': '手动停止，未判定恢复', 'error': '检测异常，未判定恢复',
                                    'paused': '暂停观测，未判定恢复', 'disconnected': '摄像头观测中断，未判定恢复'}[item['kind']]}
                          for item in reversed(changes)], hide_index=True, width='stretch')
        else:
            st.write('尚未确认火烟事件；单帧检出不会直接生成报警日志。')


show_detection()


# The map is outside the inference fragment: panning/zooming never counts as
# a sampled frame and the browser keeps its map while the image refreshes.
@st.fragment(run_every=1.0 if session is not None and session.status == 'running' else None)
def show_monitoring_map():
    active = st.session_state.get(SESSION_KEY)
    point = active.monitoring_point if active is not None else monitoring_point
    level = active.alarms.snapshot.level if active is not None else 0
    show_location(point, level, key='live_monitor_map')


show_monitoring_map()
