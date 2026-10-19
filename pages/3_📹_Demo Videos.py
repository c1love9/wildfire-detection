from pathlib import Path
import base64
import hashlib
from io import BytesIO
import logging

from PIL import Image
import streamlit as st
import torch

from video_detection import FireSmokeDetector, VideoSession


ROOT = Path(__file__).resolve().parents[1]
SESSION_KEY = 'video_detection_session'
STATUS = {'running': '正在检测', 'paused': '已暂停', 'completed': '检测完成', 'stopped': '已停止', 'error': '检测失败'}


@st.cache_resource
def load_video_detector(weights: str, device: str):
    return FireSmokeDetector(weights, device)


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
source_mode = st.radio('视频来源', ['内置样例', '上传视频'], horizontal=True, disabled=locked)
source_path = None
source_name = None
if source_mode == '内置样例':
    samples = sorted((ROOT / 'demo-videos').glob('*'))
    if samples:
        sample_name = st.selectbox('选择样例视频', [path.name for path in samples], disabled=locked)
        source_path = ROOT / 'demo-videos' / sample_name
        source_name = sample_name
        st.caption('部分内置样例自带旧检测框，本次检测结果以当前帧数量和“当前帧检测明细”为准。')
    else:
        st.info('没有找到内置视频，可以切换到“上传视频”。')
else:
    upload = st.file_uploader('上传待检测视频', type=['mp4', 'mov', 'avi', 'mkv'], disabled=locked)
    if upload is not None:
        source_path = save_upload(upload)
        source_name = upload.name

model_col, device_col, conf_col = st.columns(3)
models = sorted((ROOT / 'fire-models').glob('*.pt'))
with model_col:
    model_names = [path.stem for path in models]
    if not model_names:
        st.error('缺少火烟模型，请按运行说明取回fire_n.pt。')
        st.stop()
    selected_model = st.selectbox('检测模型', model_names, index=model_names.index('fire_n') if 'fire_n' in model_names else 0, disabled=locked)
with device_col:
    devices = ['CPU'] + (['GPU (CUDA)'] if torch.cuda.is_available() else [])
    device_label = st.selectbox('计算设备', devices, index=len(devices) - 1, disabled=locked)
    device = 'cuda:0' if device_label == 'GPU (CUDA)' else 'cpu'
with conf_col:
    confidence = st.slider('置信度阈值', 0.05, 0.95, 0.20, 0.05, disabled=locked)
with st.expander('检测设置'):
    iou = st.slider('重叠框抑制阈值（NMS IoU）', 0.1, 0.9, 0.5, 0.05, disabled=locked)
    st.caption('每一帧都会参与检测。处理速度受设备与页面刷新影响，视频时间与原片保持对应。')

signature = (str(source_path), selected_model, device, confidence, iou)
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
    if st.button('开始检测', type='primary', disabled=locked or source_path is None, width='stretch'):
        try:
            if session is not None:
                session.stop()
            with st.spinner('正在加载模型并打开视频…'):
                detector = load_video_detector(str(ROOT / 'fire-models' / (selected_model + '.pt')), device)
                st.session_state[SESSION_KEY] = VideoSession(source_path, detector, confidence, iou)
            st.session_state['video_detection_signature'] = signature
            st.session_state['video_detection_name'] = source_name
            st.session_state.pop('video_detection_error', None)
            st.rerun()
        except (ValueError, OSError, RuntimeError) as exc:
            st.session_state['video_detection_error'] = str(exc)
with pause_col:
    pause_label = '继续检测' if session is not None and session.status == 'paused' else '暂停检测'
    if st.button(pause_label, disabled=not locked, width='stretch'):
        if session.status == 'paused':
            session.resume()
        else:
            session.pause()
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
        st.info('选择视频后点击“开始检测”，这里会逐帧显示带框画面。')
        return
    if active_session.status == 'running':
        try:
            active_session.step()
        except Exception:
            logging.getLogger(__name__).exception('Video inference failed')
        if active_session.status in {'completed', 'error'}:
            st.rerun()
    st.caption(f"{st.session_state['video_detection_name']} · {STATUS[active_session.status]}")
    latest = active_session.latest
    if latest is not None:
        frame_col, time_col, fire_col, smoke_col = st.columns(4)
        frame_col.metric('当前帧', f'{latest.number} / {active_session.info.frame_count or "未知"}')
        time_col.metric('视频时间（估计）' if latest.time_estimated else '视频时间', format_time(latest.media_seconds))
        fire_col.metric('火焰', latest.counts.get('fire', 0))
        smoke_col.metric('烟雾', latest.counts.get('smoke', 0))
        # Keep each image in the same update as its metrics. Separate media URLs
        # can expire while an earlier browser request is still in flight.
        encoded = BytesIO()
        Image.fromarray(latest.rgb).save(encoded, format='JPEG', quality=85)
        image_url = 'data:image/jpeg;base64,' + base64.b64encode(encoded.getvalue()).decode('ascii')
        st.image(image_url, caption='逐帧检测结果', width='content')
        st.caption(f'处理速度 {active_session.processing_fps:.1f} 帧/秒 · 本帧推理 {latest.inference_ms:.1f} ms · 源视频 {active_session.info.fps:.2f} 帧/秒')
        if active_session.info.frame_count:
            st.progress(min(1.0, latest.number / active_session.info.frame_count))
        with st.expander('当前帧检测明细'):
            rows = [{'类别': {'fire': '火焰', 'smoke': '烟雾'}.get(box['class'], box['class']), '置信度': round(box['confidence'], 3), '原图边界框（像素）': ', '.join(str(round(value)) for value in box['xyxy'])} for box in latest.detections]
            if rows:
                st.dataframe(rows, hide_index=True, width='stretch')
            else:
                st.write('当前帧未检出火焰或烟雾。')
    if active_session.info.fps_estimated:
        st.info('未能读取源视频帧率，帧率显示值按30帧/秒估计。')
    if active_session.status == 'completed':
        st.success(f'检测完成：共处理{active_session.processed_frames}帧；含火焰{active_session.fire_frames}帧，含烟雾{active_session.smoke_frames}帧。')
    elif active_session.status == 'error':
        st.error(active_session.error)


show_detection()
