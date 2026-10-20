"""Bounded Streamlit model cache keyed by file content, device and provenance."""

import json
import torch
import streamlit as st

from model_runtime import YoloDetector


@st.cache_resource(max_entries=4, show_spinner=False)
def _load(weights, sha256, device, imgsz, metadata_json, fire_smoke):
    return YoloDetector(weights, device, imgsz=imgsz, expected_sha256=sha256,
                        model_metadata=json.loads(metadata_json), require_fire_smoke=fire_smoke)


def load_detector(choice, device, imgsz=None):
    choice.verify()  # Always check even on a cache hit.
    return _load(choice.path, choice.sha256, device, imgsz or choice.imgsz,
                 json.dumps(choice.metadata(), sort_keys=True), choice.origin != 'general')


def show_model(choice):
    if choice.error:
        st.error(choice.error)
    else:
        label = '自训模型' if choice.origin == 'trained' else '上游模型' if choice.origin == 'upstream' else 'COCO通用模型'
        st.caption(f'{label}：{choice.label} · 权重 {choice.sha256[:12]}')
        if choice.origin == 'trained':
            st.caption(f'训练任务：{choice.training_id} · 已训练 {choice.completed_epochs} 轮 · 执行者：{choice.actor}')
            st.caption('训练完成表示流程完成；模型效果以自己的独立测试结果为准。')


@st.cache_resource(max_entries=2, show_spinner='正在加载并预热SAM分割模型…')
def _sam(device):
    from segmentation import SamSegmenter
    return SamSegmenter(device)


def with_segmentation(detector, enabled):
    if not enabled:
        return detector
    from segmentation import SegmentedDetector, verify_model
    verify_model()
    return SegmentedDetector(detector, _sam(detector.device))


def show_segmentation(result, downloads=False):
    from io import BytesIO
    import numpy as np
    from PIL import Image
    areas = result.areas()
    if result.segmentation_status == 'disabled':
        return
    if result.segmentation_error:
        st.warning('本帧分割不可用，面积未知；检测与原有报警继续。' + result.segmentation_error)
        return
    columns = st.columns(3)
    for column, (key, label) in zip(columns, [('fire', '火焰画面占比'), ('smoke', '烟雾画面占比'), ('combined', '火烟并集占比')]):
        column.metric(label, f'{areas[key] * 100:.2f}%')
        if downloads:
            mask = result.masks[key] if key != 'combined' else np.logical_or(result.masks['fire'], result.masks['smoke'])
            buffer = BytesIO()
            Image.fromarray(mask.astype(np.uint8) * 255).save(buffer, format='PNG')
            column.download_button('下载' + label + '掩码', buffer.getvalue(), key + '_mask.png', 'image/png')
    st.caption('按原图像素计算，仅为画面占比；分割结果来自检测框提示SAM，未进行火灾数据微调。')
