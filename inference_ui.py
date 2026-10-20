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
