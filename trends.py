"""Bounded live window and durable one-Hz image-coverage samples."""
from collections import deque
import sqlite3
from dataset_store import csv_bytes


class TrendRecorder:
    def __init__(self, store, run_id):
        self.store, self.run_id = store, run_id
        self.recent = deque(maxlen=300)
        self.last = self.pending = None
        self.error = None

    def add(self, frame, level):
        result = frame.result
        areas = result.areas() if result is not None else {'fire': None, 'smoke': None, 'combined': None}
        self.pending = {'frame': frame.number, 'media_seconds': frame.media_seconds, **areas, 'level': level,
                        'status': result.segmentation_status if result is not None else 'disabled', 'elapsed_ms': frame.inference_ms}
        if self.last is None or frame.media_seconds - self.last['media_seconds'] >= 1 - 1e-9 or level != self.last['level']:
            self.flush()

    def flush(self):
        if self.pending is None or self.last is not None and self.last['frame'] == self.pending['frame']:
            return
        sample = self.pending.copy()
        self.recent.append(sample)
        self.last = sample
        if self.store:
            try: self.store.save_sample(self.run_id, sample)
            except (sqlite3.Error, OSError): self.error = '面积趋势写入失败，当前画面继续；请检查磁盘。'


def trend_rows(samples):
    rows, previous = [], None
    for sample in samples:
        value = sample['combined']
        dt = sample['media_seconds'] - previous['media_seconds'] if previous is not None else 0
        rate = (value - previous['combined']) * 100 / dt if dt > 0 and value is not None and previous['combined'] is not None else None
        rows.append({'时间（秒）': sample['media_seconds'], '帧号': sample['frame'],
                     **{label: sample[key] * 100 if sample[key] is not None else None for key, label in
                        [('fire', '火焰占比（%）'), ('smoke', '烟雾占比（%）'), ('combined', '火烟并集（%）')]},
                     '画面占比变化率（百分点/秒）': rate, '报警等级': sample['level'], '分割状态': sample['status'], '处理耗时（ms）': sample['elapsed_ms']})
        previous = sample
    return rows


def show_trends(samples, key):
    import streamlit as st
    import pandas as pd
    rows = trend_rows(samples)
    if not rows:
        st.info('此观测没有面积采样记录，旧日志不补造数据。')
        return
    st.subheader('面积与报警趋势')
    data = pd.DataFrame(rows).set_index('时间（秒）')
    st.line_chart(data[['火焰占比（%）', '烟雾占比（%）', '火烟并集（%）']], height=200)
    st.line_chart(data[['报警等级']], height=150)
    st.caption('每秒采样，等级变化及结束时额外保存；缺失面积留空。变化率是画面占比变化，不是物理蔓延速度。')
    st.download_button('导出面积趋势 CSV', csv_bytes(rows), 'area_trends.csv', 'text/csv', key=key)
