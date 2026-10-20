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
    rows = trend_rows(samples)
    if not rows:
        st.info('此观测没有面积采样记录，旧日志不补造数据。')
        return
    st.subheader('面积与报警趋势')
    st.markdown(trend_svg(rows), unsafe_allow_html=True)
    st.markdown(trend_svg(rows, levels=True), unsafe_allow_html=True)
    st.caption('每秒采样，等级变化及结束时额外保存；缺失面积留空。变化率是画面占比变化，不是物理蔓延速度。')
    st.download_button('导出面积趋势 CSV', csv_bytes(rows), 'area_trends.csv', 'text/csv', key=key)


def trend_svg(rows, levels=False):
    """Inline SVG owns its data, including across Streamlit dialog/fragment reruns."""
    series = [('报警等级', '#e45555')] if levels else [('火焰占比（%）','#ff7043'),('烟雾占比（%）','#42a5f5'),('火烟并集（%）','#66bb6a')]
    t0, t1 = rows[0]['时间（秒）'], rows[-1]['时间（秒）']
    maximum = 3 if levels else max(1, max((row[name] or 0 for row in rows for name,_ in series),default=1)*1.1)
    x = lambda row: 50+(row['时间（秒）']-t0)/max(t1-t0,.001)*800
    y = lambda value: 175-value/maximum*140
    body = ['<svg viewBox="0 0 900 220" role="img" aria-label="'+('报警等级趋势' if levels else '面积趋势')+'" style="width:100%;background:#18222d;border-radius:6px">',
            '<path d="M50 30 V175 H860" fill="none" stroke="#9aa7b5"/>',
            f'<text x="8" y="38" fill="#ddd" font-size="13">{maximum:.1f}</text>',
            f'<text x="45" y="202" fill="#ddd" font-size="13">{t0:.2f}秒</text><text x="800" y="202" fill="#ddd" font-size="13">{t1:.2f}秒</text>']
    for index,(name,color) in enumerate(series):
        commands=[]; connected=False; previous=None
        for row in rows:
            value=row[name]
            if value is None: connected=False;continue
            if levels and connected: commands.append(f'L{x(row):.2f} {y(previous):.2f}')
            commands.append(f'{"L" if connected else "M"}{x(row):.2f} {y(value):.2f}')
            body.append(f'<circle cx="{x(row):.2f}" cy="{y(value):.2f}" r="2" fill="{color}"><title>{row["时间（秒）"]:.2f}秒 {name} {value:.3f}</title></circle>')
            connected=True;previous=value
        body.append(f'<path d="{" ".join(commands)}" fill="none" stroke="{color}" stroke-width="2"/>')
        body.append(f'<text x="{80+index*240}" y="20" fill="{color}" font-size="14">{name}</text>')
    return ''.join(body)+'</svg>'
