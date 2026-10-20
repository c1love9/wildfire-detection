from pathlib import Path
import sqlite3

import streamlit as st

from alarm import LEVEL_NAMES
from alarm_log import AlarmLog, END_NAMES, event_rows, events_csv, alarm_log_path


ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = alarm_log_path(ROOT)

st.set_page_config(page_title='报警日志', page_icon='🚨', layout='wide')
st.title('报警日志')
st.caption('本机保存的火烟确认事件；最高等级用于筛选。停止、视频结束或异常表示观察结束，不表示火情恢复。')
st.button('刷新记录')
if not LOG_PATH.exists():
    st.info('还没有报警记录，请先在视频页开始检测。')
    st.stop()

try:
    store = AlarmLog(LOG_PATH)
    source_col, level_col = st.columns(2)
    source = source_col.selectbox('视频来源筛选', ['全部', *store.sources()])
    level_name = level_col.selectbox('最高等级筛选', ['全部', '关注', '预警', '报警'])
    peak_level = None if level_name == '全部' else next(level for level, name in LEVEL_NAMES.items() if name == level_name)
    events = store.events(source=None if source == '全部' else source, peak_level=peak_level)
    st.caption(f'显示最近 {len(events)} 条匹配事件（最多500条）；“未结束”可能是仍在检测或上次运行未正常结束。')
    if not events:
        st.info('没有匹配的已确认事件。')
        st.stop()
    st.dataframe(event_rows(events), hide_index=True, width='stretch')
    st.download_button('导出当前筛选 CSV', data=events_csv(events), file_name='alarm_events.csv', mime='text/csv')
    selected = st.selectbox('查看事件变化', events, format_func=lambda event: f"{event['source']} · {event['started_at']} · {event['id'][:12]}")
    transitions = store.transitions(selected['id'])
    if transitions:
        reasons = {'confirmed': '连续确认', 'escalated': '升级', 'downgraded': '降级', 'recovered': '恢复', **END_NAMES}
        st.dataframe([{'帧号': row['frame'], '视频时间（秒）': round(row['media_seconds'], 3),
                       '原等级': LEVEL_NAMES[row['from_level']], '新等级': LEVEL_NAMES[row['to_level']],
                       '原因': reasons[row['kind']], '记录时间（UTC）': row['recorded_at']}
                      for row in transitions], hide_index=True, width='stretch')
except (sqlite3.Error, OSError):
    st.error('无法读取报警日志，请检查日志文件是否完整以及目录权限。')
