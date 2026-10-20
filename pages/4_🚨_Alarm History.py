from pathlib import Path
import json
import sqlite3

import streamlit as st

from alarm import LEVEL_NAMES
from alarm_log import AlarmLog, END_NAMES, event_rows, events_csv, alarm_log_path, alarm_media_root
from recording import resolve_media_asset


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
    source = source_col.selectbox('输入来源筛选', ['全部', *store.sources()])
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
    is_camera = json.loads(selected.get('metadata_json', '{}')).get('input_type') == 'camera'
    time_label = '本段观测时间' if is_camera else '视频时间'
    if is_camera:
        st.caption('摄像头时间从本段开始观测起计时；暂停后继续将重新计时与确认。实际日期见记录时间，观测中断不表示火情恢复。')
    initial_media = store.media(selected['id'])

    @st.fragment(run_every=1.0 if initial_media and initial_media['status'] in {'recording', 'encoding'} else None)
    def show_event_media():
        media = store.media(selected['id'])
        if media is None:
            st.info('此事件没有保存的截图或片段，可能来自旧版本或录制已关闭。')
            return
        if initial_media['status'] in {'recording', 'encoding'} and media['status'] not in {'recording', 'encoding'}:
            st.rerun()
        root = alarm_media_root(LOG_PATH)
        details = media['details']
        st.subheader('报警截图与片段')
        st.caption(f"首次确认：第{details['trigger_frame']}帧 · {time_label} {details['trigger_seconds']:.3f} 秒")
        snapshot_col, video_col = st.columns(2)
        snapshot = resolve_media_asset(root, media['snapshot_path'], '.jpg')
        with snapshot_col:
            if snapshot:
                st.image(str(snapshot), caption='首次确认时的带框截图', width='content')
            else:
                st.info('截图文件未保存或已不在媒体目录中。')
        with video_col:
            if media['status'] in {'recording', 'encoding'}:
                st.info('正在收集后续画面或生成视频；完成后自动刷新。若上次运行已中断，这份片段可能无法完成。')
            elif media['status'] in {'ready', 'partial'}:
                video = resolve_media_asset(root, media['clip_path'], '.mp4')
                if video:
                    st.video(str(video), format='video/mp4')
                    st.caption(f"实际片段：{details['start_seconds']:.3f}～{details['end_seconds']:.3f} 秒 · 首次确认位于片段约 {details['trigger_seconds'] - details['start_seconds']:.3f} 秒处 · 静音")
                    if details.get('time_estimated'):
                        st.caption('部分源时间戳为估计值。')
                    if media['status'] == 'partial':
                        reasons = []
                        if details['pre_truncated']:
                            reasons.append('确认前画面不足（视频开头或缓存限制）')
                        if details['post_truncated']:
                            reasons.append('确认后画面不足（输入结束、暂停、停止或异常）')
                        st.info('；'.join(reasons) + '，已保存实际可用范围。')
                else:
                    st.warning('片段文件已不在媒体目录中，请检查媒体文件是否随数据库一起保留。')
        if media['error']:
            st.warning('本事件媒体保存未完成：' + media['error'])

    show_event_media()
    transitions = store.transitions(selected['id'])
    if transitions:
        reasons = {'confirmed': '连续确认', 'escalated': '升级', 'downgraded': '降级', 'recovered': '恢复', **END_NAMES}
        st.dataframe([{'帧号': row['frame'], f'{time_label}（秒）': round(row['media_seconds'], 3),
                       '原等级': LEVEL_NAMES[row['from_level']], '新等级': LEVEL_NAMES[row['to_level']],
                       '原因': reasons[row['kind']], '记录时间（UTC）': row['recorded_at']}
                      for row in transitions], hide_index=True, width='stretch')
except (sqlite3.Error, OSError):
    st.error('无法读取报警日志，请检查日志文件是否完整以及目录权限。')
