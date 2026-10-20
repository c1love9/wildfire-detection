"""Acknowledgeable, deduplicated browser notifications; sound requires opt-in."""
import io
import math
import struct
import wave


def notification_key(run_id, event_id, level):
    return f'{run_id}:{event_id}:{level}'


def tone():
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as output:
        output.setnchannels(1); output.setsampwidth(2); output.setframerate(16000)
        output.writeframes(b''.join(struct.pack('<h', int(6000 * math.sin(2 * math.pi * 880 * index / 16000))) for index in range(6400)))
    return buffer.getvalue()


def notification_pending(session):
    import streamlit as st
    if session.status != 'running' or not session.alarms.current_event:
        return False
    event = session.alarms.current_event
    level = session.alarms.snapshot.level
    if not level:
        return False
    key = notification_key(session.alarms.run_id, event['id'], level)
    return key not in st.session_state.get('alarm_notifications_seen', [])


def show_notifications(session, sound=False):
    import streamlit as st
    from alarm import LEVEL_NAMES
    if not notification_pending(session):
        return
    event, level = session.alarms.current_event, session.alarms.snapshot.level
    key = notification_key(session.alarms.run_id, event['id'], level)
    seen = st.session_state.setdefault('alarm_notifications_seen', [])
    # Mark before opening the dialog; reruns cannot replay an acknowledged alert.
    seen.append(key)
    del seen[:-300]

    @st.dialog('火烟报警')
    def notice():
        st.warning(f'{session.alarms.source} · {LEVEL_NAMES[level]}')
        st.write('事件编号：' + event['id'][:12])
        st.caption('此提示为课程原型判定，请结合实际画面核查。')
        if sound:
            st.audio(tone(), format='audio/wav', autoplay=True)
        if st.button('确认并关闭', key='ack_' + key):
            st.rerun()
    notice()
