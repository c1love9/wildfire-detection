from pathlib import Path
import streamlit as st
from image_input import demo_image_choices
from ui_theme import apply_theme, page_header

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title='精选演示图片', page_icon='📸', layout='wide')
apply_theme()
page_header('精选演示图片', '查看无框输入原图，在Home中使用当前模型生成检测结果。',
            section='DEMO GALLERY', tags=('室内与室外', '明火与烟雾', '正常场景'))
st.page_link('1_🔥_Home.py', label='进入图片检测', icon='🔎')
try:
    samples = demo_image_choices(ROOT)
    columns = st.columns(3, gap='large')
    for index, sample in enumerate(samples):
        with columns[index % 3]:
            with st.container(border=True):
                st.markdown('**'+sample['label']+'**')
                st.image(str(sample['path']), width=1000)
    st.caption('精选训练集样例用于功能演示；正式性能请查看独立测试结果。')
except (OSError, ValueError, KeyError, TypeError) as exc:
    st.error('演示图片无法读取：'+str(exc))

