from pathlib import Path
import streamlit as st
from channels import load_channels, save_channels, ROOT
from monitoring import load_monitoring_points

st.set_page_config(page_title='命名通道',page_icon='📺')
st.title('命名监控通道')
st.caption('一次检测一个通道；在Demo Videos选择命名通道。变更配置仅影响下一次观测，历史记录保留原快照。')
try:
    rows=load_channels();points=load_monitoring_points(ROOT/'configs/monitoring_points.json')
    selected=st.selectbox('编辑通道',range(len(rows)),format_func=lambda i:rows[i]['name'])
    row=rows[selected]
    with st.form('channel_form'):
        name=st.text_input('通道名称',value=row['name'])
        kind=st.selectbox('输入类型',['video','camera'],index=0 if row['kind']=='video' else 1)
        videos=[str(p.relative_to(ROOT)).replace('\\','/') for p in sorted((ROOT/'demo-videos').glob('*')) if p.suffix.lower() in {'.mp4','.mov','.mkv','.avi'}]
        video=st.selectbox('本地视频',videos,index=videos.index(row['source']) if row['source'] in videos else 0)
        camera=st.number_input('USB编号',min_value=0,max_value=9,value=row['source'] if row['kind']=='camera' else 0)
        point=st.selectbox('关联模拟监控点',[p.id for p in points],index=[p.id for p in points].index(row['point_id']),format_func=lambda key:next(p.name for p in points if p.id==key))
        if st.form_submit_button('保存通道配置'):
            rows[selected]={**row,'name':name,'kind':kind,'source':video if kind=='video' else int(camera),'point_id':point}
            save_channels(rows);st.success('已保存，下次开始检测时生效。')
    st.dataframe(rows,hide_index=True)
except (OSError,ValueError,TypeError) as exc:st.error(str(exc))
