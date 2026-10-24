import torch
import json
import uuid
import sqlite3
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st
from dataset_store import DatasetStore,dataset_root
from annotations import AnnotationStore,validate_boxes
from annotation_ui import box_editor,box_preview
from quality import QualityStore,WorkTimer,pause_work_timers

ROOT=Path(__file__).resolve().parents[1]
st.set_page_config(page_title='标注质量与面积误差',page_icon='🔬',layout='wide');st.title('标注质量与面积误差')
st.caption('独立工作区不向标注者显示模型候选或另一人的结果；访问码不是系统级账号认证。实际双人独立性需真人与操作规程保证。')
dataset=DatasetStore(dataset_root(ROOT));store=QualityStore(dataset);catalog=store.annotations.catalog()

def pause_annotation_timers():
    pause_work_timers([value for key,value in st.session_state.items() if key.startswith('blind_')])


def leave_blind_workspace():
    pause_annotation_timers()
    for key in list(st.session_state):
        if key!='blind_token' and key.startswith(('blind_','box_editor_')):
            del st.session_state[key]
    st.session_state['blind_token']=''

mode=st.radio('质量工作区',['创建独立标注任务','独立标注','管理与仲裁','面积误差','工时对照'],horizontal=True,on_change=pause_annotation_timers)
try:
    if mode=='创建独立标注任务':
        assets=st.multiselect('独立标注图片',[r['id'] for r in catalog],format_func=lambda k:k[:16])
        a=st.text_input('标注者A');b=st.text_input('标注者B');admin=st.text_input('任务管理者')
        if not catalog:st.info('待标注：当前没有正式图片。不会自动注入测试素材。')
        if st.button('创建独立任务',disabled=not assets or not a.strip() or not b.strip() or not admin.strip()):
            task,tokens=store.create(assets,a,b,admin);st.session_state['new_quality_access']={'task':task,'tokens':tokens}
        if 'new_quality_access' in st.session_state:
            st.warning('访问码仅在创建者当前会话显示，请分别交给对应人员，不要公开管理员访问码。')
            st.json(st.session_state['new_quality_access'])
    elif mode=='独立标注':
        token=st.text_input('独立标注访问码',type='password',key='blind_token',on_change=pause_annotation_timers)
        if token:
            access=store.access(token)
            if access['role']=='admin':raise ValueError('请使用A或B访问码。')
            progress=store.progress(token)
            st.caption(f"任务编号：{access['task_id']}；当前身份：{access['actor']}")
            if st.session_state.pop('blind_feedback',None)==(access['task_id'],access['role']):
                st.success('此图已独立提交并锁定。')
            st.progress(progress['submitted']/progress['total'],text=f"本人已提交 {progress['submitted']} / {progress['total']} 幅")
            st.button('退出本次独立标注',on_click=leave_blind_workspace)
            pending_only=st.checkbox('仅显示尚未提交的图片',value=True,key='blind_pending_only',on_change=pause_annotation_timers)
            all_keys=[item['asset']['id'] for item in access['snapshot']]
            numbers={key:index+1 for index,key in enumerate(all_keys)}
            visible=progress['remaining_asset_ids'] if pending_only else all_keys
            if not visible:
                st.success('本人的全部图片已提交。请退出，等待另一位标注者独立完成。')
                st.stop()
            key=st.selectbox('本人的待标注图片',visible,format_func=lambda value:f"Q{numbers[value]:03d} · {value[:12]}",on_change=pause_annotation_timers)
            asset=store.frozen_asset(access,key);submitted=store.own_submission(token,key)
            st.caption('当前身份：'+access['actor']+'；不载入预标注或另一位标注者的框。')
            if submitted:st.success('此图已独立提交并锁定。');st.json(submitted['boxes'])
            else:
                work_key=access['task_id']+access['role']+key
                work=st.session_state.setdefault('blind_'+work_key,{'boxes':[],'editor':uuid.uuid4().hex,'timer':WorkTimer(),'activity':None})
                timer=work['timer'];timer.expire()
                c1,c2=st.columns(2)
                if c1.button('开始或继续工作计时'):timer.start()
                if c2.button('暂停工作计时'):timer.pause()
                payload=box_editor(dataset.asset_path(asset),asset['width'],asset['height'],work['boxes'],work['editor'])
                if isinstance(payload,dict) and payload.get('document_key')==work['editor']:
                    work['boxes']=validate_boxes(payload['boxes'],asset['width'],asset['height'])
                    stamp=payload.get('activity_stamp_ms')
                    if stamp is not None and stamp!=work['activity']:timer.activity();work['activity']=stamp
                st.caption(f'已记录工作计时 {timer.seconds():.1f} 秒；空闲超过60秒停止累加。提交会自动暂停计时，休息时也可手动暂停。未保存计时随服务重启丢弃。')
                negative=st.checkbox('独立检查后确认整图无fire/smoke') if not work['boxes'] else False
                blockers=[]
                if timer.seconds()<=0:blockers.append('请先点击“开始或继续工作计时”，开展本图标注')
                if not work['boxes'] and not negative:blockers.append('请画出目标框；确实无火烟时勾选整图无目标确认')
                if blockers:st.info('暂不能提交：'+'；'.join(blockers)+'。')
                else:st.caption('本图可以提交：点击后自动暂停计时、保存并锁定；待办筛选会转到下一幅。')
                if st.button('提交并锁定本图独立标注',disabled=bool(blockers)):
                    store.submit_timed(token,key,work['boxes'],timer,negative)
                    st.session_state['blind_feedback']=(access['task_id'],access['role']);st.rerun()
    elif mode=='管理与仲裁':
        token=st.text_input('管理访问码',type='password')
        if token:
            access=store.access(token,True)
            progress=store.progress(token)
            st.caption(f"任务编号：{access['task_id']}；A已提交 {progress['A']}/{progress['total']}，B已提交 {progress['B']}/{progress['total']}，已仲裁 {progress['references']}/{progress['total']}")
            if not store.completed(access):st.info('等待两名标注者全部完成；此时不展示框或预标注。')
            else:
                submissions=store.submissions(token)
                keys=[item['asset']['id'] for item in access['snapshot']]
                key=st.selectbox('仲裁图片',keys,format_func=lambda value:f"Q{keys.index(value)+1:03d} · {value[:12]}");asset=store.frozen_asset(access,key)
                a,b=[next(r for r in submissions if r['asset_id']==key and r['role']==role) for role in ['A','B']]
                st.image(str(dataset.asset_path(asset)),caption='人工仲裁参考原图',width=700)
                c1,c2=st.columns(2)
                for column,submission,label in [(c1,a,'A独立结果'),(c2,b,'B独立结果')]:
                    column.image(box_preview(dataset.asset_path(asset),submission['boxes']),caption=label)
                    with column.expander(label+'坐标'):st.json(submission['boxes'])
                saved=store.reference(token,key)
                work_key='arbitration_'+access['task_id']+key
                work=st.session_state.setdefault(work_key,{'boxes':saved['boxes'] if saved else a['boxes'], 'editor':uuid.uuid4().hex})
                st.subheader('人工仲裁参考画布')
                st.caption('起始使用已保存参考或A的框，便于编辑；A与B都不是自动真值。请对照原图核查遗漏、误检、类别与边界。')
                ca,cb=st.columns(2)
                for column,label,boxes in [(ca,'从A载入参考画布',a['boxes']),(cb,'从B载入参考画布',b['boxes'])]:
                    if column.button(label):
                        work.update(boxes=boxes,editor=uuid.uuid4().hex);st.rerun()
                payload=box_editor(dataset.asset_path(asset),asset['width'],asset['height'],work['boxes'],work['editor'])
                if isinstance(payload,dict) and payload.get('document_key')==work['editor']:
                    work['boxes']=validate_boxes(payload['boxes'],asset['width'],asset['height'])
                with st.expander('查看当前人工参考框JSON'):
                    st.code(json.dumps(work['boxes'],ensure_ascii=False,indent=2),language='json')
                note=st.text_input('人工仲裁说明',key='arbitration_note_'+access['task_id']+key)
                if st.button('保存人工参考版本',disabled=not note.strip()):
                    store.arbitrate(token,key,work['boxes'],note);st.success('人工参考已追加保存。');saved=store.reference(token,key)
                if saved:
                    st.caption(f"已保存人工参考版本 {saved['seq']}；修改画布后须再次保存，才能将新结果用于训练。")
                    confirmed=st.checkbox('我已核查已保存的仲裁参考，确认用于训练标签',key='approve_reference_'+access['task_id']+key)
                    negative=st.checkbox('仲裁后确认整图无fire/smoke',key='approve_negative_'+access['task_id']+key) if not saved['boxes'] else False
                    latest=store.annotations.latest(key)
                    if st.button('确认此仲裁参考用于训练',disabled=not confirmed or work['boxes']!=saved['boxes'] or not saved['boxes'] and not negative):
                        store.approve_reference(token,key,saved['seq'],latest['id'] if latest else None,negative)
                        st.success('已保存为Annotations人工通过版本；A/B独立原始记录保持不变。')
                if st.button('生成标注质量报告'):
                    report=store.report(token);st.json(report);st.download_button('下载标注质量报告',json.dumps(report,ensure_ascii=False,indent=2),'annotation_quality.json','application/json')
    elif mode=='面积误差':
        if not catalog:st.info('待评测：没有图像及独立人工参考掩码。')
        else:
            key=st.selectbox('掩码对应图片',[r['id'] for r in catalog]);label=st.selectbox('掩码类别',['fire','smoke','combined'])
            reference=st.file_uploader('独立人工参考掩码PNG',type=['png']);prediction=st.file_uploader('模型预测掩码PNG',type=['png'])
            actor=st.text_input('面积评测执行者');note=st.text_input('参考来源与绘制方法')
            independent=st.checkbox('参考掩码由人工独立绘制，未使用待评估模型的结果充当真值')
            if st.button('计算并保存面积误差',disabled=not reference or not prediction or not independent or not actor.strip() or not note.strip()):
                output=store.compare_masks(key,np.array(Image.open(reference)),np.array(Image.open(prediction)),actor,note,independent,label)
                st.json(output);st.download_button('下载面积误差记录',json.dumps(output,ensure_ascii=False,indent=2),'mask_error.json','application/json')
    else:
        st.caption('填写实测有效工作秒数，不使用页面停留时长。相同人员、相同对照组编号的两种条件形成一对；需另行保证任务难度、质量标准一致并平衡执行顺序。')
        with st.form('worktime'):
            group=st.text_input('工时对照组编号');actor=st.text_input('计时人员');condition=st.selectbox('工作条件',['manual','assisted'])
            seconds=st.number_input('实测有效工作秒数',min_value=.1,max_value=86400.,value=1.);note=st.text_input('计时方法与任务说明')
            if st.form_submit_button('保存实测工时'):store.worktime(group,actor,condition,seconds,note);st.success('实测记录已保存。')
        report=store.worktime_report()
        if report['records']:st.dataframe(report['records'],hide_index=True)
        if report['paired_estimates']:st.dataframe(report['paired_estimates'],hide_index=True)
        else:st.info('待估算：没有完整的人工/辅助复核配对计时。')
        st.download_button('下载工时对照记录',json.dumps(report,ensure_ascii=False,indent=2),'worktime.json','application/json')
except (OSError,ValueError,RuntimeError,sqlite3.Error,StopIteration) as exc:st.error(str(exc))
