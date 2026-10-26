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
from quality_export import quality_archive

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

mode=st.radio('质量工作区',['创建独立标注任务','独立标注','报告与导出','面积误差','工时对照'],horizontal=True,on_change=pause_annotation_timers)
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
    elif mode=='报告与导出':
        token=st.text_input('管理访问码',type='password')
        if token:
            access=store.access(token,True)
            progress=store.progress(token)
            st.caption(f"任务编号：{access['task_id']}；A已提交 {progress['A']}/{progress['total']}，B已提交 {progress['B']}/{progress['total']}")
            if not store.completed(access):st.info('等待两名标注者全部完成；此时不展示框或预标注。')
            else:
                submissions=store.submissions(token)
                st.caption('A/B原始标注分别保留；无需逐图仲裁。预标注分别与A、B比较，两套结果同时报告。导出不生成训练复核通过记录。')
                options={'both':'A/B完整包（含质量报告）','A':'仅A原始标注','B':'仅B原始标注'}
                selection=st.selectbox('导出内容',list(options),format_func=options.get)
                if st.button('生成导出包'):
                    content,summary=quality_archive(store,token,selection)
                    st.session_state['quality_download']={'task':access['task_id'],'selection':selection,'content':content}
                download=st.session_state.get('quality_download',{})
                if download.get('task')==access['task_id'] and download.get('selection')==selection:
                    st.download_button('下载标注ZIP',download['content'],f"{access['task_id']}_{selection}.zip",'application/zip',on_click='ignore')
                if st.button('生成标注质量报告'):
                    st.session_state['quality_report']=store.report(token)
                report=st.session_state.get('quality_report',{})
                if report.get('task_id')==access['task_id']:
                    summary=report['agreement_summary']
                    st.write({'匹配框平均IoU':summary['mean_iou_of_matched'],'匹配框对数':summary['matched_pairs'],
                              'A未匹配框':summary['unmatched_A'],'B未匹配框':summary['unmatched_B']})
                    rows=[{'人工参考':role,**result['overall']} for role,result in report['preannotation_quality'].items() if result is not None]
                    if rows:st.dataframe(rows,hide_index=True)
                    else:st.info('待评估：本任务未冻结可用预标注；仍可查看A/B一致性并导出原始标签。')
                    st.caption('IoU均值仅针对匹配框；P/R为冻结预标注对各份人工参考的结果，不是正式测试精度。工时节省另需受控计时。')
                    st.download_button('下载标注质量报告',json.dumps(report,ensure_ascii=False,indent=2),'annotation_quality.json','application/json',on_click='ignore')
                    with st.expander('查看完整质量报告'):st.json(report)
                if st.checkbox('查看原图与A/B标注'):
                    keys=[item['asset']['id'] for item in access['snapshot']]
                    key=st.selectbox('查看图片',keys,format_func=lambda value:f"Q{keys.index(value)+1:03d} · {value[:12]}");asset=store.frozen_asset(access,key)
                    st.image(str(dataset.asset_path(asset)),caption='原图',width=700)
                    for column,role in zip(st.columns(2),('A','B')):
                        submission=next(row for row in submissions if row['asset_id']==key and row['role']==role)
                        column.image(box_preview(dataset.asset_path(asset),submission['boxes']),caption=role+'独立结果（只读）')
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
