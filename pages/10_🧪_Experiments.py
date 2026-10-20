from pathlib import Path
import json
import torch
import cv2
import streamlit as st
from dataclasses import asdict
from dataset_store import DatasetStore,dataset_root
from annotations import AnnotationStore
from training import TrainConfig,TrainingStore,history_rows
from experiments import ExperimentStore,alarm_ablation
from model_catalog import fire_models
from inference_ui import load_detector

ROOT=Path(__file__).resolve().parents[1]
st.set_page_config(page_title='优化实验',page_icon='🧪',layout='wide');st.title('优化对照实验')
st.caption('两组仅改变多尺度训练开关；同初始权重、同数据、同种子，统一640尺寸评测。不预设优化一定有效。')
dataset=DatasetStore(dataset_root(ROOT));store=ExperimentStore(dataset);training=TrainingStore(dataset)
exports=[r for r in AnnotationStore(dataset).exports() if r['kind']=='split']
device=st.selectbox('实验计算设备',['cpu']+(['cuda:0'] if torch.cuda.is_available() else []),index=1 if torch.cuda.is_available() else 0)
with st.form('new_pair'):
    export=st.selectbox('成对实验数据集',[r['id'] for r in exports]) if exports else None
    actor=st.text_input('实验执行者');note=st.text_input('成对实验说明')
    epochs=st.number_input('成对训练轮数',1,500,50);batch=st.number_input('成对批大小',1,64,4)
    if not exports:st.info('待实验：尚无已复核训练数据。')
    if st.form_submit_button('保存成对实验配置',disabled=not export):
        try:store.create(export,TrainConfig(epochs=int(epochs),batch=int(batch),device='0' if device.startswith('cuda') else 'cpu'),actor,note);st.success('成对配置已保存。')
        except (OSError,ValueError) as exc:st.error(str(exc))
records=store.records()
if records:
    selected=st.selectbox('优化实验记录',records)
    try:
        manifest=store.read(selected);st.json(manifest)
        columns=st.columns(2)
        for column,variant in zip(columns,['baseline','multiscale']):
            with column:
                st.subheader('固定尺寸基线' if variant=='baseline' else '多尺度训练')
                if st.button('启动 '+variant,disabled=variant in manifest['jobs']):
                    store.start(selected,variant);st.rerun()
                if variant in manifest['jobs']:
                    job=training.get(manifest['jobs'][variant],refresh=True);st.write('训练状态：'+job['status'])
                    rows=history_rows(training.directory(job['id']))
                    if rows:st.dataframe(rows,hide_index=True)
                    curve=training.directory(job['id'])/'output/results.png'
                    if curve.is_file():st.image(str(curve),caption=variant+' 损失与验证曲线')
        if st.button('统一评估两组test',disabled=bool(manifest['evaluation_sha'])):store.evaluate(selected,device);st.rerun()
        if manifest['evaluation_sha']:
            output=store.evaluation(selected);st.json(output['difference_multiscale_minus_baseline'])
            st.download_button('下载成对实验结果',json.dumps(output,ensure_ascii=False,indent=2),'experiment.json','application/json')
    except (OSError,ValueError,RuntimeError) as exc:st.error(str(exc))
else:st.info('待实验：没有保存的优化配置。')
st.divider();st.subheader('单帧触发与连续确认对照')
video=st.selectbox('报警对照样例',[p.name for p in sorted((ROOT/'demo-videos').glob('*'))])
if st.button('运行样例前10秒策略对照'):
    try:
        detector=load_detector(fire_models(ROOT,dataset.root)['fire_n'],device,640)
        capture=cv2.VideoCapture(str(ROOT/'demo-videos'/video));fps=capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or fps<=0:raise ValueError('样例帧率无效。')
        frames=[]
        try:
            for index in range(min(600,round(fps*10))):
                ok,bgr=capture.read()
                if not ok:break
                frames.append((index/fps,detector.predict_frame(bgr,.2,.5).detections))
        finally:capture.release()
        result=alarm_ablation(frames);st.json(result)
        directory=dataset.root/'experiments';path=directory/('alarm_ablation_'+__import__('uuid').uuid4().hex+'.json')
        path.write_text(json.dumps({'source':video,'frames':frames,'result':result},ensure_ascii=False),encoding='utf-8')
        st.download_button('下载报警对照记录',path.read_bytes(),path.name,'application/json')
    except (OSError,ValueError,RuntimeError) as exc:st.error(str(exc))
