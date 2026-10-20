from pathlib import Path
import json
import torch
import streamlit as st
from PIL import Image
from comparison import ComparisonStore, DinoDetector, predict_pair
from grounding_prelabel import GroundingPrelabeler, PrelabelConfig
from dataset_store import DatasetStore,dataset_root,csv_bytes
from model_catalog import fire_models
from inference_ui import load_detector

ROOT=Path(__file__).resolve().parents[1]
st.set_page_config(page_title='模型统一对比',page_icon='⚖️',layout='wide')
st.title('YOLO与开放域模型对比')
st.caption('图片并排推理不产生精度结论；量化评测使用成功训练任务的独立test快照。GroundingDINO未微调，也用于本项目预标注，报告必须披露。')
dataset=DatasetStore(dataset_root(ROOT));store=ComparisonStore(dataset)
choices=fire_models(ROOT,dataset.root)
key=st.selectbox('YOLO对比模型',list(choices),format_func=lambda k:choices[k].label)
device=st.selectbox('对比计算设备',['cpu']+(['cuda:0'] if torch.cuda.is_available() else []),index=1 if torch.cuda.is_available() else 0)
fire_prompt=st.text_input('对比火焰提示词',PrelabelConfig().fire_prompt)
smoke_prompt=st.text_input('对比烟雾提示词',PrelabelConfig().smoke_prompt)
actor=st.text_input('对比执行者');note=st.text_input('对比说明')

@st.cache_resource(max_entries=2)
def grounding(device):return GroundingPrelabeler(device)

def models():
    config=PrelabelConfig(fire_prompt=fire_prompt,smoke_prompt=smoke_prompt)
    return load_detector(choices[key],device,640),DinoDetector(grounding(device),config)

mode=st.radio('对比工作区',['图片并排检测','统一test评测'],horizontal=True)
if mode=='图片并排检测':
    upload=st.file_uploader('对比图片',type=['png','jpg','jpeg'])
    demo=st.selectbox('已有演示图片',[p.name for p in sorted((ROOT/'demo-images').glob('*.jpeg')) if p.stat().st_size>1000])
    if st.button('运行并排检测'):
        try:
            with st.spinner('运行两个模型…'):
                image=Image.open(upload if upload else ROOT/'demo-images'/demo).convert('RGB');left,right=predict_pair(image,*models())
            a,b=st.columns(2);a.image(left,caption='YOLO · 置信度≥0.25',width=640);b.image(right,caption='GroundingDINO · 置信度≥0.25',width=640)
            st.success('并排推理完成；没有独立参考标注，不计算准确率。')
        except (OSError,ValueError,RuntimeError) as exc:st.error(str(exc))
else:
    st.caption('固定640×640、batch=1、FP32；固定阈值P/R/F1，AP使用0.50～0.95的IoU。与训练页的最大F1点指标分开保存。')
    if choices[key].origin!='trained':st.info('待评测：请先选择本人成功训练的模型。当前不以演示模型充当正式实验。')
    if st.button('运行统一测试评测',disabled=choices[key].origin!='trained' or not actor.strip() or not note.strip()):
        try:
            yolo,dino=models();progress=st.progress(0.)
            identifier=store.run(choices[key].training_id,yolo,dino,actor,note,progress.progress)
            st.success('评测完成：'+identifier)
        except (OSError,ValueError,RuntimeError) as exc:st.error(str(exc))
records=[r for r in store.records() if r['status']=='completed']
st.subheader('历史对比结果')
if records:
    selected=st.selectbox('对比记录',[r['id'] for r in records])
    try:
        result=store.result(selected);rows=[{'模型':name,**values['overall'],'平均耗时ms':values['mean_ms_per_image'],'FPS':values['fps'],'权重字节':values['file_bytes']} for name,values in result['models'].items()]
        st.dataframe(rows,hide_index=True);st.download_button('下载对比CSV',csv_bytes(rows),'comparison.csv','text/csv')
        st.download_button('下载完整对比记录',json.dumps(result,ensure_ascii=False,indent=2),'comparison.json','application/json')
        st.json(result['protocol'])
    except (OSError,ValueError) as exc:st.error(str(exc))
else:st.info('待评测：没有已完成的统一对比。')
