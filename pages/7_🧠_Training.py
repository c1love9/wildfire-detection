"""Reviewed dataset training and explicit validation/held-out test evaluation."""

from pathlib import Path

import torch
import streamlit as st
import pandas as pd
import altair as alt

from annotations import AnnotationStore
from dataset_store import DatasetStore, dataset_root
from training import ACTIVE, EvalConfig, TrainConfig, TrainingStore, bounded, history_rows


ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title='YOLO训练与评估', page_icon='🧠', layout='wide')
st.title('YOLO模型训练与评估')
st.caption('使用已复核数据集训练新的fire/smoke模型；每次任务保存独立数据快照、参数、日志、曲线与权重。')
dataset = DatasetStore(dataset_root(ROOT))
store = TrainingStore(dataset)
jobs = store.jobs()
active = any(row['status'] in ACTIVE for row in jobs)
exports = {row['id']: row for row in AnnotationStore(dataset).exports() if row['kind'] == 'split'}
completed = {row['id']: row for row in jobs if row['kind'] == 'train' and row['status'] == 'completed'}
actor = st.text_input('实验执行者姓名或编号', max_chars=100, key='training_actor')
note = st.text_input('实验说明', max_chars=2000, placeholder='例如：按采集组隔离的YOLOv8n基线，固定种子42。', key='training_note')
mode = st.radio('训练工作区', ['新建训练', '模型评估'], horizontal=True)
devices = ['CPU'] + (['GPU (CUDA)'] if torch.cuda.is_available() else [])
if active:
    st.info('当前数据目录有任务运行中。可切换页面，进程会继续；完成后点击“刷新任务列表”新建下一项。')


def started(identifier):
    st.session_state['training_selected_job'] = identifier
    st.rerun()


if mode == '新建训练':
    if not exports:
        st.info('请先在Dataset整理与划分数据，再到Annotations复核并生成完整YOLO训练数据集。')
    chosen = st.selectbox('已复核YOLO数据集', list(exports), format_func=lambda key: f"{key} · {exports[key]['manifest']['image_counts']}") if exports else None
    st.caption('初始模型：上游COCO预训练YOLOv8n；训练新建0=fire、1=smoke检测头。不会覆盖现有报警权重。')
    defaults = TrainConfig.load()
    with st.form('training_parameters'):
        a, b, c = st.columns(3)
        epochs = a.number_input('训练轮数', 1, 500, defaults.epochs)
        imgsz = b.selectbox('训练图像尺寸', [160, 320, 640, 960, 1280], index=2)
        batch = c.number_input('训练批大小', 1, 64, defaults.batch)
        a, b, c = st.columns(3)
        device = a.selectbox('训练计算设备', devices, index=len(devices) - 1)
        optimizer = b.selectbox('优化器', ['AdamW', 'SGD'])
        lr = c.number_input('初始学习率', 0.000001, 0.1, defaults.lr0, format='%.6f')
        a, b = st.columns(2)
        seed = a.number_input('随机种子', 0, 2147483647, defaults.seed)
        patience = b.number_input('早停耐心轮数（0表示关闭）', 0, 500, defaults.patience)
        multi_scale = st.checkbox('多尺度训练',value=defaults.multi_scale,help='每批训练随机改变尺寸；公平对比时统一最终评估尺寸。Experiments可保存只有此因素不同的成对配置。')
        st.caption('单设备、workers=0、固定种子、deterministic=True、FP32；每轮仅验证val，test留给单独评估。')
        run = st.form_submit_button('开始训练', type='primary', disabled=not chosen or active or not actor.strip() or not note.strip())
    if run:
        try:
            config = TrainConfig(epochs=int(epochs), imgsz=imgsz, batch=int(batch), device='0' if device == 'GPU (CUDA)' else 'cpu',
                                 optimizer=optimizer, lr0=lr, seed=int(seed), patience=int(patience),multi_scale=multi_scale)
            started(store.start_train(chosen, config, actor, note))
        except (OSError, ValueError) as exc:
            st.error(str(exc))
else:
    st.caption('best.pt由训练时的验证集选出。测试集应在方案确定后评估；反复根据测试分数调参会破坏其独立性。')
    parent = st.selectbox('已完成训练任务', list(completed)) if completed else None
    if not completed:
        st.info('尚无成功完成的训练任务。')
    with st.form('evaluation_parameters'):
        a, b, c = st.columns(3)
        split = a.selectbox('评估集合', ['test', 'val'])
        imgsz = b.selectbox('评估图像尺寸', [160, 320, 640, 960, 1280], index=([160, 320, 640, 960, 1280].index(completed[parent]['request']['config']['imgsz']) if parent else 2))
        batch = c.number_input('评估批大小', 1, 64, 8)
        a, b, c = st.columns(3)
        device = a.selectbox('评估计算设备', devices, index=len(devices) - 1)
        conf = b.number_input('评估最低置信度', 0.0001, 0.99, 0.001, format='%.4f')
        iou = c.number_input('评估NMS IoU', 0.01, 0.99, 0.7, 0.05)
        run = st.form_submit_button('开始评估', type='primary', disabled=not parent or active or not actor.strip() or not note.strip())
    if run:
        try:
            started(store.start_eval(parent, EvalConfig(split=split, imgsz=imgsz, batch=int(batch), device='0' if device == 'GPU (CUDA)' else 'cpu',
                                                      conf=conf, iou=iou), actor, note))
        except (OSError, ValueError) as exc:
            st.error(str(exc))

st.divider()
st.subheader('任务与实验记录')
if st.button('刷新任务列表'):
    st.rerun()
if jobs:
    names = {'queued': '准备中', 'running': '运行中', 'cancelling': '取消中', 'cancelled': '已取消', 'interrupted': '意外中断', 'completed': '已完成', 'failed': '失败'}
    selected = st.selectbox('查看任务', [row['id'] for row in jobs], key='training_selected_job',
                            format_func=lambda key: f"{'训练' if key.startswith('train_') else '评估'} · {key} · {names[next(row['status'] for row in jobs if row['id'] == key)]}")

    selected_active = next(row['status'] for row in jobs if row['id'] == selected) in ACTIVE

    @st.fragment(run_every=3 if selected_active else None)
    def show_job(identifier):
        job = store.get(identifier, refresh=True)
        if selected_active and job['status'] not in ACTIVE:
            st.rerun()
        directory = store.directory(identifier)
        st.write(f"任务状态：{names[job['status']]} · {job['progress'].get('stage', '等待工作进程')}")
        st.caption(f"执行者：{job['actor']} · 创建时间（UTC）：{job['created_at']} · {job['note']}")
        st.caption(f'本机保存目录：{directory}')
        if job['error']:
            st.error(job['error'])
        if job['status'] in ACTIVE:
            done = job['progress'].get('completed_epochs', 0)
            total = job['request']['config'].get('epochs', 1)
            st.progress(min(1.0, (done or 0) / total), text=f'已完成 {done or 0}/{total} 轮' if job['kind'] == 'train' else '正在评估')
            if st.button('取消当前任务', key='cancel_' + identifier, disabled=job['status'] == 'cancelling'):
                store.cancel(identifier)
                st.rerun(scope='fragment')
        try:
            rows = history_rows(directory)
            if rows:
                frame = pd.DataFrame(rows).set_index('epoch')
                for title, columns in [('训练损失', [c for c in frame if c.startswith('train/')]),
                                       ('验证损失', [c for c in frame if c.startswith('val/')]),
                                       ('验证指标', [c for c in frame if c.startswith('metrics/')])]:
                    st.write(title)
                    long = frame[columns].reset_index().melt('epoch', var_name='series', value_name='value')
                    chart = alt.Chart(long).mark_line(point=True).encode(
                        x=alt.X('epoch:Q', axis=alt.Axis(format='.0f', tickMinStep=1, tickCount=min(8, len(rows)))),
                        y='value:Q', color='series:N', tooltip=['epoch:Q', 'series:N', 'value:Q']).properties(height=240)
                    # A new history length/job gets a new Vega view. Reusing
                    # line_chart views across fragments can retain stale data IDs.
                    st.altair_chart(chart, key=f'curve_{identifier}_{title}_{len(rows)}', use_container_width=True)
            if job['status'] == 'completed':
                result = store.result(identifier)
                report = result['metrics']
                st.subheader(f"{report['split']} 集评估结果")
                for column, key, label in zip(st.columns(5), ['precision', 'recall', 'f1', 'map50', 'map50_95'], ['Precision', 'Recall', 'F1', 'mAP@0.5', 'mAP@0.5:0.95']):
                    value = report['overall'][key]
                    column.metric(label, '无参考目标' if value is None else f'{value:.4f}')
                st.dataframe(report['per_class'], hide_index=True, width='stretch')
                st.caption('P/R/F1使用该评估集合上Ultralytics的最大平均F1点；缺少参考目标的类别为null。速度为固定批大小下的验证计时，非摄像头端到端速度。')
                st.json({'speed_ms_per_image': report['speed_ms_per_image'], 'inference_fps': report['inference_fps']}, expanded=False)
                plots = sorted(path.name for path in (directory / 'output').glob('*.png'))
                if plots:
                    plot = st.selectbox('查看保存图表', plots, index=plots.index('results.png') if 'results.png' in plots else 0, key='plot_' + identifier)
                    st.image(str(directory / 'output' / plot), width=1000)
                files = ['request.json', 'effective_parameters.json', 'environment.json', 'dataset_receipt.json', 'metrics.json', 'result.json']
                files += ['output/results.csv', 'output/results.png'] if job['kind'] == 'train' else []
                if job['kind'] == 'train':
                    files += [result['weights'][name]['path'] for name in ('best', 'last')]
                    st.info('此任务的best.pt已加入检测模型列表；可在Home图片检测或Demo Videos视频/摄像头中选择，核验后加载。')
                file = st.selectbox('选择下载文件', files, key='download_' + identifier)
                path = bounded(directory, file)
                st.download_button('下载实验文件', path.read_bytes(), file_name=identifier + '_' + path.name,
                                   mime='application/octet-stream', on_click='ignore', key='download_button_' + identifier)
        except (OSError, ValueError, KeyError) as exc:
            st.warning(f'记录暂不可读：{exc}')
        with st.expander('参数与工作日志'):
            st.json(job['request'])
            st.code(store.read_log(identifier) or '工作进程尚未写入日志。', language='text')

    show_job(selected)
else:
    st.info('还没有训练或评估任务。新任务会保留独立记录，不覆盖此前实验。')
