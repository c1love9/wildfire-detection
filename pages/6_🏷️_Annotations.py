"""GroundingDINO proposals, visual human review and traceable YOLO output."""

from dataclasses import replace
import json
import logging
from pathlib import Path
import time
import uuid

import torch
import streamlit as st

from annotation_ui import box_editor
from annotations import AnnotationStore, validate_boxes
from dataset_store import DatasetStore, dataset_root
from grounding_prelabel import GroundingPrelabeler, PrelabelConfig, model_directory


ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title='预标注与人工复核', page_icon='🏷️', layout='wide')
st.title('自动预标注与人工复核')
st.caption('GroundingDINO本机推理，结果先作为候选；人工检查后保存复核版本。YOLO类别固定为0=fire、1=smoke。')
dataset = DatasetStore(dataset_root(ROOT))
store = AnnotationStore(dataset)
catalog = store.catalog()
by_id = {row['id']: row for row in catalog}
sources = {row['asset_id']: row['source_name'] for row in dataset.snapshot()['occurrences']}


def image_name(key):
    row = by_id[key]
    status = '已通过' if row['decision'] == 'approved' else '草稿' if row['decision'] else '待复核'
    return f"{sources.get(key, key[:12])} · {key[:12]} · {status}"


@st.cache_resource
def load_grounder(device, directory):
    return GroundingPrelabeler(device, directory)


def new_work(asset, review, proposal=None):
    boxes = review['boxes'] if review else proposal['boxes'] if proposal else []
    return {'document_key': uuid.uuid4().hex, 'boxes': validate_boxes(boxes, asset['width'], asset['height']),
            'base_revision': review['id'] if review else None,
            'proposal_id': review['proposal_id'] if review else proposal['id'] if proposal else None,
            'input_sha256': asset['sha256'], 'started': time.monotonic()}


columns = st.columns(4)
for column, title, count in zip(columns, ['归档图片', '人工通过', '待复核/草稿', '最近预标注失败'],
                                 [len(catalog), sum(row['decision'] == 'approved' for row in catalog),
                                  sum(row['decision'] != 'approved' for row in catalog), sum(row['proposal_status'] == 'failed' for row in catalog)]):
    column.metric(title, count)
actor = st.text_input('执行者/复核者姓名或编号', key='annotation_actor', max_chars=100)
mode = st.radio('标注工作区', ['自动预标注', '人工复核', 'YOLO导出'], horizontal=True)
if st.session_state.get('annotation_feedback'):
    st.success(st.session_state['annotation_feedback'])
if not catalog:
    st.info('还没有图像，请先在Dataset导入图片或从视频抽帧。演示素材也可用于试用流程，但不计入正式数据。')

if mode == '自动预标注':
    defaults = PrelabelConfig.load()
    st.caption('按类别分别使用英文提示词；用句点分隔同义词。修改提示词可产生新预标注版本，已有人工复核不会被覆盖。')
    fire_prompt = st.text_input('火焰提示词', defaults.fire_prompt, max_chars=200)
    smoke_prompt = st.text_input('烟雾提示词', defaults.smoke_prompt, max_chars=200)
    with st.expander('模型与推理参数'):
        st.caption(f'{defaults.model_id} · 固定版本 {defaults.revision[:12]} · 未进行火灾数据微调')
        devices = ['CPU'] + (['GPU (CUDA)'] if torch.cuda.is_available() else [])
        device_label = st.selectbox('预标注计算设备', devices, index=len(devices) - 1)
        box_col, text_col, nms_col = st.columns(3)
        box_threshold = box_col.number_input('候选框阈值', 0.01, 0.99, defaults.box_threshold, 0.05)
        text_threshold = text_col.number_input('文本匹配阈值', 0.01, 0.99, defaults.text_threshold, 0.05)
        nms_iou = nms_col.number_input('同类重叠框抑制IoU', 0.01, 0.99, defaults.nms_iou, 0.05)
        short_edge = st.selectbox('预处理短边尺寸', [480, 640, 800], index=1)
    scope = st.selectbox('预标注范围', ['尚无预标注的图片', '全部图片', '手动选择图片'])
    maximum = st.number_input('本次最多处理图片', min_value=1, max_value=100, value=20)
    if scope == '手动选择图片':
        requested = st.multiselect('选择预标注图片', list(by_id), format_func=image_name)
    else:
        requested = [row['id'] for row in catalog if scope == '全部图片' or row['proposal_status'] is None]
    requested = requested[:int(maximum)]
    st.caption(f'本次将处理 {len(requested)} 幅；不向模型服务上传图像。模型文件首次准备见项目使用说明。')
    if st.button('开始自动预标注', type='primary', disabled=not requested or not actor.strip()):
        try:
            config = replace(defaults, fire_prompt=fire_prompt, smoke_prompt=smoke_prompt, box_threshold=box_threshold,
                             text_threshold=text_threshold, nms_iou=nms_iou, shortest_edge=short_edge,
                             longest_edge=min(1333, round(short_edge * 1.666)))
            progress = st.progress(0.0, text='正在加载本机模型…')
            model = load_grounder('cuda:0' if device_label == 'GPU (CUDA)' else 'cpu', str(model_directory()))
            st.session_state['annotation_last_results'] = store.prelabel(
                requested, model, config, actor, progress=lambda done, total: progress.progress(done / total, text=f'已处理 {done}/{total} 幅'))
            st.session_state['annotation_feedback'] = '预标注任务结束，请进入人工复核；候选框未自动批准。'
            st.rerun()
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            logging.getLogger(__name__).exception('Preannotation failed')
            st.error(f'预标注未完成：{exc}')
    if st.session_state.get('annotation_last_results'):
        st.dataframe([{'图片': row['asset_id'][:12], '状态': '待人工复核' if row['status'] == 'ready' else '失败',
                       '候选框数': row['box_count'], '说明': row['error'] or ''} for row in st.session_state['annotation_last_results']], hide_index=True, width='stretch')

elif mode == '人工复核' and catalog:
    asset_id = st.selectbox('复核图片', list(by_id), format_func=image_name)
    try:
        asset = store.image(asset_id)
        latest = store.latest(asset_id)
        proposals = {row['id']: row for row in store.proposals(asset_id)}
        ready = {key: row for key, row in proposals.items() if row['status'] == 'ready'}
        work_map = st.session_state.setdefault('annotation_working', {})
        if asset_id not in work_map:
            work_map[asset_id] = new_work(asset, latest, next(iter(ready.values()), None))
        if ready:
            proposal_id = st.selectbox('预标注版本', list(ready), format_func=lambda key: f"{ready[key]['created_at']} · {len(ready[key]['boxes'])}框 · {ready[key]['operator']}")
            if st.button('采用所选预标注'):
                selected = ready[proposal_id]
                work = new_work(asset, latest)
                work.update(boxes=validate_boxes(selected['boxes'], asset['width'], asset['height']), proposal_id=proposal_id)
                work_map[asset_id] = work
                st.rerun()
            with st.expander('查看该次模型与提示词记录'):
                st.json(ready[proposal_id]['metadata'])
                st.dataframe([{'类别': box['class'], '置信度': round(box.get('confidence', 0), 3), '匹配词': box.get('phrase', ''),
                               '原图坐标': box['xyxy']} for box in ready[proposal_id]['boxes']], hide_index=True)
        elif proposals:
            st.warning('此图最近预标注失败，可重新运行或手动画框；失败记录不会被当作无目标。')
        else:
            st.info('此图还没有预标注，可直接手动画框并复核。')
        if st.button('重新载入已保存版本'):
            work_map[asset_id] = new_work(asset, latest, next(iter(ready.values()), None))
            st.rerun()
        work = work_map[asset_id]
        if (latest['id'] if latest else None) != work['base_revision']:
            st.warning('此图已在其他窗口更新。可保留当前画布查看，但保存前须重新载入。')
        st.caption(f"原图 {asset['width']}×{asset['height']} · 标签坐标对应原图 · {asset_id[:16]}")
        payload = box_editor(dataset.asset_path(asset), asset['width'], asset['height'], work['boxes'], work['document_key'])
        if isinstance(payload, dict) and payload.get('document_key') == work['document_key']:
            work['boxes'] = validate_boxes(payload['boxes'], asset['width'], asset['height'])
        st.caption(f"当前画布 {len(work['boxes'])} 个框；下方保存后才写入修改记录。")
        negative = st.checkbox('我已检查整张图，确认没有 fire 或 smoke', key='negative_' + work['document_key']) if not work['boxes'] else False
        note = st.text_area('复核/修改说明', key='note_' + work['document_key'], max_chars=2000, placeholder='例如：删除灯光误检、补充左下角烟雾；或核对无误。')
        draft_col, approve_col = st.columns(2)
        draft = draft_col.button('保存草稿', disabled=not actor.strip() or not note.strip())
        approve = approve_col.button('确认复核通过', type='primary', disabled=not actor.strip() or not note.strip() or (not work['boxes'] and not negative))
        if draft or approve:
            store.save_review(asset_id, work['boxes'], actor, note, decision='approved' if approve else 'draft',
                              expected_revision=work['base_revision'], proposal_id=work['proposal_id'], negative_confirmed=negative,
                              elapsed_seconds=min(86400, time.monotonic() - work['started']))
            work_map[asset_id] = new_work(asset, store.latest(asset_id))
            st.session_state['annotation_feedback'] = '已保存人工复核版本。' if approve else '已保存草稿，尚未通过复核。'
            st.rerun()
        history = store.history(asset_id)
        if history:
            st.subheader('人工修改记录')
            st.dataframe([{'版本': row['id'][:12], '复核者': row['reviewer'], '状态': '通过' if row['decision'] == 'approved' else '草稿',
                           '新增': len(row['diff']['added']), '删除': len(row['diff']['removed']), '修改': len(row['diff']['changed']),
                           '说明': row['note'], '记录时间（UTC）': row['created_at']} for row in reversed(history)], hide_index=True, width='stretch')
        st.download_button('下载此图完整修改记录', json.dumps({'asset': asset, 'reviews': history, 'proposals': list(proposals.values())}, ensure_ascii=False, indent=2),
                           file_name=asset_id + '_audit.json', mime='application/json', on_click='ignore')
    except (OSError, ValueError) as exc:
        st.error(str(exc))

elif mode == 'YOLO导出':
    st.caption('只导出最新版本已人工通过的标签；无目标图像须明确确认，才产生空TXT。每次导出保留对应人工版本及完整修改记录。')
    approved = [row['id'] for row in catalog if row['decision'] == 'approved']
    selected = st.multiselect('导出已复核图片', approved, default=approved, format_func=image_name)
    if st.button('导出已复核标签 ZIP', disabled=not selected):
        try:
            path = store.export_labels(selected)
            st.session_state['annotation_feedback'] = f'已生成标签包 {path.name}，可在下方下载。'
            st.rerun()
        except (OSError, ValueError) as exc:
            st.error(str(exc))
    st.subheader('导出已划分的YOLO训练目录')
    st.caption('先在Dataset保存当前划分，并复核该划分的全部图片。导出images/labels的train、val、test目录及data.yaml，保持采集组隔离。')
    splits = {row['id']: row for row in dataset.splits() if row['current']}
    version = st.selectbox('用于导出的当前划分', list(splits)) if splits else None
    if st.button('生成YOLO训练数据集', disabled=not version):
        try:
            path = store.export_split(version)
            st.session_state['annotation_feedback'] = f'已生成训练目录：{path}。'
            st.rerun()
        except (OSError, ValueError) as exc:
            st.error(str(exc))
    exports = {row['id']: row for row in store.exports()}
    if exports:
        selected_export = st.selectbox('已保存导出', list(exports), format_func=lambda key: f"{'标签包' if exports[key]['kind'] == 'labels' else '训练数据'} · {exports[key]['created_at']} · {key[:20]}")
        record = exports[selected_export]
        relative = record['path'] if record['kind'] == 'labels' else record['manifest'].get('zip_path')
        st.caption(f"本机保存位置：{dataset.root / record['path']}")
        if relative:
            path = (dataset.root / relative).resolve()
            if path.is_relative_to(dataset.root / 'exports') and path.is_file():
                st.download_button('下载所选导出 ZIP', path.read_bytes(), file_name=path.name, mime='application/zip', on_click='ignore')
            else:
                st.warning('导出文件不在预期目录中，请检查完整数据目录。')
        else:
            st.info('图像总量超过128MB，训练目录已生成；请在本机目录中使用，未在网页内存中打包。')
