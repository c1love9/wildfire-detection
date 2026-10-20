"""Local internship data collection, catalog, progress and grouped splits."""

from pathlib import Path
import json
import sqlite3
import uuid

# Keep the same native-library order when this is the first opened page.
import torch
import streamlit as st

from camera_source import CameraConfig
from data_collection import capture_clip, capture_snapshot, extract_frames, import_paths
from dataset_store import DatasetStore, IMAGE_SUFFIXES, VIDEO_SUFFIXES, ORIGINS, SCENES, csv_bytes, dataset_root


ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title='数据管理', page_icon='📚', layout='wide')
st.title('数据采集与管理')
st.caption('素材保存在本机。来源与场景由采集者声明，数量统计不代替人工核验；演示/测试素材不计入正式进度。')
try:
    store = DatasetStore(dataset_root(ROOT))
except (OSError, sqlite3.Error):
    st.error('无法打开数据目录，请检查目录权限、磁盘空间与数据清单是否完整。')
    st.stop()
import_tab, media_tab, split_tab = st.tabs(['导入与采集', '素材与抽帧', '统计与划分'])

with import_tab:
    with st.expander('新建采集批次', expanded=not store.batches()):
        with st.form('new_dataset_batch'):
            batch_name = st.text_input('批次名称', placeholder='例如：10月20日室外白天第一批')
            group_key = st.text_input('采集组编号', placeholder='同一场景、同次拍摄、同一原视频的素材使用相同编号')
            origin_label = st.selectbox('素材来源', ['请选择', *ORIGINS.values()])
            source_ref = st.text_input('来源说明', placeholder='采集日期/设备；公开数据须写数据集名称、链接与许可说明')
            content = st.multiselect('场景内容声明', ['fire', 'smoke', 'normal', 'unknown'], default=['unknown'],
                                    help='fire火焰，smoke烟雾，normal正常无火，unknown尚未核实。此处不是目标框标注。')
            columns = st.columns(len(SCENES))
            scene = {key: column.selectbox(label, options, index=len(options) - 1) for column, (key, options), label in
                     zip(columns, SCENES.items(), ['室内外', '光照', '火灾规模', '拍摄距离', '拍摄视角'])}
            if st.form_submit_button('创建批次'):
                try:
                    origin = next((key for key, value in ORIGINS.items() if value == origin_label), '')
                    st.session_state['dataset_active_batch'] = store.create_batch(batch_name, group_key, origin, source_ref, content, scene)
                    st.success('已创建批次。')
                except ValueError as exc:
                    st.error(str(exc))
    batches = {row['id']: row for row in store.batches()}
    batch_id = st.selectbox('当前导入批次', list(batches), key='dataset_active_batch',
                            format_func=lambda key: f"{batches[key]['name']} · {ORIGINS[batches[key]['origin']]} · 组 {batches[key]['group_key']}") if batches else None
    if batch_id:
        st.caption(f"来源：{batches[batch_id]['source_ref']} · 内容声明：{', '.join(json.loads(batches[batch_id]['content_json']))}")
    else:
        st.info('先创建批次，填写真实来源和采集组，再导入或采集。')
    uploads = st.file_uploader('选择图片或视频（可多选）', type=sorted(suffix[1:] for suffix in IMAGE_SUFFIXES | VIDEO_SUFFIXES),
                               accept_multiple_files=True, disabled=batch_id is None, key=f'dataset_upload_{batch_id or "none"}')
    if st.button('导入上传文件', disabled=not batch_id or not uploads):
        results = []
        with st.spinner('正在校验并归档素材…'):
            for upload in uploads:
                temporary = store.root / (uuid.uuid4().hex + Path(upload.name).suffix.lower())
                try:
                    temporary.write_bytes(upload.getbuffer())
                    result = import_paths(store, [temporary], batch_id, source_names={temporary.name: upload.name})[0]
                    results.append(result)
                finally:
                    temporary.unlink(missing_ok=True)
        st.session_state['dataset_import_results'] = results
    with st.expander('从本机文件夹批量导入'):
        directory_text = st.text_input('素材文件夹绝对路径')
        recursive = st.checkbox('包含子文件夹', value=True)
        st.caption('整次导入使用当前采集组。不同拍摄场景请分别建批次；同一原视频剪出的片段仍须填写同一组编号。源文件不会被修改。')
        if st.button('导入文件夹', disabled=not batch_id):
            try:
                directory = Path(directory_text).resolve()
                if not directory_text or not directory.is_dir() or directory.is_relative_to(store.root):
                    raise ValueError('请选择数据归档目录以外的有效素材文件夹。')
                paths = sorted(path for path in (directory.rglob('*') if recursive else directory.glob('*'))
                               if path.is_file() and not path.is_symlink() and not path.resolve().is_relative_to(store.root)
                               and path.suffix.lower() in IMAGE_SUFFIXES | VIDEO_SUFFIXES)
                if not paths or len(paths) > 5000:
                    raise ValueError('文件夹中须有1～5000个受支持素材；过多时请分批导入。')
                with st.spinner(f'正在校验 {len(paths)} 个文件…'):
                    st.session_state['dataset_import_results'] = import_paths(store, paths, batch_id)
            except (OSError, ValueError) as exc:
                st.error(str(exc))
    if st.session_state.get('dataset_import_results'):
        st.dataframe(st.session_state['dataset_import_results'], hide_index=True, width='stretch')
    with st.expander('USB摄像头采集'):
        st.caption('点击后才会打开摄像头并保存原始画面，不叠加检测框。截图或短片结束后自动释放设备；须选择自采或演示批次。')
        index = st.number_input('采集摄像头编号', min_value=0, max_value=9, value=0, step=1)
        seconds = st.number_input('短片采集时长（秒）', min_value=1.0, max_value=30.0, value=5.0, step=1.0)
        snapshot_col, clip_col = st.columns(2)
        snapshot_clicked = snapshot_col.button('采集一张截图', disabled=not batch_id)
        clip_clicked = clip_col.button('采集一段短片', disabled=not batch_id)
        mode = 'snapshot' if snapshot_clicked else 'clip' if clip_clicked else None
        if mode:
            try:
                with st.spinner('正在采集，完成后会自动关闭摄像头…'):
                    config = CameraConfig(index=int(index))
                    result = capture_snapshot(store, batch_id, config) if mode == 'snapshot' else capture_clip(store, batch_id, seconds, config)
                st.success('采集已保存。' if result['added'] else '采集完成，重复内容已关联至现有素材。')
            except (OSError, RuntimeError, ValueError) as exc:
                st.error(str(exc))

with media_tab:
    assets = store.assets()
    if not assets:
        st.info('尚无素材。')
    else:
        by_id = {row['id']: row for row in assets}
        asset_id = st.selectbox('选择归档素材', list(by_id), format_func=lambda key: f"{'图片' if by_id[key]['kind'] == 'image' else '视频'} · {key[:16]}")
        asset = by_id[asset_id]
        st.caption(f"{asset['width']}×{asset['height']} · 文件SHA-256：{asset['sha256']}")
        if st.checkbox('预览所选素材'):
            try:
                path = store.asset_path(asset)
                st.image(str(path), width='content') if asset['kind'] == 'image' else st.video(str(path))
            except OSError as exc:
                st.error(str(exc))
        st.dataframe([row for row in store.manifest_rows() if row['asset_id'] == asset_id], hide_index=True, width='stretch')
        if asset['kind'] == 'video':
            linked_batches = sorted({row['batch_id'] for row in store.snapshot()['occurrences'] if row['asset_id'] == asset_id})
            parent_batch = st.selectbox('抽帧沿用批次', linked_batches, format_func=lambda key: batches[key]['name'])
            interval = st.number_input('抽帧间隔（秒）', min_value=0.1, max_value=3600.0, value=1.0, step=0.1)
            maximum = st.number_input('单次抽帧上限', min_value=1, max_value=2000, value=500, step=1)
            if st.button('从所选视频抽帧'):
                try:
                    with st.spinner('按视频原始时间戳抽帧并去重…'):
                        result = extract_frames(store, asset_id, parent_batch, interval, int(maximum))
                    st.success(f"抽取 {result['sampled']} 帧，新增 {result['added']} 张，重复 {result['duplicates']} 张。来源与采集组自动继承。")
                except (OSError, ValueError) as exc:
                    st.error(str(exc))

with split_tab:
    st.button('刷新统计')
    stats = store.statistics()
    columns = st.columns(4)
    for column, title, value in zip(columns, ['去重图片', '去重视频', '视频抽帧图片', '重复来源记录'],
                                     [stats['total_images'], stats['total_videos'], stats['derived_images'], stats['duplicate_occurrences']]):
        column.metric(title, value)
    targets = [{'指标（依据来源及场景声明）': title, '当前数量': round(current, 3), '题目目标': target,
                '数量进度': '达到数量，仍须核实来源/质量' if current >= target else '尚不足'}
               for title, current, target in [('自采火烟图像（张）', stats['own_fire_images'], 500), ('自采火烟视频（段）', stats['own_fire_videos'], 300),
                                              ('自采火烟视频总时长（秒）', stats['own_fire_video_seconds'], 600),
                                              ('正常场景图像（张）', stats['normal_images'], 200), ('自采占比（%）', stats['own_ratio'] * 100, 30)]]
    st.dataframe(targets, hide_index=True, width='stretch')
    st.caption(f"正式候选素材 {stats['eligible_assets']} 条 · 排除演示/测试或混合演示来源 {stats['excluded_demo_or_test']} 条 · 来源冲突 {stats['provenance_conflicts']} 条 · 场景冲突 {stats['scene_conflicts']} 条。")
    st.caption('占比分母为去重图像（含抽帧）与原视频条目之和；重复视频时长只计一次。来源冲突不计为自采，场景冲突不计入火烟/正常数量。统计尚未经过目标框人工复核。')
    st.dataframe([{'声明类别': key, '正式候选图片数': value} for key, value in stats['declared_classes'].items()], hide_index=True)
    if stats['scene_distribution']:
        st.dataframe(stats['scene_distribution'], hide_index=True, width='stretch')
    st.download_button('导出来源清单 CSV', csv_bytes(store.manifest_rows()), file_name='dataset_manifest.csv', mime='text/csv')
    st.download_button('导出数量与场景统计 JSON', json.dumps(stats, ensure_ascii=False, indent=2), file_name='dataset_statistics.json', mime='application/json')
    st.subheader('按采集组划分数据集')
    st.caption('同一采集组、原视频及抽帧、重复图片/视频关联到同一集合。至少需要3个互不关联且含图片的正式采集组。当前只生成素材清单，标注与训练将在后续阶段完成。')
    train_col, val_col, test_col, seed_col = st.columns(4)
    ratios = (train_col.number_input('训练集比例（%）', 1, 98, 70), val_col.number_input('验证集比例（%）', 1, 98, 20),
              test_col.number_input('测试集比例（%）', 1, 98, 10))
    seed = seed_col.number_input('划分随机种子', 0, 4294967295, 42)
    if st.button('生成划分预览'):
        try:
            st.session_state['dataset_split_preview'] = store.plan_split(tuple(value / 100 for value in ratios), int(seed))
        except ValueError as exc:
            st.session_state.pop('dataset_split_preview', None)
            st.error(str(exc))
    preview = st.session_state.get('dataset_split_preview')
    if preview:
        st.dataframe([{'集合': name, '图片数': count, '实际占比（%）': round(count / sum(preview['image_counts'].values()) * 100, 2)}
                      for name, count in preview['image_counts'].items()], hide_index=True)
        stale = preview['fingerprint'] != stats['fingerprint'] or preview['requested_ratios'] != [value / 100 for value in ratios] or preview['seed'] != seed
        if stale:
            st.warning('数据或参数已变化，请重新生成预览。')
        st.caption('整组划分可能偏离目标比例，不会拆开同组素材凑数。')
        if st.button('保存本次划分版本', disabled=stale):
            try:
                version = store.save_split(preview)
                st.success(f'已保存 {version}。原素材不移动，后续导入不会覆盖此版本。')
            except (OSError, ValueError) as exc:
                st.error(str(exc))
    versions = {row['id']: row for row in store.splits()}
    if versions:
        selected_version = st.selectbox('查看已保存划分', list(versions))
        st.caption('此版本与当前数据一致。' if versions[selected_version]['current'] else '这是历史快照；当前数据已更新，原版本仍可复现。')
        directory = store.root / 'splits' / selected_version
        for name in ('manifest.json', 'train.txt', 'val.txt', 'test.txt'):
            st.download_button('下载 ' + name, (directory / name).read_bytes(), file_name=name)
