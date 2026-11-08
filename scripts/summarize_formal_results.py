"""Summarize real frozen exports and completed experiments without editing labels."""
import argparse
import collections
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from comparison import evaluate_predictions
from dataset_store import now
from training import read_json, write_json, history_rows


def save_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def summarize(workspace):
    workflow = read_json(workspace/'workflow.json')
    formal = read_json(workspace/'formal_experiments.json')
    export = Path(workflow['export_path'])
    snapshots = read_json(export/'audit.json')
    split = read_json(export/'source_split.json')
    source = read_json(ROOT/'data/minimum_collection/registration.json')['records']
    provenance = {row['asset_id']: row for row in source}
    splits = {row['id']: row['split'] for row in split['assets']}
    counts = collections.Counter()
    by_split, by_source, scenes = {}, {}, {}
    rows = []
    for row in snapshots:
        key = row['asset']['id']; boxes = row['review']['boxes']; origin = provenance[key]
        values = collections.Counter(images=1, positive=int(bool(boxes)), negative=int(not boxes))
        values.update({'images_'+label: int(any(b['class']==label for b in boxes)) for label in ('fire','smoke')})
        values.update('boxes_'+b['class'] for b in boxes)
        counts.update(values)
        if origin.get('accounting_origin')=='own' and boxes:
            counts.update(['approved_own_positive_images'])
        by_split.setdefault(splits[key], collections.Counter()).update(values)
        by_source.setdefault(origin['source'], collections.Counter()).update(values)
        for dimension, value in origin['scene'].items():
            scenes.setdefault(dimension, collections.Counter()).update([value])
        rows.append({'asset_id': key, 'split': splits[key], 'source': origin['source'],
                     'positive': bool(boxes), 'fire_boxes': sum(b['class']=='fire' for b in boxes),
                     'smoke_boxes': sum(b['class']=='smoke' for b in boxes),
                     'review_revision': row['review']['id'], 'reviewer': row['review']['reviewer'],
                     'sha256': row['asset']['sha256'], **{'declared_'+k: v for k,v in origin['scene'].items()}})
    summary = {'created_at': now(), 'export_id': workflow['export_id'], 'split_id': workflow['split_id'],
               'totals': dict(counts), 'by_split': {k: dict(v) for k,v in by_split.items()},
               'by_source': {k: dict(v) for k,v in by_source.items()},
               'source_declared_scene_profiles': {k: dict(v) for k,v in scenes.items()},
               'scene_profiles_human_verified_per_image': False,
               'videos': sum(row['kind']=='video' for row in source),
               'video_seconds': sum(float(row.get('duration', 0)) for row in source if row['kind']=='video'),
               'accounting_basis': 'Teacher reportedly accepts public video curation as self-collection; original provenance remains public.'}
    write_json(workspace/'final_dataset_statistics.json', summary)
    save_csv(workspace/'final_image_statistics.csv', rows)
    curves = {}
    for variant, identifier in formal['jobs'].items():
        directory = ROOT/'data/fire_dataset/training'/identifier
        history = history_rows(directory)
        best = max(history, key=lambda r:r['metrics/mAP50-95(B)'])
        curves[variant] = {'completed_epochs': len(history), 'best_validation_epoch_csv': int(best['epoch']),
                           'best_validation_map50_95_csv': best['metrics/mAP50-95(B)'],
                           'last_validation_map50_95': history[-1]['metrics/mAP50-95(B)'],
                           'last_10_validation_range': [min(r['metrics/mAP50-95(B)'] for r in history[-10:]),max(r['metrics/mAP50-95(B)'] for r in history[-10:])],
                           'train_box_loss_first_5_median': statistics.median(r['train/box_loss'] for r in history[:5]),
                           'train_box_loss_last_5_median': statistics.median(r['train/box_loss'] for r in history[-5:]),
                           'val_box_loss_first_5_median': statistics.median(r['val/box_loss'] for r in history[:5]),
                           'val_box_loss_last_5_median': statistics.median(r['val/box_loss'] for r in history[-5:]),
                           'training_wall_seconds': read_json(directory/'result.json')['metrics']['training_wall_seconds']}
    write_json(workspace/'convergence_analysis.json', curves)
    comparison = read_json(ROOT/'data/fire_dataset/comparisons'/formal['comparison_id']/'result.json')
    stratified = []
    for dimension in ('source', 'setting', 'lighting', 'distance', 'view'):
        groups = collections.defaultdict(list)
        for key in comparison['references']:
            record = provenance[key]
            groups[record['source'] if dimension=='source' else record['scene'].get(dimension, '未说明')].append(key)
        for value, keys in groups.items():
            refs = {key: comparison['references'][key] for key in keys}
            for name, predictions in comparison['predictions'].items():
                metric = evaluate_predictions(refs, {key: predictions[key] for key in keys})
                stratified.append({'dimension': dimension, 'source_declared_value': value,
                                   'human_scene_confirmation': False, 'model': name, 'images': len(keys),
                                   'gt_instances': sum(len(v) for v in refs.values()), **metric['overall']})
    save_csv(workspace/'scene_generalization.csv', stratified)
    save_csv(workspace/'device_generalization.csv', [{'device_identity': 'unknown', 'images': len(comparison['references']),
              'status': 'Not established: source dataset names are not independent camera identities.'}])
    negative_rows = []
    for key, boxes in comparison['references'].items():
        if not boxes:
            for model, predictions in comparison['predictions'].items():
                detections = [b for b in predictions[key] if b['confidence']>=.25]
                negative_rows.append({'asset_id': key, 'model': model, 'predicted_boxes': len(detections),
                                      'false_positive_image': bool(detections)})
    save_csv(workspace/'negative_false_alarms.csv', negative_rows)
    write_json(ROOT/'reports/formal_experiments_20261108.json', formal)
    write_json(ROOT/'reports/final_dataset_statistics_20261108.json', summary)
    percentage = lambda value: '未记录' if value is None else f'{value*100:.2f}%'
    lines = ['# 人工复核后的正式训练与评测结果', '', '日期：2026年11月8日。所有指标来自真实训练与冻结测试副本；不使用合成测试数字。', '',
             f"最终已复核700幅：含火/烟{counts['positive']}幅，正常{counts['negative']}幅；fire框{counts['boxes_fire']}个、smoke框{counts['boxes_smoke']}个。正常样本距200幅还差{max(0,200-counts['negative'])}幅。300段视频共900秒，来自46个公开原视频，不代表300个独立事件。", '',
             '|集合|图像|含目标|正常|fire框|smoke框|','|---|---:|---:|---:|---:|---:|']
    for name in ('train','val','test'):
        x = by_split[name]; lines.append(f"|{name}|{x['images']}|{x['positive']}|{x['negative']}|{x['boxes_fire']}|{x['boxes_smoke']}|")
    lines += ['', '两组从同一COCO YOLOv8n初始化，640、batch8、AdamW、lr0=0.001、seed42、FP32、50轮、关闭早停。仅multi_scale开关不同；内置线性学习率调度、增强、权重衰减等以各任务effective_parameters.json为准。', '',
              f"主模型在查看test前按验证集mAP50-95选为{formal['selection']['variant']}；任务{formal['selection']['training_id']}。没有按test选模或重新调阈值。", '',
              '|实验或模型|P|R|F1|mAP50|mAP50-95|平均ms|','|---|---:|---:|---:|---:|---:|---:|']
    for row in formal['rows']:
        lines.append('|'+row['experiment']+'|'+'|'.join(percentage(row[k]) for k in ('precision','recall','f1','map50','map50_95'))+f"|{row['mean_ms']:.2f}|")
    delta = formal['difference_multiscale_minus_baseline']
    lines += ['', f"多尺度相对固定尺寸的mAP50-95变化为{delta['map50_95']*100:+.2f}个百分点，F1变化为{delta['f1']*100:+.2f}个百分点。优化是否有效由这些实际差值判断，不能只保留有利指标。", '',
              '统一评测640×640等比例填充、batch1、FP32、同一RTX5060 Laptop GPU；P/R/F1在conf0.25、同类一对一IoU0.5计算，按有GT类别宏平均；AP低阈值0.001，IoU0.50～0.95。GPU预热同步计时包含预/后处理，排除读盘、加载和画框。实验两组依次运行、YOLO/DINO交替运行，速度会受系统负载影响，不作严格硬件极限结论。', '',
              '对比开放域模型为固定Tiny版本，fire.和smoke.分两次独立查询、文本阈值0.25，未微调；不使用增强预标注的Base+辅助YOLO融合。DINO曾参与预标注，最终test采用用户确认版本；人工独立性和标注质量仍需按实际操作协议说明。', '',
              '|训练组|CSV最佳val轮|val mAP50-95范围 最后10轮|实际训练分钟|','|---|---:|---|---:|']
    for variant, curve in curves.items():
        a,b = curve['last_10_validation_range']
        lines.append(f"|{variant}|{curve['best_validation_epoch_csv']}|{percentage(a)}～{percentage(b)}|{curve['training_wall_seconds']/60:.2f}|")
    lines += ['', '曲线如出现训练损失继续下降、验证指标波动或验证损失偏高，应解释为泛化受限或可能过拟合；完成50轮不等于已经证明充分收敛。best来自val，实际test表现不用于再选择轮数。', '',
              '限制：test76幅只有1幅正常图，不能据此声称误报率稳定或达到工程要求。源声明场景分层见scene_generalization.csv，未逐图确认的场景不能当作人工实证；拍摄设备身份未知，未证明跨设备泛化。早期火星与真实燃烧阶段仍待确认。', '',
              '用户同意暂留待补：独立人工参考掩码与实际面积误差、有效辅助复核工时及节省率。封面身份暂不填写。A/B原始独立标注与700幅最终复核不改写、不增加强制逐图仲裁。', '']
    (ROOT/'docs/formal_results.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'statistics': summary['totals'], 'selection': formal['selection'], 'curves': curves}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--workspace', type=Path, required=True)
    summarize(parser.parse_args().workspace.resolve())
