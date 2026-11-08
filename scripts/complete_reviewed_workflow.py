"""Resume a frozen, reviewed training pair and its declared test protocol.

This command never creates human reviews or changes labels. The workspace must
contain workflow.json prepared before training or test inspection. Results and
model selection are saved before independent test predictions are calculated.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # Windows native libraries must load before image/Arrow imports.

from annotations import AnnotationStore
from comparison import ComparisonStore, DinoDetector
from dataset_store import DatasetStore, file_sha, now
from experiments import ExperimentStore
from grounding_prelabel import GroundingPrelabeler, PrelabelConfig
from inference_ui import load_detector
from model_catalog import fire_models
from training import TrainingStore, read_json, write_json


def wait_for_job(store, identifier):
    previous = None
    while True:
        row = store.get(identifier, refresh=True)
        state = (row['status'], row['progress'].get('completed_epochs'))
        if state != previous:
            print(json.dumps({'time': now(), 'job': identifier, 'status': state[0],
                              'progress': row['progress']}, ensure_ascii=False), flush=True)
            previous = state
        if row['status'] == 'completed':
            return store.result(identifier)
        if row['status'] not in ('running', 'queued', 'cancelling'):
            raise RuntimeError(f"{identifier}: {row['status']}: {row.get('error')}")
        time.sleep(10)


def select_by_validation(jobs, results):
    """The independent test must never enter this selection."""
    values = {name: result['metrics']['overall']['map50_95'] for name, result in results.items()}
    if any(value is None for value in values.values()):
        raise ValueError('Validation mAP50-95 is required for both variants.')
    variant = 'multiscale' if values['multiscale'] > values['baseline'] else 'baseline'
    return {'selected_at': now(), 'selection_rule': 'highest validation mAP50-95; ties choose baseline',
            'validation_map50_95': values, 'variant': variant, 'training_id': jobs[variant],
            'test_used_for_selection': False}


def run(dataset_root, workspace):
    marker = workspace / 'workflow.json'
    workflow = read_json(marker)
    dataset = DatasetStore(dataset_root)
    training, experiments = TrainingStore(dataset), ExperimentStore(dataset)
    experiment_id = workflow['experiment_id']
    manifest = experiments.read(experiment_id)
    results = {}
    for variant in ('baseline', 'multiscale'):
        if variant not in manifest['jobs']:
            experiments.start(experiment_id, variant)
            manifest = experiments.read(experiment_id)
        elif training.get(manifest['jobs'][variant],refresh=True)['status'] in ('interrupted','failed'):
            experiments.retry(experiment_id,variant)
            manifest = experiments.read(experiment_id)
        results[variant] = wait_for_job(training, manifest['jobs'][variant])
    selection_path = workspace / 'model_selection.json'
    selection = select_by_validation(manifest['jobs'], results)
    if selection_path.exists():
        saved = read_json(selection_path)
        assert saved['training_id'] == selection['training_id'], 'Do not replace a frozen model selection.'
        selection = saved
    else:
        write_json(selection_path, selection)
    print(json.dumps({'selected_before_test': selection}, ensure_ascii=False), flush=True)

    # Explicitly freeze the actual Tiny implementation: one class-conditioned
    # query per class. It is not the Base+YOLO fusion used in enhanced prelabels.
    protocol_path = workspace / 'open_model_protocol.json'
    config = PrelabelConfig(fire_prompt='fire.', smoke_prompt='smoke.')
    from dataclasses import asdict
    protocol = {'created_at': now(), 'model': asdict(config), 'queries': config.prompts(),
                'implementation': 'DinoDetector forces 640x640 and AP candidate threshold 0.001',
                'class_strategy': 'independent per-class queries; not joint Base+YOLO fusion',
                'device': 'cuda:0', 'fine_tuning_samples_in_this_project': 0,
                'preannotation_dependency': 'Tiny and Base DINO contributed earlier candidates; final labels were user-approved.'}
    if not protocol_path.exists():
        write_json(protocol_path, protocol)

    pair = experiments.read(experiment_id)
    evaluation = experiments.evaluation(experiment_id) if pair['evaluation_sha'] else experiments.evaluate(experiment_id, 'cuda:0')
    load_detector.clear() if hasattr(load_detector, 'clear') else None
    # The inner Streamlit resource cache otherwise retains both YOLO variants.
    from inference_ui import _load
    _load.clear()
    gc.collect()
    torch.cuda.empty_cache()

    comparison_path = workspace / 'comparison_id.json'
    comparison = ComparisonStore(dataset)
    if comparison_path.exists():
        comparison_id = read_json(comparison_path)['id']
        comparison_result = comparison.result(comparison_id)
    else:
        choice = fire_models(ROOT, dataset.root)['trained:' + selection['training_id'] + ':best']
        yolo = load_detector(choice, 'cuda:0', 640)
        dino = DinoDetector(GroundingPrelabeler('cuda:0'), config)
        comparison_id = comparison.run(selection['training_id'], yolo, dino, '自动实验执行',
                                       '人工复核700幅的冻结test；先按val选模；Tiny独立fire./smoke.提示，未微调，不使用辅助YOLO融合。',
                                       progress=lambda value: print(f'comparison_progress={value:.3f}', flush=True))
        write_json(comparison_path, {'id': comparison_id})
        comparison_result = comparison.result(comparison_id)

    rows = []
    for name, result in evaluation['results'].items():
        rows.append({'experiment': name, **result['overall'], 'mean_ms': result['mean_ms'],
                     'negative_images': result['negative_images'],
                     'negative_false_positives': result['negative_images_with_false_detection']})
    for name, result in comparison_result['models'].items():
        rows.append({'experiment': 'comparison_' + name, **result['overall'],
                     'mean_ms': result['mean_ms_per_image'], 'negative_images': result['negative_images'],
                     'negative_false_positives': result['negative_images_with_false_detection']})
    with (workspace / 'formal_metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {'status': 'completed', 'created_at': now(), 'experiment_id': experiment_id,
              'jobs': manifest['jobs'], 'selection': selection, 'comparison_id': comparison_id,
              'test_images': len(evaluation['reference']), 'rows': rows,
              'difference_multiscale_minus_baseline': evaluation['difference_multiscale_minus_baseline'],
              'workflow_sha256': file_sha(marker),
              'limitations': ['Test split has only one approved normal image; its false-positive rate is not stable.',
                              'Known source groups are separated, but full public-pretraining/source-event overlap is unknown.',
                              'No independent manual masks or measured assisted-review worktime have been supplied.']}
    write_json(workspace / 'formal_experiments.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=ROOT / 'data/fire_dataset')
    parser.add_argument('--workspace', type=Path, required=True)
    args = parser.parse_args()
    run(args.dataset_root.resolve(), args.workspace.resolve())
