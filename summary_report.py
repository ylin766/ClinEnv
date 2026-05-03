#!/usr/bin/env python3
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

# 设置绘图风格
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

MODELS = {
    'gpt-5.4': 'gpt-5.4',
    'mini': 'gpt-5.4-mini',
    'nano': 'gpt-5.4-nano',
    'oss-20b': '-home-hice1-ylin766-bmed-sp-wang-YuxingData-Checkpoints-vllm_models-gpt-oss-20b'
}

MODEL_ORDER = ['gpt-5.4', 'mini', 'nano', 'oss-20b']

def extract_detailed_metrics(stages):
    metrics = {
        'f1_overall': 0, 'f1_medication': [], 'f1_procedure': [], 'f1_diagnosis': [],
        'nurse_cov': [], 'patient_cov': [], 'lab_cov': [], 'efficiency': [], 'lab_waste': []
    }
    total_hits, total_gt, total_pred = 0, 0, 0
    stage_data = []

    for i, stage in enumerate(stages):
        outcome = stage.get('outcome_score', {})
        total_hits += outcome.get('hits', 0)
        total_gt += outcome.get('total', 0)
        prec = outcome.get('precision', 0)
        if prec > 0: total_pred += outcome.get('hits', 0) / prec
            
        task_hits = defaultdict(float)
        task_total = defaultdict(int)
        for match in outcome.get('matches', []):
            t = match.get('gt_item', {}).get('type', 'unknown')
            task_hits[t] += match.get('score', 0)
            task_total[t] += 1
        
        for t in ['medication', 'procedure', 'diagnosis']:
            if task_total.get(t, 0) > 0:
                metrics[f'f1_{t}'].append(task_hits[t] / task_total[t])

        process = stage.get('process_score', {})
        per_speaker = process.get('info_coverage', {}).get('per_speaker', {})
        metrics['nurse_cov'].append(per_speaker.get('nurse', {}).get('coverage', 0))
        metrics['patient_cov'].append(per_speaker.get('patient', {}).get('coverage', 0))
        metrics['lab_cov'].append(per_speaker.get('lab', {}).get('coverage', 0))
        metrics['efficiency'].append(process.get('info_coverage', {}).get('efficiency', 0))
        
        # 记录非诊断阶段得分用于趋势图
        if i < len(stages) - 1:
            stage_data.append({'stage_idx': i, 'f1': outcome.get('f1', 0)})

    recall = total_hits / total_gt if total_gt > 0 else 0
    precision = total_hits / total_pred if total_pred > 0 else 0
    metrics['f1_overall'] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    final_metrics = {'f1_overall': metrics['f1_overall'], 'stages': stage_data}
    for k in ['f1_medication', 'f1_procedure', 'f1_diagnosis', 'nurse_cov', 'patient_cov', 'lab_cov', 'efficiency']:
        final_metrics[k] = np.mean(metrics[k]) if metrics[k] else 0
        
    return final_metrics

def main():
    manifest = [json.loads(line) for line in Path('data/aligned_101_manifest.jsonl').read_text().strip().split('\n')]
    results, stage_rows = [], []
    
    for case in manifest:
        sid, hid = case['subject_id'], case['hadm_id']
        has_all = all((Path('data/cases') / sid / hid / 'evals' / m / 'latest.json').exists() for m in MODELS.values())
        if not has_all: continue
            
        for m_key, m_name in MODELS.items():
            data = json.loads((Path('data/cases') / sid / hid / 'evals' / m_name / 'latest.json').read_text())
            res = extract_detailed_metrics(data.get('stages', []))
            for s in res.pop('stages'):
                stage_rows.append({'model': m_key, 'stage_idx': s['stage_idx'], 'f1': s['f1']})
            res['model'] = m_key
            results.append(res)
    
    df, stage_df = pd.DataFrame(results), pd.DataFrame(stage_rows)
    
    # 1. 打印详细表格
    summary = df.groupby('model').mean(numeric_only=True).reindex(MODEL_ORDER)
    print("\n【四模型细化指标对比表】\n" + "-" * 140)
    cols = ['f1_overall', 'f1_medication', 'f1_procedure', 'f1_diagnosis', 'nurse_cov', 'patient_cov', 'lab_cov', 'efficiency']
    print(f"{'模型':<12}" + "".join([f"{c:<16}" for c in cols]) + "\n" + "-" * 140)
    for model in MODEL_ORDER:
        row = summary.loc[model]
        print(f"{model:<12}" + "".join([f"{row[c]:<16.3f}" for c in cols]))
    print("-" * 140)

    # 2. 绘制 Outcome & Process
    for name, metrics in [('outcome', ['f1_medication', 'f1_procedure', 'f1_diagnosis', 'f1_overall']), 
                          ('process', ['nurse_cov', 'patient_cov', 'lab_cov', 'efficiency'])]:
        plt.figure(figsize=(14, 6))
        m_df = df.melt(id_vars='model', value_vars=metrics, var_name='Metric', value_name='Score')
        sns.barplot(data=m_df, x='Metric', y='Score', hue='model', hue_order=MODEL_ORDER)
        plt.title(f"{name.capitalize()} Scores Comparison")
        plt.savefig(f"evaluation/plots/detailed_{name}_v3.png")

    # 3. 优化后的 Stage 趋势图
    agg_stage = stage_df.groupby(['model', 'stage_idx']).agg(avg_f1=('f1', 'mean'), count=('f1', 'count')).reset_index()
    agg_stage = agg_stage[agg_stage['count'] >= 5]
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=agg_stage, x='stage_idx', y='avg_f1', hue='model', hue_order=MODEL_ORDER, marker='o', linewidth=2.5)
    
    # 强制 X 轴为整数
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    for _, row in agg_stage.iterrows():
        plt.text(row['stage_idx'], row['avg_f1']+0.01, f"n={int(row['count'])}", ha='center', fontsize=9)
    plt.title("Average Score per Stage (N >= 5, Excluding Diagnosis)")
    plt.savefig("evaluation/plots/stage_trend_v3.png")
    
    print("\n新图表已保存至 evaluation/plots/ (版本 v3)")

if __name__ == "__main__":
    main()
