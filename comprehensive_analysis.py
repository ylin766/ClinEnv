#!/usr/bin/env python3
import json
import os
from pathlib import Path
from collections import defaultdict
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 设置绘图风格
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # 支持中文
plt.rcParams['axes.unicode_minus'] = False

# 模型映射
MODELS = {
    'gpt-5.4': 'gpt-5.4',
    'mini': 'gpt-5.4-mini',
    'nano': 'gpt-5.4-nano',
    'oss-20b': '-home-hice1-ylin766-bmed-sp-wang-YuxingData-Checkpoints-vllm_models-gpt-oss-20b'
}

def calculate_case_metrics(stages):
    """计算整个case的综合指标"""
    total_hits = 0
    total_gt = 0
    total_pred = 0
    
    nurse_cov = []
    patient_cov = []
    lab_cov = []
    efficiency = []
    lab_waste = []
    
    type_stats = defaultdict(lambda: {'hits': 0, 'total': 0, 'pred': 0})

    for stage in stages:
        # Outcome
        outcome = stage.get('outcome_score', {})
        total_hits += outcome.get('hits', 0)
        total_gt += outcome.get('total', 0)
        prec = outcome.get('precision', 0)
        if prec > 0:
            total_pred += outcome.get('hits', 0) / prec
            
        # Type specific
        for match in outcome.get('matches', []):
            t = match.get('gt_item', {}).get('type', 'unknown')
            type_stats[t]['hits'] += match.get('score', 0)
            type_stats[t]['total'] += 1
            # Pred count is tricky with hungarian matches, but we can approximate
            
        # Process
        process = stage.get('process_score', {})
        per_speaker = process.get('info_coverage', {}).get('per_speaker', {})
        nurse_cov.append(per_speaker.get('nurse', {}).get('coverage', 0))
        patient_cov.append(per_speaker.get('patient', {}).get('coverage', 0))
        lab_cov.append(per_speaker.get('lab', {}).get('coverage', 0))
        efficiency.append(process.get('efficiency', 0))
        
        lab_cost = stage.get('lab_cost', {})
        lab_waste.append(lab_cost.get('wasted_ratio', 0))

    recall = total_hits / total_gt if total_gt > 0 else 0
    precision = total_hits / total_pred if total_pred > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Calculate type F1s
    type_f1s = {}
    for t, s in type_stats.items():
        type_f1s[t] = s['hits'] / s['total'] if s['total'] > 0 else 0

    return {
        'f1': f1,
        'recall': recall,
        'precision': precision,
        'nurse_cov': np.mean(nurse_cov) if nurse_cov else 0,
        'patient_cov': np.mean(patient_cov) if patient_cov else 0,
        'lab_cov': np.mean(lab_cov) if lab_cov else 0,
        'efficiency': np.mean(efficiency) if efficiency else 0,
        'lab_waste': np.mean(lab_waste) if lab_waste else 0,
        'type_f1s': type_f1s
    }

def load_data():
    manifest_path = Path("data/aligned_101_manifest.jsonl")
    cases_manifest = [json.loads(line) for line in manifest_path.read_text().strip().split('\n')]
    
    all_results = []
    for case in cases_manifest:
        case_id = f"{case['subject_id']}_{case['hadm_id']}"
        for model_key, model_name in MODELS.items():
            eval_file = Path(f"data/cases/{case['subject_id']}/{case['hadm_id']}/evals/{model_name}/latest.json")
            if eval_file.exists():
                data = json.loads(eval_file.read_text())
                stages = data.get('stages', [])
                metrics = calculate_case_metrics(stages)
                
                # Add stage-level data for trend analysis
                stage_data = []
                for i, stage in enumerate(stages[:-1]): # Exclude diagnosis stage
                    outcome = stage.get('outcome_score', {})
                    process = stage.get('process_score', {})
                    context_range = stage.get('context_range', [])
                    length = context_range[1] - context_range[0] if len(context_range) == 2 else 0
                    
                    stage_data.append({
                        'stage_idx': i,
                        'f1': outcome.get('f1', 0),
                        'length': length
                    })
                
                res = {
                    'case_id': case_id,
                    'model': model_key,
                    'overall_f1': metrics['f1'],
                    'nurse_cov': metrics['nurse_cov'],
                    'patient_cov': metrics['patient_cov'],
                    'lab_cov': metrics['lab_cov'],
                    'efficiency': metrics['efficiency'],
                    'lab_waste': metrics['lab_waste'],
                    'type_f1s': metrics['type_f1s'],
                    'stages': stage_data
                }
                all_results.append(res)
    return pd.DataFrame(all_results)

def plot_scale_effect(df):
    """1. 所有维度分析的一个图，证明scale效应 (Radar Chart)"""
    metrics = ['overall_f1', 'nurse_cov', 'patient_cov', 'lab_cov', 'efficiency']
    labels = ['Outcome F1', 'Nurse Coverage', 'Patient Coverage', 'Lab Coverage', 'Efficiency']
    
    # 归一化均值
    model_means = df.groupby('model')[metrics].mean().reindex(['nano', 'mini', 'gpt-5.4', 'oss-20b'])
    
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
    for i, (model, row) in enumerate(model_means.iterrows()):
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, color=colors[i], linewidth=2, label=model)
        ax.fill(angles, values, color=colors[i], alpha=0.25)
        
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Model Performance Across Dimensions (Scale Effect)")
    plt.savefig("evaluation/plots/scale_effect_radar.png", bbox_inches='tight')
    plt.close()

def plot_stage_scores(df):
    """2. 分析每个模型在不同stage的平均得分 (排除最后一个stage)"""
    stage_rows = []
    for _, row in df.iterrows():
        for s in row['stages']:
            stage_rows.append({
                'model': row['model'],
                'stage_idx': s['stage_idx'],
                'f1': s['f1']
            })
    stage_df = pd.DataFrame(stage_rows)
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=stage_df, x='stage_idx', y='f1', hue='model', marker='o', 
                 hue_order=['nano', 'mini', 'gpt-5.4', 'oss-20b'])
    plt.title("Average Score per Stage (Excluding Diagnosis)")
    plt.xlabel("Stage Index")
    plt.ylabel("Average F1")
    plt.savefig("evaluation/plots/stage_scores.png")
    plt.close()

def plot_length_effect(df):
    """3. 单个stage的长度对不同模型得分的影响"""
    stage_rows = []
    for _, row in df.iterrows():
        for s in row['stages']:
            stage_rows.append({
                'model': row['model'],
                'length': s['length'],
                'f1': s['f1']
            })
    len_df = pd.DataFrame(stage_rows)
    # Binning lengths
    len_df['length_bin'] = pd.cut(len_df['length'], bins=[0, 20, 50, 100, 500, 2000], 
                                  labels=['0-20', '20-50', '50-100', '100-500', '500+'])
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=len_df, x='length_bin', y='f1', hue='model',
                hue_order=['nano', 'mini', 'gpt-5.4', 'oss-20b'])
    plt.title("Impact of Stage Length on F1 Score")
    plt.xlabel("Context Length (Number of Events)")
    plt.ylabel("Average F1")
    plt.savefig("evaluation/plots/length_effect.png")
    plt.close()

def plot_source_preference(df):
    """4. 信息来源偏好"""
    metrics = ['nurse_cov', 'patient_cov', 'lab_cov']
    source_means = df.groupby('model')[metrics].mean().reindex(['nano', 'mini', 'gpt-5.4', 'oss-20b'])
    
    source_means.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title("Information Source Preference (Coverage)")
    plt.ylabel("Sum of Coverage")
    plt.savefig("evaluation/plots/source_preference.png")
    plt.close()

def plot_task_performance(df):
    """6. 不同任务下不同模型的表现变化 (Medication, Procedure, Diagnosis)"""
    task_rows = []
    for _, row in df.iterrows():
        for task, score in row['type_f1s'].items():
            if task in ['medication', 'procedure', 'diagnosis']:
                task_rows.append({
                    'model': row['model'],
                    'task': task,
                    'f1': score
                })
    task_df = pd.DataFrame(task_rows)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=task_df, x='task', y='f1', hue='model',
                hue_order=['nano', 'mini', 'gpt-5.4', 'oss-20b'])
    plt.title("Performance by Task Type")
    plt.savefig("evaluation/plots/task_performance.png")
    plt.close()

def analyze_overlap(df):
    """死者重叠 (Overlap of shared failures/difficult cases)"""
    # 这里的“死者”指代 F1 < 0.2 的案例
    df['is_failure'] = df['overall_f1'] < 0.2
    
    pivot = df.pivot(index='case_id', columns='model', values='is_failure').dropna()
    
    models = ['nano', 'mini', 'gpt-5.4', 'oss-20b']
    overlap_matrix = np.zeros((4, 4))
    
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            shared = (pivot[m1] & pivot[m2]).sum()
            overlap_matrix[i, j] = shared
            
    plt.figure(figsize=(8, 6))
    sns.heatmap(overlap_matrix, annot=True, xticklabels=models, yticklabels=models, cmap="YlOrRd")
    plt.title("Shared Failures (F1 < 0.2) Overlap Count")
    plt.savefig("evaluation/plots/failure_overlap.png")
    plt.close()
    
    print("\n【死者重叠分析 (F1 < 0.2)】")
    print(overlap_matrix)

def main():
    print("正在加载数据...")
    df = load_data()
    
    if not os.path.exists("evaluation/plots"):
        os.makedirs("evaluation/plots")
        
    print("正在生成绘图...")
    plot_scale_effect(df)
    plot_stage_scores(df)
    plot_length_effect(df)
    plot_source_preference(df)
    plot_task_performance(df)
    analyze_overlap(df)
    
    print("分析完成。图片已保存至 evaluation/plots/")

if __name__ == "__main__":
    main()
