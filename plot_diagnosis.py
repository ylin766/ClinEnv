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

def main():
    manifest = [json.loads(line) for line in Path('data/aligned_101_manifest.jsonl').read_text().strip().split('\n')]
    stage_rows = []
    
    for case in manifest:
        sid, hid = case['subject_id'], case['hadm_id']
        has_all = all((Path('data/cases') / sid / hid / 'evals' / m / 'latest.json').exists() for m in MODELS.values())
        if not has_all: continue
            
        for m_key, m_name in MODELS.items():
            data = json.loads((Path('data/cases') / sid / hid / 'evals' / m_name / 'latest.json').read_text())
            stages = data.get('stages', [])
            for i, s in enumerate(stages):
                label = str(i)
                if i == len(stages) - 1:
                    label = "Diagnosis"
                stage_rows.append({
                    'model': m_key,
                    'stage_label': label,
                    'f1': s.get('outcome_score', {}).get('f1', 0)
                })
    
    df = pd.DataFrame(stage_rows)
    
    # 为了绘图排序，我们需要定义顺序
    # 0, 1, 2, 3, 4, Diagnosis
    all_labels = sorted([l for l in df['stage_label'].unique() if l != "Diagnosis"], key=lambda x: int(x))
    all_labels.append("Diagnosis")
    
    # 聚合
    agg = df.groupby(['model', 'stage_label']).agg(avg_f1=('f1', 'mean'), count=('f1', 'count')).reset_index()
    agg = agg[agg['count'] >= 5] # 过滤低样本
    
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=agg, x='stage_label', y='avg_f1', hue='model', hue_order=MODEL_ORDER, marker='o', linewidth=2.5)
    
    for _, row in agg.iterrows():
        plt.text(row['stage_label'], row['avg_f1']+0.01, f"n={int(row['count'])}", ha='center', fontsize=9)
        
    plt.title("Score Trend Including Final Diagnosis Stage (N >= 5)")
    plt.ylabel("Average F1")
    plt.xlabel("Stage (Process 0,1,2... -> Final Diagnosis)")
    plt.savefig("evaluation/plots/stage_trend_with_diagnosis.png")
    
    print("\n包含诊断阶段的趋势图已保存至 evaluation/plots/stage_trend_with_diagnosis.png")

if __name__ == "__main__":
    main()
