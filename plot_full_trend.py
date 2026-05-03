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
                stage_rows.append({
                    'model': m_key,
                    'stage_idx': i,
                    'f1': s.get('outcome_score', {}).get('f1', 0)
                })
    
    df = pd.DataFrame(stage_rows)
    
    # 聚合
    agg = df.groupby(['model', 'stage_idx']).agg(avg_f1=('f1', 'mean'), count=('f1', 'count')).reset_index()
    agg = agg[agg['count'] >= 5] # 过滤低样本
    
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=agg, x='stage_idx', y='avg_f1', hue='model', hue_order=MODEL_ORDER, marker='o', linewidth=2.5)
    
    # 强制 X 轴为整数
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    for _, row in agg.iterrows():
        plt.text(row['stage_idx'], row['avg_f1']+0.01, f"n={int(row['count'])}", ha='center', fontsize=9)
        
    plt.title("Score Trend Including All Stages (Process + Diagnosis, N >= 5)")
    plt.ylabel("Average F1")
    plt.xlabel("Stage Index")
    plt.savefig("evaluation/plots/stage_trend_full_v4.png")
    
    print("\n包含所有阶段的纯索引趋势图已保存至 evaluation/plots/stage_trend_full_v4.png")

if __name__ == "__main__":
    main()
