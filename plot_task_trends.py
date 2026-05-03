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
TASKS = ['medication', 'procedure', 'diagnosis']

def main():
    manifest = [json.loads(line) for line in Path('data/aligned_101_manifest.jsonl').read_text().strip().split('\n')]
    task_rows = []
    
    for case in manifest:
        sid, hid = case['subject_id'], case['hadm_id']
        has_all = all((Path('data/cases') / sid / hid / 'evals' / m / 'latest.json').exists() for m in MODELS.values())
        if not has_all: continue
            
        for m_key, m_name in MODELS.items():
            data = json.loads((Path('data/cases') / sid / hid / 'evals' / m_name / 'latest.json').read_text())
            stages = data.get('stages', [])
            for i, s in enumerate(stages):
                outcome = s.get('outcome_score', {})
                matches = outcome.get('matches', [])
                
                # 按任务统计本 stage 的得分
                hits = defaultdict(float)
                totals = defaultdict(int)
                for m in matches:
                    t = m.get('gt_item', {}).get('type', 'unknown')
                    if t in TASKS:
                        hits[t] += m.get('score', 0)
                        totals[t] += 1
                
                for t in TASKS:
                    if totals[t] > 0:
                        task_rows.append({
                            'model': m_key,
                            'stage_idx': i,
                            'task': t,
                            'f1': hits[t] / totals[t]
                        })
    
    df = pd.DataFrame(task_rows)
    
    # 创建 1x3 子图
    fig, axes = plt.subplots(1, 3, figsize=(22, 6), sharex=True)
    
    for idx, task in enumerate(TASKS):
        ax = axes[idx]
        task_df = df[df['task'] == task]
        agg = task_df.groupby(['model', 'stage_idx']).agg(avg_f1=('f1', 'mean'), count=('f1', 'count')).reset_index()
        agg = agg[agg['count'] >= 3] # 任务级样本量放宽到 3，因为任务分布更稀疏
        
        sns.lineplot(data=agg, x='stage_idx', y='avg_f1', hue='model', hue_order=MODEL_ORDER, 
                     marker='o', linewidth=2, ax=ax)
        
        ax.set_title(f"Task: {task.capitalize()}")
        ax.set_ylabel("Average F1")
        ax.set_xlabel("Stage Index")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        # 标注样本量
        for _, row in agg.iterrows():
            ax.text(row['stage_idx'], row['avg_f1'] + 0.01, f"n={int(row['count'])}", ha='center', fontsize=8)

    plt.suptitle("Performance Trend by Task Type Across Stages", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("evaluation/plots/task_trends_by_stage.png")
    
    print("\n按任务分类的阶段趋势图已保存至 evaluation/plots/task_trends_by_stage.png")

if __name__ == "__main__":
    main()
