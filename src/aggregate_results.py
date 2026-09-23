"""
Results Aggregation and Statistical Significance Module
Compiles Tables 4, 5, and 6 and computes paired Student's t-tests.
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import ttest_rel
from .config import OUTPUTS_DIR, MODEL_SMALL, MODEL_LARGE

def aggregate_and_save_all_metrics(all_precisions, all_generalities, taskqa_records):
    metrics = {
        'table4_simulation_precision': {},
        'table5_simulation_generality': {},
        'table6_task_accuracy': {},
        'statistical_tests': {}
    }

    # Format Table 4
    for key, p_data in all_precisions.items():
        metrics['table4_simulation_precision'][key] = {
            'macro_precision': p_data['macro_precision'],
            'micro_precision': p_data['micro_precision'],
            'simulatable_count': p_data['total_simulatable'],
            'correct_count': p_data['total_correct'],
            'unsimulatable_count': p_data['total_unsimulatable']
        }

    # Format Table 5
    for key, g_data in all_generalities.items():
        metrics['table5_simulation_generality'][key] = g_data

    # Format Table 6 (Task Accuracy)
    for key, records in taskqa_records.items():
        valid_ans = sum(1 for r in records if r.get('pred_ans') in ['yes', 'no'])
        metrics['table6_task_accuracy'][key] = {
            'valid_answer_rate': round(valid_ans / len(records) * 100, 2) if records else 0.0,
            'total_questions': len(records)
        }

    # Statistical significance test (Small vs Large on CoT)
    key_small_cot = f'{MODEL_SMALL}_cot'
    key_large_cot = f'{MODEL_LARGE}_cot'
    if key_small_cot in all_precisions and key_large_cot in all_precisions:
        s_scores = np.nan_to_num(all_precisions[key_small_cot]['per_example_precision'])
        l_scores = np.nan_to_num(all_precisions[key_large_cot]['per_example_precision'])
        min_len = min(len(s_scores), len(l_scores))
        if min_len >= 2:
            s_aligned = s_scores[:min_len]
            l_aligned = l_scores[:min_len]
            t_stat, p_val = ttest_rel(l_aligned, s_aligned)
            metrics['statistical_tests']['large_vs_small_cot_ttest'] = {
                't_statistic': round(float(t_stat), 4),
                'p_value': round(float(p_val), 4),
                'delta_precision': round(float(np.mean(l_aligned) - np.mean(s_aligned)) * 100, 2),
                'aligned_sample_count': min_len
            }

    out_file = OUTPUTS_DIR / 'metrics.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print(f'\n[AGGREGATION COMPLETE] Metrics saved to {out_file}')
    return metrics
