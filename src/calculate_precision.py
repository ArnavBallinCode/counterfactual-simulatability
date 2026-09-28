"""
Simulation Precision Evaluation Module
Computes Macro and Micro Simulation Precision over the simulatable subset C*
following Section 3 of Chen et al. (ICML 2024).
"""

import json
import numpy as np
from pathlib import Path
from .config import OUTPUTS_DIR

def compute_simulation_precision(sim_records, actual_records):
    assert len(sim_records) == len(actual_records)

    per_example_precision = []
    total_simulatable = 0
    total_correct = 0
    total_unsimulatable = 0

    for sim_ex, act_ex in zip(sim_records, actual_records):
        assert sim_ex['orig_id'] == act_ex['orig_id']
        ex_simulatable = 0
        ex_correct = 0

        sim_dict = {item['cf_id']: item['sim_prediction'] for item in sim_ex['simulations']}
        act_dict = {item['cf_id']: item['actual_ans'] for item in act_ex['actual_answers']}

        for cf_id, sim_pred in sim_dict.items():
            if cf_id not in act_dict:
                continue
            act_pred = act_dict[cf_id]

            # In the paper: only count cases where simulator could guess (simulatable subset C*)
            if sim_pred in ['yes', 'no']:
                ex_simulatable += 1
                total_simulatable += 1
                if sim_pred == act_pred:
                    ex_correct += 1
                    total_correct += 1
            else:
                total_unsimulatable += 1

        if ex_simulatable > 0:
            per_example_precision.append(ex_correct / ex_simulatable)
        else:
            per_example_precision.append(np.nan)

    valid_scores = [s for s in per_example_precision if not np.isnan(s)]
    mean_precision = float(np.mean(valid_scores) * 100) if valid_scores else 0.0
    overall_micro_precision = float((total_correct / total_simulatable) * 100) if total_simulatable > 0 else 0.0

    return {
        'macro_precision': round(mean_precision, 2),
        'micro_precision': round(overall_micro_precision, 2),
        'total_simulatable': total_simulatable,
        'total_correct': total_correct,
        'total_unsimulatable': total_unsimulatable,
        'per_example_precision': per_example_precision
    }
