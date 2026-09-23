"""
Simulation Generality (Diversity) Evaluation Module — initial version
Computes pairwise BLEU-4 diversity across simulatable counterfactuals.
"""

import json
import numpy as np
from sacrebleu.metrics import BLEU

def calculate_pairwise_bleu(texts):
    if len(texts) <= 1:
        return np.nan
    bleu = BLEU(effective_order=True)
    scores = []
    for i in range(len(texts)):
        for j in range(len(texts)):
            if i != j:
                s = bleu.sentence_score(texts[i], [texts[j]]).score / 100.0
                scores.append(s)
    return float(np.mean(scores)) if scores else np.nan

def compute_simulation_generality(sim_records):
    bleu_gens = []
    for ex in sim_records:
        simulatable_qns = [
            item['question'] for item in ex['simulations']
            if item['sim_prediction'] in ['yes', 'no']
        ]
        if len(simulatable_qns) > 1:
            p_bleu = calculate_pairwise_bleu(simulatable_qns)
            if not np.isnan(p_bleu):
                bleu_gens.append(1.0 - p_bleu)
    return {
        'bleu_generality': round(float(np.mean(bleu_gens)), 3) if bleu_gens else 0.0,
        'num_evaluable_explanations': len(bleu_gens)
    }
