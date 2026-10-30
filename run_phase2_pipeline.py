"""
Phase 2 Extension Pipeline Runner: Contrastive Decision-Boundary Prompting (CDBP)
Runs the CDBP extension across all model/explanation conditions on StrategyQA,
then computes Before vs. After comparison metrics against Phase 1 baseline.

Usage:
    python run_phase2_pipeline.py --limit 30 --num_cfs 10
"""

import argparse
import json
import sys
import time
import numpy as np
from pathlib import Path
from scipy.stats import ttest_rel

# Add replication dir to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_SMALL, MODEL_LARGE, SIMULATOR_MODEL, OUTPUTS_DIR, FIGURES_DIR
from src.load_strategyqa import load_strategyqa_dataset
from src.generate_contrastive import generate_contrastive_explanations
from src.generate_counterfactuals import generate_counterfactuals_for_explanations
from src.run_simulation import run_simulation_for_counterfactuals
from src.get_actual_answers import evaluate_actual_answers_on_counterfactuals
from src.calculate_precision import compute_simulation_precision
from src.calculate_generality import compute_simulation_generality
from src.plot_phase2_results import plot_phase2_comparisons


def load_phase1_metrics():
    """Load Phase 1 baseline metrics from outputs/metrics.json."""
    metrics_file = OUTPUTS_DIR / 'metrics.json'
    if metrics_file.exists():
        with open(metrics_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def compute_simulatable_yield(sim_records):
    """
    Compute the simulatable yield: fraction of counterfactuals where the
    simulator produced a definitive yes/no answer (not bot/unknown).
    This is the key diagnostic metric for the bot filter collapse.
    """
    total_cfs = 0
    total_simulatable = 0
    total_bot = 0

    for ex in sim_records:
        for item in ex['simulations']:
            total_cfs += 1
            if item['sim_prediction'] in ['yes', 'no']:
                total_simulatable += 1
            else:
                total_bot += 1

    yield_pct = (total_simulatable / total_cfs * 100) if total_cfs > 0 else 0
    return {
        'total_counterfactuals': total_cfs,
        'simulatable': total_simulatable,
        'unsimulatable_bot': total_bot,
        'simulatable_yield_pct': round(yield_pct, 2)
    }


def generate_phase2_comparison(phase1_metrics, phase2_precisions, phase2_generalities,
                                phase2_yields, phase2_taskqa):
    """
    Generate the Before vs. After comparison table and save to metrics_phase2.json.
    """
    comparison = {
        'description': 'Phase 2 Extension: Contrastive Decision-Boundary Prompting (CDBP) vs Phase 1 Baseline',
        'conditions': {}
    }

    # Phase 1 baseline precision data
    p1_prec = phase1_metrics.get('table4_simulation_precision', {}) if phase1_metrics else {}

    for key in phase2_precisions:
        # Map contrastive key back to baseline key
        # e.g., "openai/gpt-oss-20b_contrastive_cot" -> "openai/gpt-oss-20b_cot"
        baseline_key = key.replace('_contrastive_cot', '_cot').replace('_contrastive_posthoc', '_posthoc')

        p1_data = p1_prec.get(baseline_key, {})
        p2_data = phase2_precisions[key]
        p2_yield = phase2_yields.get(key, {})
        p2_gen = phase2_generalities.get(key, {})

        # Compute Phase 1 simulatable yield from stored counts
        p1_sim = p1_data.get('simulatable_count', 0)
        p1_unsim = p1_data.get('unsimulatable_count', 0)
        p1_total = p1_sim + p1_unsim
        p1_yield = round(p1_sim / p1_total * 100, 2) if p1_total > 0 else 0

        condition_report = {
            'baseline_key': baseline_key,
            'extension_key': key,
            'precision': {
                'phase1_macro': p1_data.get('macro_precision', 'N/A'),
                'phase2_macro': p2_data['macro_precision'],
                'delta_pp': round(p2_data['macro_precision'] - p1_data.get('macro_precision', 0), 2) if isinstance(p1_data.get('macro_precision'), (int, float)) else 'N/A'
            },
            'simulatable_yield': {
                'phase1_yield_pct': p1_yield,
                'phase2_yield_pct': p2_yield.get('simulatable_yield_pct', 0),
                'phase1_simulatable': p1_sim,
                'phase1_unsimulatable': p1_unsim,
                'phase2_simulatable': p2_yield.get('simulatable', 0),
                'phase2_unsimulatable': p2_yield.get('unsimulatable_bot', 0),
                'delta_yield_pp': round(p2_yield.get('simulatable_yield_pct', 0) - p1_yield, 2)
            },
            'generality': {
                'phase2_bleu': p2_gen.get('bleu_generality', 0),
                'phase2_cosine': p2_gen.get('cosine_generality', 0),
                'phase2_jaccard': p2_gen.get('jaccard_generality', 0)
            }
        }
        comparison['conditions'][key] = condition_report

    # Statistical significance: paired t-test on precision (Phase 1 vs Phase 2)
    comparison['statistical_tests'] = {}

    for model_name, model_label in [(MODEL_SMALL, 'small'), (MODEL_LARGE, 'large')]:
        for et in ['cot', 'posthoc']:
            p1_key = f'{model_name}_{et}'
            p2_key = f'{model_name}_contrastive_{et}'

            if p2_key in phase2_precisions:
                p2_scores = phase2_precisions[p2_key].get('per_example_precision', [])
                p2_scores = np.array([s if not np.isnan(s) else 0.0 for s in p2_scores])

                model_tag = model_name.replace('/', '_').replace(':', '_')
                p1_sim_file = OUTPUTS_DIR / f'simulations_{model_tag}_{et}.jsonl'
                p1_act_file = OUTPUTS_DIR / f'actual_answers_{model_tag}_{et}.jsonl'

                if p1_sim_file.exists() and p1_act_file.exists():
                    with open(p1_sim_file, 'r', encoding='utf-8') as f:
                        p1_sims = [json.loads(l) for l in f]
                    with open(p1_act_file, 'r', encoding='utf-8') as f:
                        p1_acts = [json.loads(l) for l in f]
                    p1_res = compute_simulation_precision(p1_sims, p1_acts)
                    p1_scores = np.array([s if not np.isnan(s) else 0.0 for s in p1_res['per_example_precision']])

                    min_len = min(len(p1_scores), len(p2_scores))
                    if min_len >= 2:
                        t_stat, p_val = ttest_rel(p2_scores[:min_len], p1_scores[:min_len])
                        comparison['statistical_tests'][f'{model_label}_{et}_paired_ttest'] = {
                            't_statistic': round(float(t_stat), 4),
                            'p_value': round(float(p_val), 4),
                            'p1_mean': round(float(np.mean(p1_scores[:min_len])) * 100, 2),
                            'p2_mean': round(float(np.mean(p2_scores[:min_len])) * 100, 2),
                            'delta_pp': round(float(np.mean(p2_scores[:min_len]) - np.mean(p1_scores[:min_len])) * 100, 2),
                            'aligned_count': min_len
                        }
                else:
                    comparison['statistical_tests'][f'{model_label}_{et}_summary'] = {
                        'phase2_macro_precision': phase2_precisions[p2_key]['macro_precision'],
                        'phase1_macro_precision': p1_prec.get(p1_key, {}).get('macro_precision', 'N/A'),
                        'sample_count': len(p2_scores)
                    }

    # Save comparison metrics
    out_file = OUTPUTS_DIR / 'metrics_phase2.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2)

    print(f'\n[PHASE 2 COMPARISON] Saved to {out_file}')
    return comparison


def print_comparison_table(comparison):
    """Print a human-readable Before vs. After comparison table."""
    print('\n' + '=' * 90)
    print('PHASE 2 RESULTS: Contrastive Decision-Boundary Prompting (CDBP) vs Phase 1 Baseline')
    print('=' * 90)

    for key, cond in comparison.get('conditions', {}).items():
        prec = cond['precision']
        yld = cond['simulatable_yield']
        gen = cond['generality']

        print(f'\n--- {key} ---')
        print(f'  Precision:        Phase 1 = {prec["phase1_macro"]}%  |  Phase 2 (CDBP) = {prec["phase2_macro"]}%  |  Delta = {prec["delta_pp"]} pp')
        print(f'  Simulatable Yield: Phase 1 = {yld["phase1_yield_pct"]}%  |  Phase 2 (CDBP) = {yld["phase2_yield_pct"]}%  |  Delta = {yld["delta_yield_pp"]} pp')
        print(f'  Phase 2 Generality: BLEU={gen["phase2_bleu"]}, Cosine={gen["phase2_cosine"]}, Jaccard={gen["phase2_jaccard"]}')

    print('\n' + '=' * 90)


def main():
    parser = argparse.ArgumentParser(description='Phase 2 Extension Pipeline: CDBP on StrategyQA')
    parser.add_argument('--limit', type=int, default=30, help='Number of StrategyQA questions (default: 30)')
    parser.add_argument('--num_cfs', type=int, default=10, help='Counterfactuals per question (default: 10)')
    parser.add_argument('--no_cache', action='store_true', help='Disable Groq response caching')
    args = parser.parse_args()

    use_cache = not args.no_cache
    start_time = time.time()
    print(f'=== Starting Phase 2 Extension Pipeline: CDBP (Limit: {args.limit}, CFs: {args.num_cfs}) ===')

    # Step 1: Load StrategyQA questions (same dataset as Phase 1)
    dataset = load_strategyqa_dataset(limit=args.limit)

    # Step 2: Load Phase 1 baseline metrics for comparison
    phase1_metrics = load_phase1_metrics()
    if phase1_metrics:
        print('[Phase 1 baseline metrics loaded for comparison]')
    else:
        print('[WARNING] Phase 1 metrics.json not found. Comparison will be partial.')

    models_to_eval = [MODEL_SMALL, MODEL_LARGE]
    expl_types = ['cot', 'posthoc']

    all_phase2_precisions = {}
    all_phase2_generalities = {}
    all_phase2_yields = {}
    all_phase2_taskqa = {}

    for model in models_to_eval:
        for expl_type in expl_types:
            key = f'{model}_contrastive_{expl_type}'
            print(f'\n========================================')
            print(f'[CDBP] Condition: Model={model} | Expl=CONTRASTIVE_{expl_type.upper()}')
            print(f'========================================')

            # Step 3: Generate CDBP explanations (the core extension)
            expl_records = generate_contrastive_explanations(
                dataset, model=model, expl_type=expl_type, use_cache=use_cache
            )
            all_phase2_taskqa[key] = expl_records

            # Step 4: Generate counterfactual questions (SimQG) - same module as Phase 1
            cf_records = generate_counterfactuals_for_explanations(
                expl_records,
                generator_model=SIMULATOR_MODEL,
                num_samples=args.num_cfs,
                use_cache=use_cache
            )

            # Step 5: Simulate (SimQA) - same module as Phase 1
            sim_records = run_simulation_for_counterfactuals(
                cf_records,
                simulator_model=SIMULATOR_MODEL,
                use_cache=use_cache
            )

            # Step 6: Get actual model answers on counterfactuals - same module as Phase 1
            act_records = evaluate_actual_answers_on_counterfactuals(
                cf_records,
                use_cache=use_cache
            )

            # Step 7: Compute metrics
            precision_res = compute_simulation_precision(sim_records, act_records)
            generality_res = compute_simulation_generality(sim_records)
            yield_res = compute_simulatable_yield(sim_records)

            all_phase2_precisions[key] = precision_res
            all_phase2_generalities[key] = generality_res
            all_phase2_yields[key] = yield_res

            print(f'[{key}] Precision: {precision_res["macro_precision"]}%')
            print(f'[{key}] Simulatable Yield: {yield_res["simulatable_yield_pct"]}% ({yield_res["simulatable"]}/{yield_res["total_counterfactuals"]})')
            print(f'[{key}] Generality (Cosine): {generality_res["cosine_generality"]}')

    # Step 8: Generate Before vs. After comparison
    comparison = generate_phase2_comparison(
        phase1_metrics, all_phase2_precisions,
        all_phase2_generalities, all_phase2_yields,
        all_phase2_taskqa
    )
    print_comparison_table(comparison)

    # Step 9: Plot publication-quality Before vs. After figures
    plot_phase2_comparisons(comparison)

    elapsed = time.time() - start_time
    print(f'\n[SUCCESS] Phase 2 Extension Pipeline finished in {elapsed/60:.2f} minutes!')


if __name__ == '__main__':
    main()

