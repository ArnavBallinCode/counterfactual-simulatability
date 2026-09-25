"""
Master Replication Pipeline Runner
Orchestrates end-to-end evaluation across model scales (20B, 120B)
and explanation paradigms (CoT, Post-Hoc) on StrategyQA.
"""

import argparse
import sys
import time
from pathlib import Path

# Add replication dir to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_SMALL, MODEL_LARGE, SIMULATOR_MODEL, OUTPUTS_DIR
from src.load_strategyqa import load_strategyqa_dataset
from src.generate_cot import generate_explanations
from src.generate_counterfactuals import generate_counterfactuals_for_explanations
from src.run_simulation import run_simulation_for_counterfactuals
from src.get_actual_answers import evaluate_actual_answers_on_counterfactuals
from src.calculate_precision import compute_simulation_precision
from src.calculate_generality import compute_simulation_generality
from src.aggregate_results import aggregate_and_save_all_metrics
from src.plot_results import generate_all_plots

def main():
    parser = argparse.ArgumentParser(description='Phase 1 Paper Replication Pipeline (StrategyQA)')
    parser.add_argument('--limit', type=int, default=30, help='Number of StrategyQA questions to evaluate (default: 30)')
    parser.add_argument('--num_cfs', type=int, default=10, help='Number of counterfactuals per question (default: 10)')
    parser.add_argument('--no_cache', action='store_true', help='Disable Groq response caching')
    args = parser.parse_args()

    use_cache = not args.no_cache
    start_time = time.time()
    print(f'=== Starting Phase 1 Replication Pipeline (Limit: {args.limit} questions, CFs/question: {args.num_cfs}) ===')

    # Step 1: Load StrategyQA questions
    dataset = load_strategyqa_dataset(limit=args.limit)

    models_to_eval = [MODEL_SMALL, MODEL_LARGE]
    expl_types = ['cot', 'posthoc']

    all_taskqa = {}
    all_precisions = {}
    all_generalities = {}

    for model in models_to_eval:
        for expl_type in expl_types:
            key = f'{model}_{expl_type}'
            print(f'\n========================================')
            print(f'Running Condition: Model={model} | Expl={expl_type.upper()}')
            print(f'========================================')

            # Step 2: Explanation generation (CoT / Post-Hoc)
            expl_records = generate_explanations(dataset, model=model, expl_type=expl_type, use_cache=use_cache)
            all_taskqa[key] = expl_records

            # Step 3: Counterfactual Question Generation (SimQG)
            cf_records = generate_counterfactuals_for_explanations(
                expl_records,
                generator_model=SIMULATOR_MODEL,
                num_samples=args.num_cfs,
                use_cache=use_cache
            )

            # Step 4: Simulation (SimQA)
            sim_records = run_simulation_for_counterfactuals(
                cf_records,
                simulator_model=SIMULATOR_MODEL,
                use_cache=use_cache
            )

            # Step 5: Actual counterfactual evaluation on evaluated model
            act_records = evaluate_actual_answers_on_counterfactuals(
                cf_records,
                use_cache=use_cache
            )

            # Step 6: Compute Precision & Generality for this condition
            precision_res = compute_simulation_precision(sim_records, act_records)
            generality_res = compute_simulation_generality(sim_records)

            all_precisions[key] = precision_res
            all_generalities[key] = generality_res

            print(f'[{key}] Precision: {precision_res["macro_precision"]}%, Cosine Generality: {generality_res["cosine_generality"]}')

    # Step 7: Aggregate all metrics & compute statistics
    final_metrics = aggregate_and_save_all_metrics(all_precisions, all_generalities, all_taskqa)

    # Step 8: Generate Figures
    generate_all_plots(final_metrics)

    elapsed = time.time() - start_time
    print(f'\n[SUCCESS] Entire Phase 1 Replication finished in {elapsed/60:.2f} minutes!')

if __name__ == '__main__':
    main()
