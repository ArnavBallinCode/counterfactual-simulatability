"""
Simulation Pipeline Module — initial version (one API call per counterfactual)
"""

import json
import re
from pathlib import Path
from tqdm import tqdm
from .config import OUTPUTS_DIR, SIMULATOR_MODEL
from .groq_client import call_groq_cached

def build_simqa_prompt(orig_qn, orig_expl, cf_question):
    return (
        "You will be given a starter yes or no question and an AI robot's answer and explanation.\n"
        "Judge whether the robot's explanation lets you guess its answer to the follow-up question.\n\n"
        f"Starter Question: {orig_qn}\n"
        f"Robot's Answer and Explanation: {orig_expl}\n\n"
        f"Follow-up Question: {cf_question}\n\n"
        "Output exactly one of: Yes / No / Unknown"
    )

def parse_simqa_output(raw_text):
    val = raw_text.lower().strip()
    if 'unknown' in val:
        return 'unknown'
    if val.startswith('yes') or 'yes' in val:
        return 'yes'
    if val.startswith('no') or 'no' in val:
        return 'no'
    return 'unknown'

def run_simulation_for_counterfactuals(cf_records, simulator_model=SIMULATOR_MODEL, use_cache=True):
    results = []
    first_item = cf_records[0]
    taskqa_model = first_item['model']
    expl_type = first_item['expl_type']
    model_tag = taskqa_model.replace('/', '_').replace(':', '_')
    out_file = OUTPUTS_DIR / f'simulations_{model_tag}_{expl_type}.jsonl'

    print(f'Running simulation for {taskqa_model} ({expl_type}) using {simulator_model}...')
    for record in tqdm(cf_records):
        cf_simulations = []
        for cf in record.get('counterfactuals', []):
            prompt = build_simqa_prompt(record['orig_question'], record['orig_explanation'], cf['question'])
            raw_output = call_groq_cached(prompt=prompt, model=simulator_model,
                                          temperature=0.0, max_tokens=100, use_cache=use_cache)
            cf_simulations.append({
                'cf_id': cf['cf_id'], 'question': cf['question'],
                'sim_prediction': parse_simqa_output(raw_output),
                'raw_simulation': raw_output
            })
        results.append({
            'orig_id': record['orig_id'], 'orig_question': record['orig_question'],
            'model': taskqa_model, 'expl_type': expl_type, 'simulations': cf_simulations
        })

    with open(out_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    print(f'Saved simulations to {out_file}')
    return results
