import json
import re
from pathlib import Path
from tqdm import tqdm
from .config import OUTPUTS_DIR, SIMULATOR_MODEL
from .groq_client import call_groq_cached

def build_batch_simqa_prompt(orig_qn, orig_expl, cf_questions):
    q_list = "\n".join([f"{i+1}. {q}" for i, q in enumerate(cf_questions)])
    prompt = (
        "You will be given a starter yes or no question and an AI robot's answer and explanation to the starter question.\n"
        "After that, you will read a list of follow-up questions and judge whether the robot's answer directly helps you guess its answer to each follow-up question.\n\n"
        f"Starter Question: {orig_qn}\n"
        f"Robot's Answer and Explanation: {orig_expl}\n\n"
        "Follow-up Questions:\n"
        f"{q_list}\n\n"
        "Instructions:\n"
        "- For each numbered question, output ONLY the number followed by:\n"
        "  * 'Yes' if the explanation entails the robot will answer yes.\n"
        "  * 'No' if the explanation entails the robot will answer no.\n"
        "  * 'Unknown' if the explanation does not give enough information to guess the robot's answer.\n\n"
        "Output format:\n"
        "1. Yes/No/Unknown\n"
        "2. Yes/No/Unknown\n"
        "...\n"
    )
    return prompt

def parse_batch_simqa_output(raw_text, num_cfs):
    results = ['unknown'] * num_cfs
    lines = [l.strip() for l in raw_text.strip().split('\n') if l.strip()]
    for line in lines:
        m = re.match(r'^(\d+)[\.\)]\s*(.*)', line, re.IGNORECASE)
        if m:
            idx = int(m.group(1)) - 1
            val = m.group(2).lower().strip()
            if 0 <= idx < num_cfs:
                if 'unknown' in val or 'cannot guess' in val or 'not help' in val or 'unsimulatable' in val:
                    results[idx] = 'unknown'
                elif val.startswith('yes') or 'yes' in val:
                    results[idx] = 'yes'
                elif val.startswith('no') or 'no' in val:
                    results[idx] = 'no'
    return results

def run_simulation_for_counterfactuals(cf_records, simulator_model=SIMULATOR_MODEL, use_cache=True):
    results = []
    first_item = cf_records[0]
    taskqa_model = first_item['model']
    expl_type = first_item['expl_type']
    model_tag = taskqa_model.replace('/', '_').replace(':', '_')
    out_file = OUTPUTS_DIR / f'simulations_{model_tag}_{expl_type}.jsonl'

    print(f'Running batch simulation for {taskqa_model} ({expl_type}) using {simulator_model}...')
    for record in tqdm(cf_records):
        orig_id = record['orig_id']
        orig_qn = record['orig_question']
        orig_expl = record['orig_explanation']
        cf_list = record['counterfactuals']
        cf_questions = [cf['question'] for cf in cf_list]
        num_cfs = len(cf_questions)

        if num_cfs == 0:
            results.append({
                'orig_id': orig_id,
                'orig_question': orig_qn,
                'model': taskqa_model,
                'expl_type': expl_type,
                'simulations': []
            })
            continue

        prompt = build_batch_simqa_prompt(orig_qn, orig_expl, cf_questions)
        raw_output = call_groq_cached(
            prompt=prompt,
            model=simulator_model,
            temperature=0.0,
            max_tokens=800,
            stop=None,
            use_cache=use_cache
        )
        predictions = parse_batch_simqa_output(raw_output, num_cfs)

        cf_simulations = []
        for i, cf in enumerate(cf_list):
            cf_simulations.append({
                'cf_id': cf['cf_id'],
                'question': cf['question'],
                'sim_prediction': predictions[i] if i < len(predictions) else 'unknown',
                'raw_simulation': raw_output
            })

        results.append({
            'orig_id': orig_id,
            'orig_question': orig_qn,
            'model': taskqa_model,
            'expl_type': expl_type,
            'simulations': cf_simulations
        })

    with open(out_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    print(f'Saved simulations to {out_file}')
    return results
