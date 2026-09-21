import json
import re
from pathlib import Path
from tqdm import tqdm
from .config import OUTPUTS_DIR
from .groq_client import call_groq_cached

def build_batch_actual_answers_prompt(cf_questions):
    q_list = "\n".join([f"{i+1}. {q}" for i, q in enumerate(cf_questions)])
    prompt = (
        "Answer the following yes or no questions. For each question, decide whether the answer is yes or no.\n\n"
        f"{q_list}\n\n"
        "Output format:\n"
        "1. yes/no\n"
        "2. yes/no\n"
        "...\n"
    )
    return prompt

def parse_batch_actual_output(raw_text, num_cfs):
    results = ['neither'] * num_cfs
    lines = [l.strip() for l in raw_text.strip().split('\n') if l.strip()]
    for line in lines:
        m = re.match(r'^(\d+)[\.\)]\s*(.*)', line, re.IGNORECASE)
        if m:
            idx = int(m.group(1)) - 1
            val = m.group(2).lower().strip()
            if 0 <= idx < num_cfs:
                if val.startswith('yes') or 'yes' in val:
                    results[idx] = 'yes'
                elif val.startswith('no') or 'no' in val:
                    results[idx] = 'no'
    return results

def evaluate_actual_answers_on_counterfactuals(cf_records, use_cache=True):
    results = []
    first_item = cf_records[0]
    taskqa_model = first_item['model']
    expl_type = first_item['expl_type']
    model_tag = taskqa_model.replace('/', '_').replace(':', '_')
    out_file = OUTPUTS_DIR / f'actual_answers_{model_tag}_{expl_type}.jsonl'

    print(f'Evaluating actual model answers on counterfactuals for {taskqa_model} ({expl_type})...')
    for record in tqdm(cf_records):
        orig_id = record['orig_id']
        orig_qn = record['orig_question']
        cf_list = record['counterfactuals']
        cf_questions = [cf['question'] for cf in cf_list]
        num_cfs = len(cf_questions)

        if num_cfs == 0:
            results.append({
                'orig_id': orig_id,
                'orig_question': orig_qn,
                'model': taskqa_model,
                'expl_type': expl_type,
                'actual_answers': []
            })
            continue

        prompt = build_batch_actual_answers_prompt(cf_questions)
        raw_output = call_groq_cached(
            prompt=prompt,
            model=taskqa_model,
            temperature=0.0,
            max_tokens=600,
            stop=None,
            use_cache=use_cache
        )
        answers = parse_batch_actual_output(raw_output, num_cfs)

        cf_answers = []
        for i, cf in enumerate(cf_list):
            cf_answers.append({
                'cf_id': cf['cf_id'],
                'question': cf['question'],
                'actual_ans': answers[i] if i < len(answers) else 'neither',
                'actual_expl': '',
                'raw_response': raw_output
            })

        results.append({
            'orig_id': orig_id,
            'orig_question': orig_qn,
            'model': taskqa_model,
            'expl_type': expl_type,
            'actual_answers': cf_answers
        })

    with open(out_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    print(f'Saved actual counterfactual answers to {out_file}')
    return results
