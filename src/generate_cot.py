"""
Explanation Generation Module (CoT and Post-Hoc)
Generates model answers and step-by-step or post-hoc rationales
using StrategyQA few-shot exemplars from Chen et al. (EMNLP 2023).
"""

import json
import re
from pathlib import Path
from tqdm import tqdm
from .config import PROMPTS_JSON_PATH, OUTPUTS_DIR
from .groq_client import call_groq_cached

def build_taskqa_prompt(expl_type, question):
    with open(PROMPTS_JSON_PATH, 'r', encoding='utf-8') as f:
        prompts_data = json.load(f)

    task_key = f'strategyqa-taskqa-{expl_type}'
    p_info = prompts_data[task_key]

    prompt = p_info['instruction'] + '\n\n'
    for dem in p_info['dem_examples']:
        prompt += p_info['template_with_label'].format(**dem)
    prompt += p_info['template_no_label'].format(question=question)
    return prompt

def parse_taskqa_output(raw_text, expl_type):
    raw = raw_text.strip()
    if expl_type == 'cot':
        lower_raw = raw.lower()
        if 'so the answer is yes' in lower_raw or lower_raw.endswith('yes.'):
            ans = 'yes'
        elif 'so the answer is no' in lower_raw or lower_raw.endswith('no.'):
            ans = 'no'
        elif re.search(r'\banswer is yes\b', lower_raw):
            ans = 'yes'
        elif re.search(r'\banswer is no\b', lower_raw):
            ans = 'no'
        else:
            ans = 'neither'
        return {'pred_ans': ans, 'pred_expl': raw, 'raw_response': raw}

    elif expl_type == 'posthoc':
        lines = [line.strip() for line in raw.split('\n') if line.strip()]
        first_line = lines[0].lower() if lines else ''
        first_line = re.sub(r'^(a|answer|choice):\s*', '', first_line).strip()

        if first_line.startswith('yes') or first_line == 'yes':
            ans = 'yes'
        elif first_line.startswith('no') or first_line == 'no':
            ans = 'no'
        elif 'justification:' in raw.lower():
            lower_raw = raw.lower()
            if 'so the answer is yes' in lower_raw or 'answer: yes' in lower_raw:
                ans = 'yes'
            elif 'so the answer is no' in lower_raw or 'answer: no' in lower_raw:
                ans = 'no'
            else:
                ans = 'neither'
        else:
            ans = 'neither'

        expl = '\n'.join(lines[1:]) if len(lines) > 1 else raw
        return {'pred_ans': ans, 'pred_expl': expl, 'raw_response': raw}

def generate_explanations(examples, model, expl_type, use_cache=True):
    assert expl_type in ['cot', 'posthoc']
    results = []
    model_tag = model.replace('/', '_').replace(':', '_')
    out_file = OUTPUTS_DIR / f'taskqa_{model_tag}_{expl_type}.jsonl'

    print(f'Generating {expl_type.upper()} explanations with model: {model} ({len(examples)} examples)...')
    for ex in tqdm(examples):
        prompt = build_taskqa_prompt(expl_type, ex['question'])
        raw_output = call_groq_cached(
            prompt=prompt,
            model=model,
            temperature=0.0,
            max_tokens=800,
            stop=['\n\nQ:', '\n\nHuman:'] if expl_type == 'cot' else '\n\n',
            use_cache=use_cache
        )
        parsed = parse_taskqa_output(raw_output, expl_type)
        record = {
            'id': ex['id'],
            'question': ex['question'],
            'model': model,
            'expl_type': expl_type,
            'pred_ans': parsed['pred_ans'],
            'pred_expl': parsed['pred_expl'],
            'raw_response': parsed['raw_response']
        }
        results.append(record)

    with open(out_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    print(f'Saved {len(results)} {expl_type} explanations to {out_file}')
    return results
