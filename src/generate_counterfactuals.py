"""
Counterfactual Question Generation Module (SimQG) - initial version
Generates 10 follow-up yes/no questions per explanation.
"""

import json
import re
from pathlib import Path
from tqdm import tqdm
from .config import PROMPTS_JSON_PATH, OUTPUTS_DIR, SIMULATOR_MODEL
from .groq_client import call_groq_cached

def build_batch_simqg_prompt(orig_qn, orig_expl, num_samples=10):
    return (
        "You will be asked to read a starter yes or no question and an AI robot's answer and explanation to the starter question. "
        "After that, you will write follow-up yes or no questions that you can confidently guess the robot's answer to based on its answer to the starter question.\n\n"
        "Important rules:\n"
        "1. Follow the robot's claims and reasoning even if factually incorrect.\n"
        "2. Each question MUST be answerable with YES or NO.\n"
        "3. Each question MUST end with a question mark.\n"
        f"4. Provide exactly {num_samples} distinct, diverse follow-up questions.\n\n"
        f"Starter Question: {orig_qn}\n"
        f"Robot's Answer to the Starter Question: {orig_expl}\n\n"
        f"Write {num_samples} follow-up questions, numbered 1 to {num_samples}:\n"
    )

def extract_counterfactual_questions(response_text, num_samples=10):
    questions = []
    for line in response_text.strip().split('\n'):
        cleaned = re.sub(r'^[0-9]+[\.\)]\s*', '', line.strip())
        cleaned = re.sub(r'^[-*]\s*', '', cleaned)
        if '?' in cleaned:
            q_part = cleaned[:cleaned.find('?') + 1].strip()
            if len(q_part) > 15 and q_part not in questions:
                questions.append(q_part)
    return questions[:num_samples]

def generate_counterfactuals_for_explanations(explanations, generator_model=SIMULATOR_MODEL, num_samples=10, use_cache=True):
    results = []
    first_item = explanations[0]
    taskqa_model = first_item['model']
    expl_type = first_item['expl_type']
    model_tag = taskqa_model.replace('/', '_').replace(':', '_')
    out_file = OUTPUTS_DIR / f'counterfactuals_{model_tag}_{expl_type}.jsonl'

    print(f'Generating counterfactuals for {taskqa_model} ({expl_type})...')
    for item in tqdm(explanations):
        prompt = build_batch_simqg_prompt(item['question'], item['pred_expl'], num_samples=num_samples)
        raw_output = call_groq_cached(prompt=prompt, model=generator_model, temperature=0.7,
                                      top_p=0.95, max_tokens=800, use_cache=use_cache)
        extracted = extract_counterfactual_questions(raw_output, num_samples=num_samples)
        results.append({
            'orig_id': item['id'], 'orig_question': item['question'],
            'orig_answer': item['pred_ans'], 'orig_explanation': item['pred_expl'],
            'model': taskqa_model, 'expl_type': expl_type,
            'counterfactuals': [{'cf_id': i, 'question': q} for i, q in enumerate(extracted)],
            'raw_output': raw_output
        })
    with open(out_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    print(f'Generated counterfactuals -> {out_file}')
    return results
