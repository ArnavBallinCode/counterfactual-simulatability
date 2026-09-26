"""
Counterfactual Question Generation Module (SimQG)
Synthesizes 10 follow-up counterfactual yes/no questions per explanation
whose answers can be inferred directly from the model's rationale.
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
    lines = response_text.strip().split('\n')

    # Line by line inspection
    for line in lines:
        cleaned = line.strip()
        cleaned = re.sub(r'^[0-9]+[\.\)]\s*', '', cleaned)
        cleaned = re.sub(r'^[-*]\s*', '', cleaned)
        if '?' in cleaned:
            q_part = cleaned[:cleaned.find('?') + 1].strip()
            if len(q_part) > 15 and q_part not in questions:
                questions.append(q_part)

    # Regex fallback if fewer than num_samples
    if len(questions) < num_samples:
        q_regex = r'\b((?:Would|Could|Can|Is|Are|Did|Do|Does|Will|Was|Were|Has|Have|Had|If)\b[^?\n]+\?)'
        for match in re.finditer(q_regex, response_text, re.IGNORECASE):
            q_text = match.group(1).strip()
            if len(q_text) > 15 and q_text not in questions:
                questions.append(q_text)

    return questions[:num_samples]

def generate_counterfactuals_for_explanations(explanations, generator_model=SIMULATOR_MODEL, num_samples=10, use_cache=True):
    results = []
    first_item = explanations[0]
    taskqa_model = first_item['model']
    expl_type = first_item['expl_type']
    model_tag = taskqa_model.replace('/', '_').replace(':', '_')
    out_file = OUTPUTS_DIR / f'counterfactuals_{model_tag}_{expl_type}.jsonl'

    print(f'Generating counterfactuals for {taskqa_model} ({expl_type}) using {generator_model}...')
    for item in tqdm(explanations):
        orig_id = item['id']
        orig_qn = item['question']
        orig_expl = item['pred_expl']

        prompt = build_batch_simqg_prompt(orig_qn, orig_expl, num_samples=num_samples)
        raw_output = call_groq_cached(
            prompt=prompt,
            model=generator_model,
            temperature=0.7,
            top_p=0.95,
            max_tokens=800,
            use_cache=use_cache
        )
        extracted = extract_counterfactual_questions(raw_output, num_samples=num_samples)

        record = {
            'orig_id': orig_id,
            'orig_question': orig_qn,
            'orig_answer': item['pred_ans'],
            'orig_explanation': orig_expl,
            'model': taskqa_model,
            'expl_type': expl_type,
            'counterfactuals': [{'cf_id': i, 'question': q} for i, q in enumerate(extracted)],
            'raw_output': raw_output
        }
        results.append(record)

    with open(out_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    total_cfs = sum(len(r['counterfactuals']) for r in results)
    print(f'Generated {total_cfs} counterfactual questions across {len(results)} explanations -> {out_file}')
    return results
