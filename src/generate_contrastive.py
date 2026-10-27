"""
Phase 2 Extension: Contrastive Decision-Boundary Prompting (CDBP) Explanation Generator
Generates explanations with explicit [Decision Rule] and [Boundary Condition] annotations
to address the unsimulatable filter collapse identified in Phase 1.

Key Difference from generate_cot.py (Phase 1 Baseline):
- Standard CoT/Post-Hoc prompts produce unconstrained factual narratives.
- CDBP prompts enforce a two-part structured contract requiring the model to
  articulate both the operational decision rule and the contrastive boundary
  condition under which its answer would flip.
"""

import json
import re
from pathlib import Path
from tqdm import tqdm
from .config import PROMPTS_DIR, OUTPUTS_DIR
from .groq_client import call_groq_cached


def load_contrastive_prompt(expl_type):
    """Load the contrastive few-shot prompt template for the given explanation type."""
    if expl_type == 'cot':
        prompt_file = PROMPTS_DIR / 'contrastive_cot_prompt.txt'
    elif expl_type == 'posthoc':
        prompt_file = PROMPTS_DIR / 'contrastive_posthoc_prompt.txt'
    else:
        raise ValueError(f'Unknown explanation type: {expl_type}')

    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read().strip()


def build_contrastive_prompt(expl_type, question, few_shot_text):
    """
    Build the full contrastive prompt by appending the test question
    to the few-shot exemplar block.
    """
    if expl_type == 'cot':
        prompt = few_shot_text + f'\n\nQ: Yes or no: {question}\nA:'
    elif expl_type == 'posthoc':
        prompt = few_shot_text + f'\n\nQ: Yes or no: {question}\nA:'
    return prompt


def parse_contrastive_output(raw_text, expl_type):
    """
    Parse the model's contrastive explanation output.
    Extracts: predicted answer (yes/no), full explanation text,
    decision rule, and boundary condition.
    """
    raw = raw_text.strip()
    lower_raw = raw.lower()

    # Extract predicted answer
    if expl_type == 'cot':
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
    elif expl_type == 'posthoc':
        lines = [line.strip() for line in raw.split('\n') if line.strip()]
        first_line = lines[0].lower() if lines else ''
        first_line = re.sub(r'^(a|answer|choice):\s*', '', first_line).strip()
        if first_line.startswith('yes') or first_line == 'yes':
            ans = 'yes'
        elif first_line.startswith('no') or first_line == 'no':
            ans = 'no'
        else:
            ans = 'neither'

    # Extract [Decision Rule] and [Boundary Condition] if present
    decision_rule = ''
    boundary_condition = ''

    dr_match = re.search(
        r'\[Decision Rule\][:\s]*(.+?)(?=\[Boundary Condition\]|So the answer is|$)',
        raw, re.IGNORECASE | re.DOTALL
    )
    if dr_match:
        decision_rule = dr_match.group(1).strip()

    bc_match = re.search(
        r'\[Boundary Condition\][:\s]*(.+?)(?=So the answer is|$)',
        raw, re.IGNORECASE | re.DOTALL
    )
    if bc_match:
        boundary_condition = bc_match.group(1).strip()

    return {
        'pred_ans': ans,
        'pred_expl': raw,
        'decision_rule': decision_rule,
        'boundary_condition': boundary_condition,
        'has_boundary': bool(decision_rule and boundary_condition),
        'raw_response': raw
    }


def generate_contrastive_explanations(examples, model, expl_type, use_cache=True):
    """
    Generate CDBP (Contrastive Decision-Boundary Prompting) explanations
    for all examples using the specified model and explanation type.

    This is the Phase 2 extension counterpart to generate_explanations() in generate_cot.py.
    The only architectural difference is the prompt template: CDBP few-shot exemplars
    include explicit [Decision Rule] and [Boundary Condition] annotations that teach
    the model to articulate contrastive decision boundaries via in-context learning.
    """
    assert expl_type in ['cot', 'posthoc']
    results = []
    model_tag = model.replace('/', '_').replace(':', '_')
    out_file = OUTPUTS_DIR / f'taskqa_{model_tag}_contrastive_{expl_type}.jsonl'

    # Load the contrastive few-shot prompt template
    few_shot_text = load_contrastive_prompt(expl_type)

    print(f'[CDBP] Generating contrastive {expl_type.upper()} explanations with model: {model} ({len(examples)} examples)...')

    boundary_count = 0
    for ex in tqdm(examples):
        prompt = build_contrastive_prompt(expl_type, ex['question'], few_shot_text)
        raw_output = call_groq_cached(
            prompt=prompt,
            model=model,
            temperature=0.0,
            max_tokens=800,
            stop=['\n\nQ:', '\n\nHuman:'] if expl_type == 'cot' else '\n\n',
            use_cache=use_cache
        )
        parsed = parse_contrastive_output(raw_output, expl_type)

        if parsed['has_boundary']:
            boundary_count += 1

        record = {
            'id': ex['id'],
            'question': ex['question'],
            'model': model,
            'expl_type': f'contrastive_{expl_type}',
            'pred_ans': parsed['pred_ans'],
            'pred_expl': parsed['pred_expl'],
            'decision_rule': parsed['decision_rule'],
            'boundary_condition': parsed['boundary_condition'],
            'has_boundary': parsed['has_boundary'],
            'raw_response': parsed['raw_response']
        }
        results.append(record)

    with open(out_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    boundary_rate = (boundary_count / len(results) * 100) if results else 0
    print(f'[CDBP] Saved {len(results)} contrastive {expl_type} explanations to {out_file}')
    print(f'[CDBP] Boundary annotation rate: {boundary_count}/{len(results)} ({boundary_rate:.1f}%)')
    return results

