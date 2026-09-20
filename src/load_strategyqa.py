"""
StrategyQA Dataset Ingestion and Preprocessing Module
Loads and standardizes questions from the official StrategyQA benchmark.
"""

import json
from pathlib import Path
from .config import RAW_DATA_PATH, PROCESSED_DATA_DIR, SEED
import random

def load_strategyqa_dataset(limit=None, seed=SEED):
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f'Raw dataset not found at {RAW_DATA_PATH}')

    with open(RAW_DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Standardize format
    examples = []
    for idx, item in enumerate(data):
        examples.append({
            'id': idx,
            'question': item['question'].strip()
        })

    if limit is not None and limit < len(examples):
        random.seed(seed)
        examples = examples[:limit]

    out_file = PROCESSED_DATA_DIR / f'eval_set_{len(examples)}.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=2)

    print(f'Loaded {len(examples)} StrategyQA evaluation questions.')
    return examples

if __name__ == '__main__':
    data = load_strategyqa_dataset()
    print('Sample question 0:', data[0])
