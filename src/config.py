"""
Configuration and Environment Settings Module
Centralizes paths, model names, and environment loading.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_PATH = DATA_DIR / 'strategyqa_raw' / 'strategyqa_test.json'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
PROMPTS_DIR = PROJECT_ROOT / 'prompts'
PROMPTS_JSON_PATH = PROMPTS_DIR / 'prompts.json'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
FIGURES_DIR = OUTPUTS_DIR / 'figures'
CACHE_DB_PATH = OUTPUTS_DIR / 'groq_cache.sqlite'

# Ensure directories exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables
load_dotenv(PROJECT_ROOT / '.env')
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')

# Model configuration
MODEL_LARGE = os.getenv('MODEL_LARGE', 'openai/gpt-oss-120b')
MODEL_SMALL = os.getenv('MODEL_SMALL', 'openai/gpt-oss-20b')
SIMULATOR_MODEL = os.getenv('SIMULATOR_MODEL', 'openai/gpt-oss-120b')
