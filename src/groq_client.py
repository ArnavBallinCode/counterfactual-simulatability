"""
Groq API Inference Client and Persistent Caching Layer
Manages API requests with fast retries and SQLite response caching.
"""

import os
import sys
import time
import sqlite3
import hashlib
import re
from groq import Groq, RateLimitError, APIError
from .config import GROQ_API_KEY, CACHE_DB_PATH, MAX_RETRIES, INITIAL_BACKOFF, REQUEST_DELAY

_client = None

def get_client():
    global _client
    if _client is None:
        if not GROQ_API_KEY or GROQ_API_KEY == 'your_groq_api_key_here':
            raise ValueError(
                'GROQ_API_KEY is not set! Please add your key to replication/.env'
            )
        _client = Groq(api_key=GROQ_API_KEY)
    return _client

def init_cache_db():
    conn = sqlite3.connect(CACHE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS response_cache (
            cache_key TEXT PRIMARY KEY,
            model TEXT,
            prompt TEXT,
            response TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def _make_cache_key(model, prompt, temperature, top_p, max_tokens, stop):
    raw = f'{model}|{temperature}|{top_p}|{max_tokens}|{stop}|{prompt}'
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()

def get_cached_response(cache_key):
    init_cache_db()
    conn = sqlite3.connect(CACHE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT response FROM response_cache WHERE cache_key = ?', (cache_key,))
    row = cursor.fetchone()
    conn.close()
    if row and row[0] and len(row[0].strip()) > 0:
        return row[0]
    return None

def set_cached_response(cache_key, model, prompt, response):
    if not response or len(response.strip()) == 0:
        return
    init_cache_db()
    conn = sqlite3.connect(CACHE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        'INSERT OR REPLACE INTO response_cache (cache_key, model, prompt, response) VALUES (?, ?, ?, ?)',
        (cache_key, model, prompt, response)
    )
    conn.commit()
    conn.close()

def call_groq_cached(prompt, model, temperature=0.0, top_p=1.0, max_tokens=800, stop=None, use_cache=True):
    cache_key = _make_cache_key(model, prompt, temperature, top_p, max_tokens, stop)
    if use_cache:
        cached = get_cached_response(cache_key)
        if cached is not None:
            return cached

    client = get_client()
    stop_sequences = [stop] if isinstance(stop, str) else stop

    for attempt in range(MAX_RETRIES):
        try:
            if REQUEST_DELAY > 0:
                time.sleep(REQUEST_DELAY)
            chat_completion = client.chat.completions.create(
                messages=[{'role': 'user', 'content': prompt}],
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop_sequences
            )
            msg = chat_completion.choices[0].message
            response_text = msg.content or ''
            if not response_text.strip() and hasattr(msg, 'reasoning') and msg.reasoning:
                response_text = msg.reasoning

            if response_text.strip() and use_cache:
                set_cached_response(cache_key, model, prompt, response_text)
            return response_text

        except RateLimitError as e:
            # Short wait of 2-4 seconds instead of hundreds of seconds
            wait_time = 2.0 + (attempt * 0.5)
            print(f'[Groq RateLimit] Quick retry in {wait_time:.1f}s... (Attempt {attempt+1}/{MAX_RETRIES})')
            time.sleep(wait_time)
        except APIError as e:
            wait_time = 2.0
            print(f'[Groq APIError] Attempt {attempt+1}. Retrying in {wait_time:.1f}s. Error: {e}')
            time.sleep(wait_time)

    raise RuntimeError(f'Failed to get response from Groq for model {model} after {MAX_RETRIES} retries.')
