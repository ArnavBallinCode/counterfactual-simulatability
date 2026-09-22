"""
Groq API Inference Client
Manages API requests with exponential backoff and rate-limit handling.
"""

import os
import sys
import time
from groq import Groq, RateLimitError, APIError
from .config import GROQ_API_KEY, MAX_RETRIES, INITIAL_BACKOFF, REQUEST_DELAY

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

def call_groq(prompt, model, temperature=0.0, top_p=1.0, max_tokens=800, stop=None):
    client = get_client()
    backoff = INITIAL_BACKOFF
    stop_sequences = [stop] if isinstance(stop, str) else stop
    for attempt in range(MAX_RETRIES + 4):
        try:
            time.sleep(REQUEST_DELAY)
            chat_completion = client.chat.completions.create(
                messages=[{'role': 'user', 'content': prompt}],
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop_sequences
            )
            return chat_completion.choices[0].message.content or ''
        except RateLimitError as e:
            print(f'[Groq RateLimit] Attempt {attempt+1}. Sleeping {backoff:.1f}s...')
            time.sleep(backoff)
            backoff *= 1.8
        except APIError as e:
            print(f'[Groq APIError] Attempt {attempt+1}. Error: {e}')
            time.sleep(backoff)
            backoff *= 1.5
    raise RuntimeError(f'Failed to get response from Groq for model {model} after retries.')
