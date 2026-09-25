import os
import sys
from pathlib import Path

# Add src and replication root to sys.path
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR.parent))

from src.config import GROQ_API_KEY, MODEL_LARGE, MODEL_SMALL
from src.groq_client import get_client, call_groq_cached

def test_connection():
    print("=== Testing Groq API Setup ===")
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        print("[ACTION REQUIRED] GROQ_API_KEY is not set in replication/.env!")
        print("Please open replication/.env and paste your Groq API key: GROQ_API_KEY=gsk_...")
        return False

    client = get_client()
    try:
        models = client.models.list()
        print("[SUCCESS] Successfully connected to Groq API!")
        print(f"Found {len(models.data)} available models on your account:")
        for m in sorted([m.id for m in models.data]):
            print(f"  - {m}")
    except Exception as e:
        print(f"[ERROR] Failed to list models: {e}")
        return False

    print(f"\nTesting inference on small model: {MODEL_SMALL}")
    try:
        ans_small = call_groq_cached("Answer in 1 word: Is water wet?", model=MODEL_SMALL, max_tokens=10, use_cache=False)
        print(f"  Small model response: {ans_small.strip()}")
    except Exception as e:
        print(f"  [ERROR] Small model inference failed: {e}")
        return False

    print(f"\nTesting inference on large model: {MODEL_LARGE}")
    try:
        ans_large = call_groq_cached("Answer in 1 word: Is water wet?", model=MODEL_LARGE, max_tokens=10, use_cache=False)
        print(f"  Large model response: {ans_large.strip()}")
    except Exception as e:
        print(f"  [ERROR] Large model inference failed: {e}")
        return False

    print("\n[READY] Groq API connectivity and test models verified successfully!")
    return True

if __name__ == "__main__":
    test_connection()
