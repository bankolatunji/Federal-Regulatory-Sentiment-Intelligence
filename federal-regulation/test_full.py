import requests, json

# Read key from the notebook (already there)
import subprocess, sys

# Use the key from the notebook
KEY = "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"

test_cases = [
    "EPA requires industrial facilities to reduce emissions by 30% with penalties up to $50,000 per day",
    "SBA announces $200 million in grants to support minority-owned businesses",
    "USDA publishes quarterly crop production statistics",
]

for text in test_cases:
    r = requests.post("http://127.0.0.1:5050/api/analyze", json={
        "text": text, "api_key": KEY, "temperature": 0.0
    })
    d = r.json()
    if "error" in d:
        print(f"ERROR: {d['error']}")
    else:
        print(f"\nREGULATION: {text[:70]}...")
        print(f"  SENTIMENT:  {d['sentiment']}")
        print(f"  CONFIDENCE: {d['confidence']}%")
        print(f"  MODEL:      {d['model_info']}")
        print(f"  Pos/Neu/Neg: {d['probabilities']['positive']}% / {d['probabilities']['neutral']}% / {d['probabilities']['restrictive']}%")
        print(f"  PIPELINE:   {' -> '.join(d['processing_steps'])}")
        print(f"  EXPLANATION (first 200 chars): {d['explanation'][:200]}...")
    print("-"*70)
