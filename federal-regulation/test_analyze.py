import requests, json

# Test without Claude (no key) - should still get ML classification
payload = {
    "text": "EPA requires industrial facilities to reduce emissions by 30% with penalties up to $50,000 per day",
    "api_key": "",   # deliberately empty
    "temperature": 0.0,
}

r = requests.post("http://127.0.0.1:5050/api/analyze", json=payload)
print("Status code:", r.status_code)
d = r.json()
# This returns 400 because no key - expected. Let's test agent internals directly
print("Response:", d)
print()

# Test the agent directly (bypassing Flask)
import sys, os
sys.path.insert(0, r"c:\Users\olatu\Downloads\Pipeline Run Status")
import agent_backend as agent

# Agent should already be initialized by the running server
# Test status
status = agent.get_status()
print("Agent status:", status)
print()

# Test just embedding + prediction (no Claude)
from langchain_core.messages import HumanMessage
from sentence_transformers import SentenceTransformer
import numpy as np

emb_model = agent._embedding_model
clf_model = agent._champion_model

test_regs = [
    "EPA requires industrial facilities to reduce emissions by 30% with penalties up to $50,000 per day",
    "SBA announces $200 million in grants to support minority-owned businesses",
    "USDA publishes quarterly crop production statistics",
]

for text in test_regs:
    emb  = emb_model.encode([text])
    pred = clf_model.predict(emb)[0]
    prob = clf_model.predict_proba(emb)[0]
    sent = agent.SENTIMENTS[pred]
    conf = float(max(prob)) * 100
    print(f"Regulation: {text[:70]}...")
    print(f"  Sentiment:   {sent}")
    print(f"  Confidence:  {conf:.1f}%")
    print(f"  Pos/Neu/Neg: {prob[0]*100:.1f}% / {prob[1]*100:.1f}% / {prob[2]*100:.1f}%")
    print()
