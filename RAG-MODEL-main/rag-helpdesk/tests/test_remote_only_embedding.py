import os
import requests

KEY = os.environ["GE_API_KEY"]  # current User or Studio key
OPENAI_BASE = "https://openai.generative.engine.capgemini.com/v1"
REST_BASE = "https://api.generative.engine.capgemini.com"

checks = [
    ("OpenAI host GET /models (Bearer)", "GET", f"{OPENAI_BASE}/models",
     {"Authorization": f"Bearer {KEY}"}, None),
    ("OpenAI host GET /embeddings/models (Bearer)", "GET", f"{OPENAI_BASE}/embeddings/models",
     {"Authorization": f"Bearer {KEY}"}, None),
    ("OpenAI host POST /embeddings (Bearer)", "POST", f"{OPENAI_BASE}/embeddings",
     {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
     {"model": "amazon.titan-embed-text-v2:0", "input": "Hello world"}),
    ("REST host GET /v1/health (x-api-key)", "GET", f"{REST_BASE}/v1/health",
     {"x-api-key": KEY}, None),
    ("REST host GET /v1/models (x-api-key)", "GET", f"{REST_BASE}/v1/models",
     {"x-api-key": KEY}, None),
]

for label, method, url, headers, body in checks:
    try:
        r = requests.request(method, url, headers=headers, json=body, timeout=30)
        print(f"{label}: HTTP {r.status_code} {r.text[:200]}")
    except requests.RequestException as exc:
        print(f"{label}: request failed: {exc}")