import os
import json
from dotenv import load_dotenv
import requests

load_dotenv()
key = os.getenv('GE_API_KEY') or os.getenv('LLM_API_KEY')
if not key:
    print('No GE_API_KEY or LLM_API_KEY found in .env or environment')
    raise SystemExit(1)

base_url = os.getenv('LLM_BASE_URL', '')
if not base_url:
    print('No LLM_BASE_URL found in .env or environment')
    raise SystemExit(1)

endpoints = [
    f"{base_url.rstrip('/')}/embeddings",
]

embed_model = os.getenv('EMBEDDING_MODEL', '')
if not embed_model:
    print('No EMBEDDING_MODEL found in .env or environment')
    raise SystemExit(1)

body = {"input": "test embedding", "model": embed_model}
headers = {"Authorization": f"Bearer {key}", "x-api-key": key, "Content-Type": "application/json"}

for ep in endpoints:
    print('\nTrying:', ep)
    try:
        resp = requests.post(ep, headers=headers, json=body, timeout=30)
        print('Status:', resp.status_code)
        try:
            print('Response JSON:', json.dumps(resp.json(), indent=2))
        except Exception:
            print('Response text:', resp.text)
    except Exception as e:
        print('Request failed:', e)
