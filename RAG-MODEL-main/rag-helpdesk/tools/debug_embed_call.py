import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / '.env')

import requests

api_key = os.getenv('GE_API_KEY') or os.getenv('LLM_API_KEY')
base = os.getenv('LLM_BASE_URL', '')
model = os.getenv('EMBEDDING_MODEL', '')

print('Using base:', base)
print('Using model:', model)
print('Using key present:', bool(api_key))

endpoint = base.rstrip('/') + '/embeddings'
headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
payload = {'model': model, 'input': ['test']}

try:
    resp = requests.post(endpoint, headers=headers, json=payload, timeout=30)
    print('Status:', resp.status_code)
    try:
        print('JSON:', json.dumps(resp.json(), indent=2))
    except Exception:
        print('Text:', resp.text)
except Exception as e:
    print('Request failed:', e)
