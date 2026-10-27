import os
import json
import requests

BASE = "https://openai.generative.engine.capgemini.com/v1"  # US; already includes /v1
AUTH = {"Authorization": f"Bearer {os.environ['GE_API_KEY']}"}

r = requests.get(f"{BASE}/embeddings/models", headers=AUTH, timeout=30)
print("HTTP", r.status_code)
body = r.json()

print("top-level type:", type(body).__name__)
if isinstance(body, dict):
    print("top-level keys:", list(body.keys()))

# Print the full raw response (contains no key; it is only the model list)
print(json.dumps(body, indent=2)[:4000])