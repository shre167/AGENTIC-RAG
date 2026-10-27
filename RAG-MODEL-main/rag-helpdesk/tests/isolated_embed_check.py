# pip install --upgrade openai python-dotenv
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

print("Base URL:", os.environ.get("LLM_BASE_URL"))
print("Key starts with:", os.environ.get("GE_API_KEY", "")[:8], "...")

client = OpenAI(
    base_url=os.environ["LLM_BASE_URL"],
    api_key=os.environ["GE_API_KEY"],
)

try:
    response = client.embeddings.create(
        model="amazon.titan-embed-text-v2:0",
        input="test sentence",
    )
    print("SUCCESS. First 5 embedding values:", response.data[0].embedding[:5])
except Exception as e:
    print("FAILED with exception:")
    print(type(e).__name__, "-", e)

sys.stdout.flush()
# Add this below your embeddings test in isolated_embed_check.py

try:
    chat_response = client.chat.completions.create(
        model="amazon.nova-lite-v1:0",
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
        max_completion_tokens=50,
    )
    print("LLM SUCCESS:", chat_response.choices[0].message.content)
except Exception as e:
    print("LLM FAILED with exception:")
    print(type(e).__name__, "-", e)