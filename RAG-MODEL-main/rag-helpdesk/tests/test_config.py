import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

repo_root = Path(__file__).resolve().parent.parent
load_dotenv(repo_root / ".env")

base_url = os.getenv("LLM_BASE_URL")
api_key = os.getenv("GE_API_KEY") or os.getenv("OPENAI_API_KEY")
llm_model = os.getenv("LLM_MODEL")
embed_model = os.getenv("EMBEDDING_MODEL")

if not base_url:
    raise RuntimeError("LLM_BASE_URL is missing. Set it in .env or in the shell.")
if not api_key:
    raise RuntimeError("GE_API_KEY is missing. Set it in .env or in the shell.")
if not llm_model:
    raise RuntimeError("LLM_MODEL is missing. Set it in .env or in the shell.")
if not embed_model:
    raise RuntimeError("EMBEDDING_MODEL is missing. Set it in .env or in the shell.")

client = OpenAI(
    base_url=base_url,
    api_key=api_key,
    timeout=60.0,
)

chat = client.chat.completions.create(
    model=llm_model,
    messages=[{"role": "user", "content": "Reply with one short sentence."}],
    max_completion_tokens=256,
)
print("chat:", chat.choices[0].message.content.strip())

candidates = [
    embed_model,
    "text-embedding-004",
    "openai.text-embedding-3-small",
    "amazon.titan-embed-text-v2:0",
]

last_error = None
for model_name in candidates:
    try:
        emb = client.embeddings.create(
            model=model_name,
            input="Smart time management.",
        )
        print(f"embedding model used: {model_name}")
        print("embedding dimensions:", len(emb.data[0].embedding))
        break
    except Exception as exc:  # pragma: no cover - diagnostic script only
        last_error = exc
        print(f"embedding model failed: {model_name} -> {type(exc).__name__}: {exc}")
else:
    raise RuntimeError(f"All embedding model candidates failed. Last error: {last_error}") from last_error