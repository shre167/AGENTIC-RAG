import os
from pathlib import Path
import sys

# ensure project root is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Use real embeddings from .env configuration instead of synthetic
# os.environ['DEV_EMBEDDINGS'] = 'true'  # REMOVED - was causing dimension mismatch

from src.rag_pipeline import RAGPipeline

p = RAGPipeline()
print('Pipeline status:', p.status())
q = 'What did Chandrayaan discover on the Moon?'
print('Query:', q)
emb = p.embedding_service.create_embedding(q)
print('Embedding length:', len(emb) if emb else 'None')
results = p.vector_store.query(emb, n_results=5)
print('Raw results keys:', list(results.keys()))
docs = results.get('documents', [])
metas = results.get('metadatas', [])
if docs and isinstance(docs[0], list):
    docs = docs[0]
if metas and isinstance(metas[0], list):
    metas = metas[0]

for i, d in enumerate(docs):
    meta = metas[i] if i < len(metas) else {}
    print(f'--- Result {i+1} ---')
    print('Filename:', meta.get('filename') or meta.get('source'))
    print('Text snippet:', d[:300])
