import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embeddings import EmbeddingService


if __name__ == '__main__':
    svc = EmbeddingService()
    print('Using model:', svc.model)
    sample = 'Test embedding from RAG helpdesk.'
    emb = svc.create_embedding(sample)
    print('Embedding type:', type(emb))
    if emb is None:
        print('Embedding was None (failed).')
    else:
        print('Embedding length:', len(emb))
        print('First 5 dims:', emb[:5])
