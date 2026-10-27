from src.rag_pipeline import RAGPipeline
from src.vector_store import ChromaVectorStore

p = RAGPipeline()
print('pipeline.status():', p.status())
vs = ChromaVectorStore()
print('collection count:', vs.get_collection_count())
