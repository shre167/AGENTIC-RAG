#!/usr/bin/env python3
"""Trace the token_optimizer and context sections."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_pipeline.pipeline import RAGPipeline
from src.config import KNOWLEDGE_BASE_DIR, VECTORSTORE_DIR

pipeline = RAGPipeline(
    vectorstore_path=Path(VECTORSTORE_DIR),
    knowledge_base_path=Path(KNOWLEDGE_BASE_DIR),
)
response = pipeline.answer_question("what is time management?", retrieval_mode="hybrid")
trace = response.get("trace") or {}

print("=== token_optimizer ===")
print(json.dumps(trace.get("token_optimizer", {}), indent=2, default=str))
print()
print("=== context ===")
print(json.dumps(trace.get("context", {}), indent=2, default=str))
print()
print("=== generation status ===")
gen = trace.get("generation", {})
print("status:", gen.get("status"))
print("error:", gen.get("error"))
