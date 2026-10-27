#!/usr/bin/env python3
"""Test query to diagnose language model failure."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def main():
    try:
        from src.rag_pipeline.pipeline import RAGPipeline
        from src.config import KNOWLEDGE_BASE_DIR, VECTORSTORE_DIR
        
        print("Initializing pipeline...")
        pipeline = RAGPipeline(
            vectorstore_path=Path(VECTORSTORE_DIR),
            knowledge_base_path=Path(KNOWLEDGE_BASE_DIR),
        )
        print("Pipeline ready.")
        print()
        
        print("Running test query...")
        response = pipeline.answer_question(
            "whats on page 60 of the book?", 
            retrieval_mode="hybrid"
        )
        print("Query completed.")
        print()
        
        if response.get("error"):
            print(f"ERROR: {response.get('error')}")
            return 1
            
        answer = response.get("answer", "No answer")
        print(f"Answer preview: {answer[:200]}...")
        print(f"Retrieved chunks: {response.get('num_retrieved', 0)}")
        
        return 0
        
    except Exception as e:
        print(f"EXCEPTION: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())