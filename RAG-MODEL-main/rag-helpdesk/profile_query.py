#!/usr/bin/env python3
"""
Profile query latency to identify performance bottlenecks.
"""

import sys
import time
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    print("Loading pipeline...")
    t0 = time.perf_counter()
    
    try:
        from src.rag_pipeline.pipeline import RAGPipeline
        
        pipeline = RAGPipeline(
            vectorstore_path=project_root / "vectorstore",
            knowledge_base_path=project_root / "knowledge_base",
        )
        
        init_ms = (time.perf_counter() - t0) * 1000
        print(f"Pipeline init: {init_ms:.0f}ms")
        print()
        
        # Test a simple query
        print("Running test query: 'What is on page 10?'")
        t1 = time.perf_counter()
        response = pipeline.answer_question("What is on page 10?", retrieval_mode="hybrid")
        query_ms = (time.perf_counter() - t1) * 1000
        
        print(f"Total query time: {query_ms:.0f}ms")
        print(f"Answer: {response.get('answer', 'N/A')[:100]}...")
        print()
        print(f"Chunks retrieved: {response.get('num_retrieved', 0)}")
        
        # Check trace info
        trace = response.get("trace", {})
        if trace:
            print()
            print("Trace info:")
            for key, value in trace.items():
                if "ms" in str(key).lower() or "time" in str(key).lower():
                    print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
