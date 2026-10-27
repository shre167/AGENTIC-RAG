"""Test script for DeepEval + Langfuse integration."""
import os
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_pipeline import RAGPipeline


def test_single_query():
    """Test single query with DeepEval evaluation and Langfuse tracing."""
    print("\n" + "="*80)
    print("DeepEval + Langfuse Integration Test")
    print("="*80 + "\n")
    
    # Check configuration
    print("Configuration Check:")
    print(f"  ENABLE_DEEPEVAL: {os.getenv('ENABLE_DEEPEVAL', 'false')}")
    print(f"  ENABLE_RAGAS_EVAL: {os.getenv('ENABLE_RAGAS_EVAL', 'false')}")
    print(f"  ENABLE_LANGFUSE: {os.getenv('ENABLE_LANGFUSE', 'false')}")
    print(f"  LANGFUSE_PUBLIC_KEY: {'[Set]' if os.getenv('LANGFUSE_PUBLIC_KEY') else '[Not set]'}")
    print(f"  LANGFUSE_SECRET_KEY: {'[Set]' if os.getenv('LANGFUSE_SECRET_KEY') else '[Not set]'}")
    print(f"  DEEPEVAL_COMPARE_ALL_RETRIEVERS: {os.getenv('DEEPEVAL_COMPARE_ALL_RETRIEVERS', 'false')}")
    print()
    
    # Initialize pipeline
    print("Initializing RAG Pipeline...")
    pipeline = RAGPipeline()
    print("[OK] Pipeline initialized\n")
    
    # Test query
    query = "What is cosmology?"
    retrieval_mode = "hybrid"
    
    print(f"Running query: '{query}'")
    print(f"Retrieval mode: {retrieval_mode}\n")
    
    try:
        response = pipeline.answer_question(
            query,
            retrieval_mode=retrieval_mode,
        )
        
        # Check response
        print("\n" + "="*80)
        print("Response Summary")
        print("="*80)
        print(f"Response type: {response.get('response_type')}")
        print(f"Answer preview: {(response.get('raw_answer') or '')[:200]}...")
        print()
        
        # Check DeepEval
        if "deepeval" in response:
            print("DeepEval Results:")
            deepeval_data = response["deepeval"]
            if "by_retrieval_mode" in deepeval_data:
                for mode, result in deepeval_data["by_retrieval_mode"].items():
                    print(f"\n  Mode: {mode}")
                    print(f"  Status: {result.get('status')}")
                    if result.get("scores"):
                        print("  Scores:")
                        for metric, metric_data in result["scores"].items():
                            if isinstance(metric_data, dict):
                                score = metric_data.get("score", 0)
                                reason = metric_data.get("reason", "")
                                print(f"    {metric}: {score:.4f}")
                                if reason:
                                    print(f"      Reason: {reason[:100]}...")
                            else:
                                print(f"    {metric}: {float(metric_data):.4f}")
                    else:
                        print(f"  Reason: {result.get('reason', 'N/A')}")
        else:
            print("DeepEval: Not evaluated (check ENABLE_DEEPEVAL)")
        
        print()
        
        # Check Langfuse
        if "observability" in response and "langfuse_trace_id" in response["observability"]:
            trace_id = response["observability"]["langfuse_trace_id"]
            print(f"Langfuse Trace ID: {trace_id}")
            print(f"View in dashboard: https://cloud.langfuse.com/trace/{trace_id}")
        else:
            print("Langfuse: Not enabled (check ENABLE_LANGFUSE and credentials)")
        
        print("\n" + "="*80)
        print("[OK] Test completed successfully")
        print("="*80 + "\n")
        
        return response
        
    except Exception as exc:
        print(f"\n[FAIL] Test failed: {exc}")
        import traceback
        traceback.print_exc()
        return None


def test_comparison():
    """Test retriever comparison mode."""
    print("\n" + "="*80)
    print("Retriever Comparison Test")
    print("="*80 + "\n")
    
    if os.getenv("DEEPEVAL_COMPARE_ALL_RETRIEVERS", "false").lower() not in ("true", "1", "yes"):
        print("[WARN] DEEPEVAL_COMPARE_ALL_RETRIEVERS is not enabled")
        print("Set DEEPEVAL_COMPARE_ALL_RETRIEVERS=true to test comparison mode")
        return
    
    pipeline = RAGPipeline()
    query = "What is JWST?"
    
    print(f"Running query: '{query}'")
    print("This will evaluate BM25, Dense/Vector, and Hybrid modes\n")
    
    try:
        response = pipeline.answer_question(query, retrieval_mode="hybrid")
        
        if "deepeval" in response and "by_retrieval_mode" in response["deepeval"]:
            print("\nRetriever Comparison Results:")
            print("-" * 80)
            
            for mode in ["bm25", "vector", "hybrid"]:
                if mode in response["deepeval"]["by_retrieval_mode"]:
                    result = response["deepeval"]["by_retrieval_mode"][mode]
                    print(f"\n{mode.upper()}:")
                    print(f"  Status: {result.get('status')}")
                    if result.get("scores"):
                        for metric, metric_data in result["scores"].items():
                            if isinstance(metric_data, dict):
                                score = metric_data.get("score", 0)
                                print(f"    {metric}: {score:.4f}")
                            else:
                                print(f"    {metric}: {float(metric_data):.4f}")
            
            print("\n" + "-" * 80)
            print("[OK] Comparison completed")
        else:
            print("[WARN] No comparison data found in response")
        
    except Exception as exc:
        print(f"[FAIL] Comparison test failed: {exc}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run single query test
    response = test_single_query()
    
    # Run comparison test if enabled
    if response and os.getenv("DEEPEVAL_COMPARE_ALL_RETRIEVERS", "false").lower() in ("true", "1", "yes"):
        print("\n")
        test_comparison()
