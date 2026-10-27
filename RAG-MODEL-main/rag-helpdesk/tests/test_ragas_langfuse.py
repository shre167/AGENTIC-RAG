"""Test script for RAGAS + Langfuse integration."""
import os
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_pipeline import RAGPipeline


def test_single_query():
    """Test single query with RAGAS evaluation and Langfuse tracing."""
    print("\n" + "="*80)
    print("RAGAS + Langfuse Integration Test")
    print("="*80 + "\n")
    
    # Check configuration
    print("Configuration Check:")
    print(f"  ENABLE_RAGAS_EVAL: {os.getenv('ENABLE_RAGAS_EVAL', 'false')}")
    print(f"  ENABLE_LANGFUSE: {os.getenv('ENABLE_LANGFUSE', 'false')}")
    print(f"  LANGFUSE_PUBLIC_KEY: {'✓ Set' if os.getenv('LANGFUSE_PUBLIC_KEY') else '✗ Not set'}")
    print(f"  LANGFUSE_SECRET_KEY: {'✓ Set' if os.getenv('LANGFUSE_SECRET_KEY') else '✗ Not set'}")
    print(f"  RAGAS_COMPARE_ALL_RETRIEVERS: {os.getenv('RAGAS_COMPARE_ALL_RETRIEVERS', 'false')}")
    print()
    
    # Initialize pipeline
    print("Initializing RAG Pipeline...")
    pipeline = RAGPipeline()
    print("✓ Pipeline initialized\n")
    
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
        
        # Check RAGAS
        if "ragas" in response:
            print("RAGAS Results:")
            ragas_data = response["ragas"]
            if "by_retrieval_mode" in ragas_data:
                for mode, result in ragas_data["by_retrieval_mode"].items():
                    print(f"\n  Mode: {mode}")
                    print(f"  Status: {result.get('status')}")
                    if result.get("scores"):
                        print("  Scores:")
                        for metric, score in result["scores"].items():
                            print(f"    {metric}: {score:.4f}")
                    else:
                        print(f"  Reason: {result.get('reason', 'N/A')}")
        else:
            print("RAGAS: Not evaluated (check ENABLE_RAGAS_EVAL)")
        
        print()
        
        # Check Langfuse
        if "observability" in response and "langfuse_trace_id" in response["observability"]:
            trace_id = response["observability"]["langfuse_trace_id"]
            print(f"Langfuse Trace ID: {trace_id}")
            print(f"View in dashboard: https://cloud.langfuse.com/trace/{trace_id}")
        else:
            print("Langfuse: Not enabled (check ENABLE_LANGFUSE and credentials)")
        
        print("\n" + "="*80)
        print("✓ Test completed successfully")
        print("="*80 + "\n")
        
        return response
        
    except Exception as exc:
        print(f"\n✗ Test failed: {exc}")
        import traceback
        traceback.print_exc()
        return None


def test_comparison():
    """Test retriever comparison mode."""
    print("\n" + "="*80)
    print("Retriever Comparison Test")
    print("="*80 + "\n")
    
    if os.getenv("RAGAS_COMPARE_ALL_RETRIEVERS", "false").lower() not in ("true", "1", "yes"):
        print("⚠ RAGAS_COMPARE_ALL_RETRIEVERS is not enabled")
        print("Set RAGAS_COMPARE_ALL_RETRIEVERS=true to test comparison mode")
        return
    
    pipeline = RAGPipeline()
    query = "What is JWST?"
    
    print(f"Running query: '{query}'")
    print("This will evaluate BM25, Dense/Vector, and Hybrid modes\n")
    
    try:
        response = pipeline.answer_question(query, retrieval_mode="hybrid")
        
        if "ragas" in response and "by_retrieval_mode" in response["ragas"]:
            print("\nRetriever Comparison Results:")
            print("-" * 80)
            
            for mode in ["bm25", "vector", "hybrid"]:
                if mode in response["ragas"]["by_retrieval_mode"]:
                    result = response["ragas"]["by_retrieval_mode"][mode]
                    print(f"\n{mode.upper()}:")
                    print(f"  Status: {result.get('status')}")
                    if result.get("scores"):
                        for metric, score in result["scores"].items():
                            print(f"    {metric}: {score:.4f}")
            
            print("\n" + "-" * 80)
            print("✓ Comparison completed")
        else:
            print("⚠ No comparison data found in response")
        
    except Exception as exc:
        print(f"✗ Comparison test failed: {exc}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run single query test
    response = test_single_query()
    
    # Run comparison test if enabled
    if response and os.getenv("RAGAS_COMPARE_ALL_RETRIEVERS", "false").lower() in ("true", "1", "yes"):
        print("\n")
        test_comparison()
