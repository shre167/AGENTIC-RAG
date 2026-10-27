# test_fixed.py
from src.rag_pipeline.generation import ask_llm

print("Testing fixed generation...")
answer = ask_llm(
    question="What is 2+2?",
    context="Mathematical fact: 2 plus 2 equals 4."
)
print(f"Answer: {answer}")