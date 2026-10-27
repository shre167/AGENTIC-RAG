import fitz
from pathlib import Path

doc = fitz.open(r"c:\Users\shrey\Downloads\RAG-MODEL-main\RAG-MODEL-main\rag-helpdesk\knowledge_base\Deep Work.pdf")
print("Total pages in Deep Work:", len(doc))
for i in range(min(30, len(doc))):
    text = doc[i].get_text("text").strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for l in lines:
        if any(w in l.lower() for w in ["chapter", "part ", "rule #", "introduction", "conclusion"]):
            print(f"Page {i+1}: {l}")
