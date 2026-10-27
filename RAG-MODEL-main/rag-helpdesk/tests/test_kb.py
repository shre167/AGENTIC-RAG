import sys
from pathlib import Path
sys.path.insert(0, r"c:\Users\shrey\Downloads\RAG-MODEL-main\RAG-MODEL-main\rag-helpdesk")
from src.config import KNOWLEDGE_BASE_DIR
from src.document_loader import load_documents, chunk_documents, list_knowledge_files

kb = Path(KNOWLEDGE_BASE_DIR)
files = list_knowledge_files(kb)
print("Files in KB:", [f.name for f in files])
