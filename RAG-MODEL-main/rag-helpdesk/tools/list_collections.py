import os
from src.config import VECTORSTORE_DIR
import chromadb
from chromadb.config import Settings

p = str(VECTORSTORE_DIR)
print('VECTORSTORE_DIR=', p)
if hasattr(chromadb, "PersistentClient"):
    client = chromadb.PersistentClient(path=p, settings=Settings(anonymized_telemetry=False))
else:
    client = chromadb.Client(Settings(persist_directory=p, anonymized_telemetry=False))
print('client created')
try:
    cols = client.list_collections()
    print('collections:', cols)
except Exception as e:
    print('list_collections error:', e)

# try get_or_create
c = client.get_or_create_collection('knowledge_base')
print('got collection:', c.name)
try:
    cnt = c.count()
    print('collection count:', cnt)
except Exception as e:
    print('count error:', e)
