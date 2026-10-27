import os
from src.rag_pipeline import RAGPipeline

# Use synthetic embeddings for a reliable local test
os.environ['DEV_EMBEDDINGS'] = 'true'

p = RAGPipeline()
print('Status:', p.status())

queries = [
    'What is a black hole and what is an event horizon?',
    'Describe the atmosphere and moons of Mars.',
    'What did the Chandrayaan missions discover on the Moon?',
    'What are exoplanets and how are they detected?',
    'What is the James Webb Space Telescope designed to observe?'
]

for q in queries:
    print('\n=== Query:', q)
    r = p.answer_question(q)
    print('Answer:')
    print(r.get('answer'))
    print('\nSources:', r.get('sources'))
    print('Evidence:', r.get('evidence'))
