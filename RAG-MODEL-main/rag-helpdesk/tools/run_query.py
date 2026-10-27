from src.rag_pipeline import RAGPipeline

p = RAGPipeline()
print('Status:', p.status())
res = p.answer_question('What is Chandrayaan and what did it discover on the Moon?')
print('Answer:', res.get('answer'))
print('Sources:', res.get('sources'))
print('Retrieved chunks:', len(res.get('retrieved_chunks', [])))
