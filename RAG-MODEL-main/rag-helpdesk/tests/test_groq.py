import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

print("BASE URL:", os.getenv("LLM_BASE_URL"))
print("MODEL:", os.getenv("LLM_MODEL"))
print("API KEY EXISTS:", bool(os.getenv("GE_API_KEY")))

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL"),
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("GE_API_KEY"),
)

response = llm.invoke("Say hello in one sentence.")

print(response.content)