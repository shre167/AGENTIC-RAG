import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv("../.env")

print("API KEY FOUND:", bool(os.getenv("GE_API_KEY")))
print("BASE URL:", os.getenv("LLM_BASE_URL"))
print("MODEL:", os.getenv("LLM_MODEL"))

client = OpenAI(
    api_key=os.getenv("GE_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)

response = client.chat.completions.create(
    model=os.getenv("LLM_MODEL"),
    messages=[
        {"role": "user", "content": "Say hello"}
    ]
)

print("LLM RESPONSE:", response.choices[0].message.content)