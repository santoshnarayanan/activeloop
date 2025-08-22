from dotenv import load_dotenv
import os

load_dotenv()

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

text = "What would be a good company name for a company that makes colorful socks?"

print(llm.invoke(text))