from dotenv import load_dotenv
import os

load_dotenv()


from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
with get_openai_callback() as cb:
    response = llm.invoke("Tell me a joke about AI.")
    print(response)