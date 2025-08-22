from dotenv import load_dotenv
import os

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

messages = [
	SystemMessage(content="You are a helpful assistant that translates English to French."),
	HumanMessage(content="Translate the following sentence: I love programming.")
]

print(llm.invoke(messages).content)