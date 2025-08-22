from dotenv import load_dotenv
import os

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?"),
    AIMessage(content="The capital of France is Paris.")
]

prompt = HumanMessage( content="I'd like to know more about the city you just mentioned.")
messages.append(prompt)
print(chat.invoke(messages))