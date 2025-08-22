from dotenv import load_dotenv
import os

from langchain.chains.summarize import load_summarize_chain

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["question"],
    template="Question: {question}\nAnswer:"
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

#chaining the two chains together
chain = prompt | llm

question = "What is the meaning of life?"
print(chain.invoke(question))