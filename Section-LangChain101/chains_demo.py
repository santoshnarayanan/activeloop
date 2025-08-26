from dotenv import load_dotenv
import os

load_dotenv()

from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain

llm = OpenAI(model="gpt-4o-mini", temperature=0.0)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

chain = LLMChain(llm=llm, prompt=prompt)
 #Run the chain only specifying the input variable
print(chain.run("eco-friendly water bottles"))