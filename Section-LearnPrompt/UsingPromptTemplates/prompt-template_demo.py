from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)


template = """Answer the question based on the context below. If the
question cannot be answered using the information provided, answer
with "I don't know".
Context: Quantum computing is an emerging field that leverages quantum mechanics to solve complex problems faster than classical computers.
...
Question: {query}
Answer: """

prompt_template = PromptTemplate(
    input_variables=["query"],
    template=template
)

chain = prompt_template | llm

question = "What is the main advantage of quantum computing over classical computing?"
response = chain.invoke({"query": question})

print("Question:", question)
print("Answer:", response.content)
