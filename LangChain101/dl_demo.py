from dotenv import load_dotenv
import os

from langchain.agents import create_react_agent, AgentExecutor

load_dotenv()

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_deeplake import DeeplakeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.tools import Tool
from langchain import hub
import deeplake
# Instantiate the LLM and embeddings models
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Create our documents
texts = [
    "Napoleon Bonaparte was born on 15 August 1769",
    "Louis XIV was born on 5 September 1638",
    "Lady Gaga was born on 28 March 1986",
    "Michael Jeffrey Jordan was born on 17 February 1963",
]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.create_documents(texts)

# Use your Activeloop org id (default is your username)
my_activeloop_org_id = "santosh"
my_activeloop_dataset_name = "langchain_course_from_zero_to_hero"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

# If the dataset already exists, you can delete it to start fresh
if deeplake.exists(dataset_path):
    deeplake.delete(dataset_path)

# Create DeepLake vector store (new way)
db = DeeplakeVectorStore(dataset_path=dataset_path, embedding_function=embeddings)

# Add the documents
db.add_documents(docs)

# Create a RetrievalQA chain
retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever()
)
print(retrieval_qa.invoke({"query": "When was Napoleon born?"})["result"])

#Wrap RetrievalQA as a Tool (expects str in, str out)
def _ask_retriever(query: str) -> str:
    return retrieval_qa.invoke({"query": query})["result"]

retrieval_tool = Tool(
    name="Retrieval QA System",
    func=_ask_retriever,
    description="Useful for answering questions grounded in the vector store."
)

# ----- Create a React agent that can call the tool -----
react_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm=llm, tools=[retrieval_tool], prompt=react_prompt)
agent_executor = AgentExecutor(agent=agent, tools=[retrieval_tool], verbose=True)

# Run the agent
resp = agent_executor.invoke({"input": "When was Napoleon Bonaparte born?"})
print(resp["output"])