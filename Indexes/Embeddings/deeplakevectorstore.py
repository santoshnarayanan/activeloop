from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_deeplake import DeeplakeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import deeplake

load_dotenv()

# create our documents
texts = [
    "Napoleon Bonaparte was born in 15 August 1769",
    "Louis XIV was born in 5 September 1638",
    "Lady Gaga was born in 28 March 1986",
    "Michael Jeffrey Jordan was born in 17 February 1963"
]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.create_documents(texts)

# The next step is to create a Deep Lake database and load our documents into it.
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

#create Deep Lake dataset
my_activeloop_org_id = "santosh"
my_activeloop_dataset_name = "langchain_course_embeddings"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

# If the dataset already exists, you can delete it to start fresh
if deeplake.exists(dataset_path):
    deeplake.delete(dataset_path)

# Create vectorstore
db = DeeplakeVectorStore(dataset_path=dataset_path, embedding_function=embeddings)

# Add documents
db.add_documents(docs)

# Create retriever from db
retriever = db.as_retriever()

# instantiate LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions based on the provided documents: {context}"),
    ("human", "{input}"),
])


# Create a document-combining chain
document_chain = create_stuff_documents_chain(llm, prompt)

# Create retrieval chain
qa_chain = create_retrieval_chain(retriever, document_chain)

# Ask question
response = qa_chain.invoke({"input": "When was Michael Jordan born?"})
print(response["answer"])