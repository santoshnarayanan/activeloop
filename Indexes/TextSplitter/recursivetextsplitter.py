from dotenv import load_dotenv
import os

from langchain.chains.summarize import load_summarize_chain

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# Load the summarization chain
summarization_chain = load_summarize_chain(llm)

# load the document using PyPDFLoader
pdf_path = "/home/santoshn/Projects/AI/activeloop-ai/data/test.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10, length_function=len)
texts = text_splitter.split_documents(pages)
for doc in texts:
    print(doc.page_content)