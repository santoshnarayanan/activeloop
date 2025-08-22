from dotenv import load_dotenv
import os

from langchain.chains.summarize import load_summarize_chain

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# Load the summarization chain
summarization_chain = load_summarize_chain(llm)

# load the document using PyPDFLoader
current_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(os.path.dirname(current_dir), 'data', 'The One Page Linux Manual.pdf')
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Summarize the document
summary = summarization_chain.run(documents)
print(summary)