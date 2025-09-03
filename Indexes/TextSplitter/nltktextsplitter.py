from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.text_splitter import NLTKTextSplitter
import nltk

load_dotenv()

# Ensure required NLTK data is available
nltk.download("punkt")
nltk.download("punkt_tab")

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# Load a ling document
file_path = "/home/santoshn/Projects/AI/activeloop-ai/data/llms.txt"
with open(file_path, encoding="unicode_escape") as f:
    text = f.read()

text_splitter = NLTKTextSplitter(chunk_size=100, chunk_overlap=20)
texts = text_splitter.split_text(text)

print(texts)