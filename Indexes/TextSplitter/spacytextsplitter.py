from dotenv import load_dotenv
from langchain.text_splitter import SpacyTextSplitter

load_dotenv()

# Load a long document
file_path = "/home/santoshn/Projects/AI/activeloop-ai/data/llms.txt"
with open(file_path, encoding="unicode_escape") as f:
    text = f.read()

# Use spaCy-based text splitter
text_splitter = SpacyTextSplitter(chunk_size=500, chunk_overlap=20, pipeline="en_core_web_sm")
texts = text_splitter.split_text(text)

print(texts[0])  # First chunk
