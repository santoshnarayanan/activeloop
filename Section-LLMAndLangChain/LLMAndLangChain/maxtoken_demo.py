from dotenv import load_dotenv
import os

from langchain_text_splitters import split_text_on_tokens

load_dotenv()

from langchain_openai import OpenAI
from langchain.text_splitter import  RecursiveCharacterTextSplitter

# load env variables
llm = OpenAI(model="gpt-4o-mini", temperature=0.0)

# Define input
input_text = "your_long_input_text"

# Determine the maximum number of tokens from documentation
max_tokens = 4097  # Adjust based on the model's max token limit

# Split the text into chunks based on the max token limit
text_splitter = RecursiveCharacterTextSplitter(chunk_size=max_tokens, chunk_overlap=0)
text_chunks = text_splitter.split_text(input_text)

# Process each chunk separately
results = []
for chunk in text_chunks:
    result = llm.invoke(chunk)
    results.append(result)


# Combine results if needed
final_results = "\n".join(results)
