from dotenv import load_dotenv
import os
load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

prompt = PromptTemplate.from_template("Question: {question}\nAnswer:")

question = "What is the capital of France?"

llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-base",
    task="text2text-generation",  # Correct task for T5 models
    temperature=0.0,
    max_new_tokens=64,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)
response = (prompt | llm).invoke({"question": question})
print("Answer:", response)



