
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings

load_dotenv()

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cpu"}

hf = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

print(hf.embed_query("What is the capital of France?"))

documents = [
    "The cat is on the mat.",
    "There is a cat on the mat.",
    "The dog is in the yard.",
    "There is a dog in the yard.",
]


embeddings = hf.embed_documents(documents)
print(embeddings)