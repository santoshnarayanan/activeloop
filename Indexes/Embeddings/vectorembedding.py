from dotenv import load_dotenv
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings

load_dotenv()
# Define the documents
documents = [
    "The cat is on the mat.",
    "There is a cat on the mat.",
    "The dog is in the yard.",
    "There is a dog in the yard.",
]
# Initialize the embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# generate embeddings for the documents
document_embeddings = embeddings.embed_documents(documents)

# Perform similarity search for given query
query = "A cat is in the yard."
query_embedding = embeddings.embed_query(query)

# Calculate similarlity scores
similarities_scores = cosine_similarity([query_embedding], document_embeddings)[0]

# Get the most similar document
most_similar_index = np.argmax(similarities_scores)
most_similar_document = documents[most_similar_index]

print(f"Most similar document:'{query}':")
print(most_similar_document)