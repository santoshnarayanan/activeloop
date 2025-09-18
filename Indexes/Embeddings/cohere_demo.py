from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings
import cohere
import os

load_dotenv()
cohere_embeddings = CohereEmbeddings(
    model="embed-multilingual-v2.0",
	cohere_api_key=os.getenv("COHERE_API_KEY")
    )

# Define a list of texts
texts = [
    "Hello from Cohere!", 
    "مرحبًا من كوهير!", 
    "Hallo von Cohere!",  
    "Bonjour de Cohere!", 
    "¡Hola desde Cohere!", 
    "Olá do Cohere!",  
    "Ciao da Cohere!", 
    "您好，来自 Cohere！", 
    "कोहेरे से नमस्ते!"
]

# Generate embeddings
embeddings = cohere_embeddings.embed_documents(texts)

# Print the embeddings
for text, embedding in zip(texts, embeddings):
    print(f"Text: {text}")
    print(f"Embedding: {embedding[:5]}") # print first 5 dimensions of each embedding
    
