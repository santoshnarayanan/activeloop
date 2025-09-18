from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_deeplake import DeeplakeVectorStore
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.document_loaders import SeleniumURLLoader
import deeplake

load_dotenv()

# Load documents from a URL
# we'll use information from the following articles
urls = ['https://beebom.com/what-is-nft-explained/',
        'https://beebom.com/how-delete-spotify-account/',
        'https://beebom.com/how-download-gif-twitter/',
        'https://beebom.com/how-use-chatgpt-linux-terminal/',
        'https://beebom.com/how-delete-spotify-account/',
        'https://beebom.com/how-save-instagram-story-with-music/',
        'https://beebom.com/how-install-pip-windows/',
        'https://beebom.com/how-check-disk-usage-linux/']

# 1. Split the documents into chunks and compute their embeddings
loader = SeleniumURLLoader(urls=urls)
docs_not_splitted = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(docs_not_splitted)

# Compute embeddings and store in Deep Lake vector store
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# create Deep Lake dataset
my_activeloop_org_id = "santosh"
my_activeloop_dataset_name = "langchain_course_customer_support"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
# If the dataset already exists, you can delete it to start fresh
try:
    if deeplake.exists(dataset_path):
        deeplake.delete(dataset_path)
except deeplake.AuthorizationError:
    print(f"Dataset {dataset_path} does not exist yet. Creating fresh one...")


# Create vectorstore
db = DeeplakeVectorStore(dataset_path=dataset_path, embedding_function=embeddings)
# Add documents
db.add_documents(docs)

# let's see the top relevant documents to a specific query
query = "how to check disk usage in linux?"
docs = db.similarity_search(query)
print(docs[0].page_content)

# 2. Craft a prompt for GPT-3 using the suggested strategies
## create a prompt template that incorporates role-prompting, relevant Knowledge Base information,
## and the user's question

template = """You are an exceptional customer support chatbot that gently answer questions.

You know the following context information.

{chunks_formatted}

Answer to the following question from a customer. Use only information from the previous context information. Do not invent stuff.

Question: {query}

Answer:"""

prompt = PromptTemplate(
    input_variables=["chunks_formatted", "query"],
    template=template,
)

# 3: Use the GPT3 model with a temperature of 0 for text generation
## To generate a response, we first retrieve the top-k (e.g., top-3) chunks most similar to the user query,
## format the prompt, and send the formatted prompt to the GPT3 model with a temperature of 0.

query = "how to check disk usage in linux?"
docs = db.similarity_search(query)
retrieved_chunks = [d.page_content for d in docs]

# format the prompt
chunks_formatted = "\n\n".join(retrieved_chunks)
prompt_formatted = prompt.format(chunks_formatted=chunks_formatted, query=query)

# generate answer
llm = OpenAI(model="gpt-4o-mini", temperature=0)

# format the prompt
final_prompt = prompt.format(chunks_formatted=chunks_formatted, query=query)
print("Final prompt:")
print(final_prompt)