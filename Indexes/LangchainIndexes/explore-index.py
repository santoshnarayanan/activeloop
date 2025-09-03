from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_deeplake import DeeplakeVectorStore
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

import deeplake

load_dotenv()

# text to write to a local file
# taken from https://www.theverge.com/2023/3/14/23639313/google-ai-language-model-palm-api-challenge-openai
text = """Google opens up its AI language model PaLM to challenge OpenAI and GPT-3
Google is offering developers access to one of its most advanced AI language models: PaLM.
The search giant is launching an API for PaLM alongside a number of AI enterprise tools
it says will help businesses “generate text, images, code, videos, audio, and more from
simple natural language prompts.”

PaLM is a large language model, or LLM, similar to the GPT series created by OpenAI or
Meta’s LLaMA family of models. Google first announced PaLM in April 2022. Like other LLMs,
PaLM is a flexible system that can potentially carry out all sorts of text generation and
editing tasks. You could train PaLM to be a conversational chatbot like ChatGPT, for
example, or you could use it for tasks like summarizing text or even writing code.
(It’s similar to features Google also announced today for its Workspace apps like Google
Docs and Gmail.)
"""

# write text to a local file
with open("example.txt", "w") as f:
    f.write(text)

# Use TextLoader to load text from the file
loader = TextLoader("example.txt")
doc = loader.load()

print(len(doc))

# -------CharacterTextSplitter to split docs into texts
splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
split_docs = splitter.split_documents(doc)
print(len(split_docs))

# embeddings allow us to effectively search for documents or portions of documents
# that relate to our query by examining their semantic similarities.
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

#--------------------------------------------------------------------------
# We'll employ the Deep Lake vector store with our embeddings in place.
#
# Deep Lake provides several advantages over the typical vector store:
#
#     It’s multimodal, which means that it can be used to store items of diverse modalities,
#     such as texts, images, audio, and video, along with their vector representations.

#     It’s serverless, which means that we can create and manage cloud datasets
#     without the need to create and managing a database instance.
#     This aspect gives a great speedup to new projects.

#     It’s possible to easily create a streaming data loader out of the data loaded
#     into a Deep Lake dataset, which is convenient for fine-tuning machine
#     learning models using common frameworks like PyTorch and TensorFlow.

#     Data can be queried and visualized easily from the web.
# ----------------------------------------------------------------------------------

# create Deep Lake dataset
# use your organization id here. (by default, org id is your username)
my_activeloop_org_id = "santosh"
my_activeloop_dataset_name = "langchain_course_indexers_retrievers"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
# Create DeepLake vector store (new way)
db = DeeplakeVectorStore(dataset_path=dataset_path, embedding_function=embeddings)

# add documents to our DeepLake dataset
db.add_documents(split_docs)

# crete retriever from db
retriever = db.as_retriever()

# create retrieval chain
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

# query our document
resp = chain.invoke({"query": "How Google plans to challenge OpenAI?"})
print(resp)
