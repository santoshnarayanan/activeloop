from dotenv import load_dotenv
from langchain.prompts import FewShotPromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_core.prompts import PromptTemplate
from langchain_deeplake import DeeplakeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

example_template = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)

examples = [
    {"input": "0°C", "output": "32°F"},
    {"input": "10°C", "output": "50°F"},
    {"input": "20°C", "output": "68°F"},
    {"input": "30°C", "output": "86°F"},
    {"input": "40°C", "output": "104°F"},
]

# Use Hub path (requires ACTIVLOOP_TOKEN) or switch to a local path if you prefer:
my_activeloop_org_id = "santosh"
my_activeloop_dataset_name = "langchain_course_fewshot_selector"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
# For local instead, uncomment:
# dataset_path = "./deeplake_fewshot_selector"
# if os.path.exists(dataset_path): shutil.rmtree(dataset_path)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# --- Create the vector store (handle param name differences across versions)
try:
    db = DeeplakeVectorStore(dataset_path=dataset_path, embedding_function=embeddings)
except TypeError:
    db = DeeplakeVectorStore(dataset_path=dataset_path, embedding_function=embeddings)

# --- ADD DATA: write the few-shot examples into the dataset ---
texts = [eg["input"] for eg in examples]
metadatas = [{"label": "fewshot", "output": eg["output"]} for eg in examples]
ids = [f"eg-{i}" for i in range(len(examples))]  # optional but handy

# This triggers OpenAI embedding calls and stores vectors+metadata in Deep Lake
db.add_texts(texts=texts, metadatas=metadatas, ids=ids)

# (Optional) sanity-check retrieval
hits = db.similarity_search("35°C", k=2)
print("Similarity search results (top 2):")
for i, doc in enumerate(hits, 1):
    print(f"{i}. text={doc.page_content!r}, meta={doc.metadata}")

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=examples,
    embeddings=embeddings,
    vectorstore_cls=DeeplakeVectorStore,
    k=1,
    input_keys=["input"],  # MUST match both examples AND runtime kwargs
    vectorstore_kwargs={"dataset_path": dataset_path},
)

prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_template,
    prefix="Convert the temperature from Celsius to Fahrenheit.",
    suffix="\nInput: {input}\nOutput:",
    input_variables=["input"],  # match runtime kwargs
)

print(prompt.invoke({"input": "10°C"}))
print()
print(prompt.invoke({"input": "30°C"}))
print()

prompt.example_selector.add_example({"input": "50°C", "output": "122°F"})
print(prompt.invoke({"input": "40°C"}))
