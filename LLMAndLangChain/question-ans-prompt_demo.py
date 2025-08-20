from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
mdl = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

pipe = pipeline(
    "text2text-generation",
    model=mdl,
    tokenizer=tok,
    max_new_tokens=8,     # short factual answer
    num_beams=5,          # <-- more reliable than sampling
    do_sample=False,      # deterministic
)

prompt = PromptTemplate.from_template(
    "Answer with the capital city only.\nQuestion: What is the capital of France?\nAnswer:"
)

llm = HuggingFacePipeline(pipeline=pipe)
print((prompt | llm).invoke({}))   # expected: "Paris"
