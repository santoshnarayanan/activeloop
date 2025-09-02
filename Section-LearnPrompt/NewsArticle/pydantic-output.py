from dotenv import load_dotenv

from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser  # optional but recommended
from pydantic import BaseModel, Field, field_validator
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from typing import List

load_dotenv()

# ---- Pydantic schema ----
class ArticleSummary(BaseModel):
    title: str = Field(description="Title of the article")
    summary: List[str] = Field(description="Bulleted list summary of the article")

    @field_validator("summary")
    def has_three_or_more_lines(cls, v: List[str]):
        if len(v) < 3:
            raise ValueError("Generated summary has less than three bullet points!")
        return v

# Base parser
base_parser = PydanticOutputParser(pydantic_object=ArticleSummary)
# Optional: wraps the base parser to auto-fix slightly invalid JSON
parser = OutputFixingParser.from_llm(parser=base_parser, llm=ChatOpenAI(model="gpt-4o-mini", temperature=0))

# ---- Prompt ----
template = """
You are a very good assistant that summarizes online articles.

Here's the article you want to summarize.

==================
Title: {article_title}

{article_text}
==================

{format_instructions}
"""

prompt = PromptTemplate(
    input_variables=["article_title", "article_text"],
    template=template,
    partial_variables={"format_instructions": base_parser.get_format_instructions()},
)

formatted = prompt.format_prompt(
    article_title="The behaviour of the students in the classroom.",
    article_text="The behaviour of the students in the classroom was disruptive and made it difficult for the teacher to conduct the lesson.",
)

# ---- LLM ----
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# IMPORTANT: use .invoke() with a STRING (or pass messages). Avoid __call__.
raw = llm.invoke(formatted.to_string())

# Parse the model output directly (no manual slicing)
parsed: ArticleSummary = parser.parse(raw.content)
print("From LLM (parsed):", parsed)
print("Title:", parsed.title)
print("Summary bullets:", parsed.summary)

# ---- Manual instantiation (uses same validation) ----
manual = ArticleSummary(
    title="Meta claims its new AI supercomputer will set records",
    summary=[
        "Meta (formerly Facebook) has unveiled an AI supercomputer that it claims will be the world’s fastest.",
        "The supercomputer is called the AI Research SuperCluster (RSC) and is yet to be fully complete.",
        "Meta says it will train models with trillions of parameters once complete.",
        "For production, Meta expects RSC will be 20x faster than its current V100-based clusters.",
        "Previous infra largely used public/open datasets; RSC will use real platform data for research like harmful content detection."
    ],
)
print("Manual (parsed):", manual)
