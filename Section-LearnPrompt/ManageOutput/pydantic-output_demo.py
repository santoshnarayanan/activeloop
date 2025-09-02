from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from typing import List

load_dotenv()

class Suggestions(BaseModel):
    words: List[str] = Field(description="list of substitue words based on context")

    @field_validator("words")
    def not_start_with_number(cls, v):
        for item in v:
            if item and item[0].isnumeric():
                raise ValueError("Words should not start with a number")
        return v

parser = PydanticOutputParser(pydantic_object=Suggestions)

template = """
Offer a list of suggestions to substitue the specified target_word based the presented context.
{format_instructions}
target_word={target_word}
context={context}
"""

prompt = PromptTemplate(
    input_variables=["target_word", "context"],
    template=template,
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

model_input = prompt.format_prompt(
    target_word="behaviour",
    context="The behaviour of the students in the classroom was disruptive and made it difficult for the teacher to conduct the lesson.",
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# Use .invoke() with a string
output = llm.invoke(model_input.to_string())

# Parse to your Pydantic model
parsed = parser.parse(output.content)
print("From LLM:", parsed)

# Manual instantiation (same validation)
manual = Suggestions(words=["conduct", "manner", "action", "demeanor", "attitude", "activity"])
print("Manual:", manual)
print("Manual.words:", manual.words)
