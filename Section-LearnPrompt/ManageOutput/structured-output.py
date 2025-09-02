from dotenv import load_dotenv
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

response_schemas = [
    ResponseSchema(name="words", description="A substitue word based on context"),
    ResponseSchema(name="reasons", description="the reasoning of why this word fits the context.")
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = parser.get_format_instructions()

# --- Prompt ---
template = (
    "Offer a list of suggestions to substitute the word '{target_word}' based on the following text:\n"
    "{context}\n\n"
    "{format_instructions}"
)

prompt = PromptTemplate(
    input_variables=["target_word", "context"],
    template=template,
    partial_variables={"format_instructions": format_instructions},
)

model_input = prompt.format_prompt(
    target_word="behaviour",
    context="The behaviour of the students in the classroom was disruptive and made it difficult for the teacher to conduct the lesson.",
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# Use .invoke() with a string; ChatOpenAI returns an object with .content
raw = llm.invoke(model_input.to_string())
parsed = parser.parse(raw.content)

print("RAW OUTPUT:\n", raw.content, "\n")
print("PARSED:\n", parsed)

# Optional: light validation that lengths match if model followed instructions
words = parsed.get("words", [])
reasons = parsed.get("reasons", [])
if isinstance(words, list) and isinstance(reasons, list) and len(words) != len(reasons):
    print("\nNote: 'words' and 'reasons' lengths differ. You can re-prompt or post-process as needed.")