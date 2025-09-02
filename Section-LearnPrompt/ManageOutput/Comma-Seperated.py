from dotenv import load_dotenv
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# --- Parser ---
parser = CommaSeparatedListOutputParser()
format_instructions = parser.get_format_instructions()

# --- Prompt ---
template = (
    "Offer a concise list of substitute words for '{target_word}' based on the following text:\n"
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

# --- LLM (chat) ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# Generate & parse
raw = llm.invoke(model_input.to_string())   # raw is a ChatResult-like object
items = parser.parse(raw.content)           # -> List[str]

print("RAW:\n", raw.content, "\n")
print("PARSED LIST:", items)
