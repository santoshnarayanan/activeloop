from dotenv import load_dotenv
from langchain.prompts.example_selector import LengthBasedExampleSelector
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
load_dotenv()
examples = [
    {"word": "happy", "antonym": "sad"},
    {"word": "tall", "antonym": "short"},
    {"word": "energetic", "antonym": "lethargic"},
    {"word": "sunny", "antonym": "gloomy"},
    {"word": "windy", "antonym": "calm"},
]

example_template = """
Word: {word}
Antonym: {antonym}
"""

example_prompt = PromptTemplate(
    input_variables=["word", "antonym"],
    template=example_template
)

example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=25,
)

dynamic_prompt=  FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input",
    suffix="Word: {input}\nAntonym:",
    input_variables=["input"],
    example_separator="\n\n",
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
chain = dynamic_prompt | llm

# response = chain.invoke({"input": "big"})
print(dynamic_prompt.format(input="big"))
# print(response.content)