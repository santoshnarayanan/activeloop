from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# create examples
examples = [
    {"color": "red", "emotion": "passion"},
    {"color": "blue", "emotion": "serenity"},
    {"color": "green", "emotion": "tranquility"},
]

# create a example template
example_template = """Color: {color}
Emotion: {emotion}"""

# create Prompt Template from above example
example_prompt = PromptTemplate(
    input_variables=["color", "emotion"],
    template=example_template
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="""The following are examples of color and emotion pairs:""",
    suffix="""\n\nWhat is the emotion associated with the color {color}?""",
    input_variables=["color"],
    example_separator="\n\n",
)

# formatted_prompt = few_shot_prompt.invoke({"color": "purple"})
chain = few_shot_prompt | llm | StrOutputParser()
response = chain.invoke({"color": "purple"})
print(response)

