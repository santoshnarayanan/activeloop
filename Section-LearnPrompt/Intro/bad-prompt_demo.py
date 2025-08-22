from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# create template for bad prompt
template = "Tell me something about {topic}"


# create Prompt Template from above example
example_prompt = PromptTemplate(
    input_variables=["topic"],
    template=template
)

chain = example_prompt | llm | StrOutputParser()
response = chain.invoke({"topic": "dogs"})
print(response)

