from  dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
# Prompt 1
template_question = """What is the name of the famous scientist who developed the theory of general relativity?
Answer: """
prompt_template = PromptTemplate(
    input_variables=[],
    template=template_question
)

# Prompt 2
template_fact = """Provide a brief description of {scientist}'s theory of general relativity.
Answer: """
prompt_template_fact = PromptTemplate(
    input_variables=["scientist"],
    template=template_fact
)

# create Runnable for first prompt
chain_question = prompt_template | llm
response_question = chain_question.invoke({})

# Extract scientist name from response
scientist = response_question.content.split("\n")[0].split(" ")[-1]
# scientist = response_question.content.strip()

# create Runnable for second prompt
chain_fact = prompt_template_fact | llm

# Input data for the second prompt
input_data = {"scientist": scientist}
response_fact = chain_fact.invoke(input_data)


# Extract the scientist's name from the response
# scientist_fact = response_fact.content.split("\n")[0].split(" ")[-1]

print("Scientist:", scientist)
print("Fact:", response_fact.content)





