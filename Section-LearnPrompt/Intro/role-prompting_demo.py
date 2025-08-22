from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

template = """
As a futuristic robot band conductor, I need you to help me come up with a song title.
What's a cool song title for a song about {theme} in the year {year}?
"""
prompt_template = PromptTemplate(
    input_variables=["theme", "year"],
    template=template
)

# Input data for the prompt
input_data = {"theme": "interstellar travel", "year": "3030"}

chain = prompt_template | llm

response = chain.invoke(input_data)

print("Theme: interstellar travel")
print("Year: 3030")
print("AI generated song title:", response)