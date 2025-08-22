
from dotenv import load_dotenv
import os

load_dotenv()

from langchain_community.llms import OpenAI
# Before executing the following code, make sure to have
# your OpenAI key saved in the “OPENAI_API_KEY” environment variable.
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.0)

text = "Suggest a personalized workout routine for someone looking to improve cardiovascular endurance and prefers outdoor activities."
print(llm(text))