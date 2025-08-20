# load env variables
from dotenv import load_dotenv
import os

load_dotenv()


from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

template = "You are an assistant that helps users find information about movies."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "Find information about the movie {movie_title}."
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

response =  chat.invoke(chat_prompt.format_prompt(movie_title="The Matrix").to_messages())
print(response.content)

