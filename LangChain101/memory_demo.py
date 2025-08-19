from dotenv import load_dotenv
import os

load_dotenv()

from langchain_community.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.0)
conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory(),
    verbose=True,
)
# Start a conversation
conversation.predict(input="Tell me about yourself.")

#Continue the conversation
conversation.predict(input="What can you do")
conversation.predict(input="How can you help me with data analysis?")

# Display the conversation
print(conversation)