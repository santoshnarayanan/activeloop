from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import(
    ChatPromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate,
    HumanMessagePromptTemplate
)

load_dotenv()

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

template="You are a helpful assistant that translates english to pirate."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
example_human = HumanMessagePromptTemplate.from_template("Hi")
example_ai = AIMessagePromptTemplate.from_template("Argh me mateys")
human_template = "{text}"
human_messsage_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, example_human, example_ai, human_messsage_prompt]
)
chain = chat_prompt | chat
res = chain.invoke("I love programming")
print(res)