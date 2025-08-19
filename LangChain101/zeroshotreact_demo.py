import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain_community.tools.google_search.tool import GoogleSearchAPIWrapper
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Set up LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

# Define a simple summarization prompt
summarize_prompt = PromptTemplate.from_template("Summarize the following text:\n{text}")

# Create the summarization chain
summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt)

# Set up Google Search tool
search = GoogleSearchAPIWrapper()
search_tool = [
    Tool(
    name="Google Search",
    func=search.run,
    description="Useful for answering questions about current events or recent information."
),
    Tool(
        name="Summarizer",
        func=summarize_chain.run,
        description="Useful for summarizing texts."
    )
    ]

# Pull the ReAct prompt
react_prompt = hub.pull("hwchase17/react")

# Create the agent
agent = create_react_agent(llm=llm, tools=search_tool, prompt=react_prompt)
agent_executor = AgentExecutor(agent=agent, tools=search_tool, verbose=True)

# Run the agent
resp = agent_executor.invoke({"input": "What is the latest news about the Mars rover?"})
print(resp["output"])