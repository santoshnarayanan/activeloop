from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# create examples
examples = [
    {
        "query": "What's the secret to happiness?",
        "answer": "Finding balance in life and learning to enjoy the small moments."
    }, {
        "query": "How can I become more productive?",
        "answer": "Try prioritizing tasks, setting goals, and maintaining a healthy work-life balance."
    }
]

# create a example template
example_template = """
User: {query}
AI: {answer}
"""

# create Prompt Template from above example
example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)


prefix = """The following are excerpts from conversations with an AI
life coach. The assistant provides insightful and practical advice to the users' questions. Here are some
examples: 
"""

suffix = """
User: {query}
AI: """


few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n",
)

chain = few_shot_prompt | llm | StrOutputParser()

# Define the user query
user_query = "What are some tips for improving communication skills?"


response = chain.invoke({"query": user_query})
print("AI Response:", response)

