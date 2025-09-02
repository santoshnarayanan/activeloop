from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

examples = [
    {
        "query": "How do I become a better programmer?",
        "answer": "Try talking to a rubber duck; it works wonders."
    }, {
        "query": "Why is the sky blue?",
        "answer": "It's nature's way of preventing eye strain."
    }
]

example_template = """
User: {query}
AI: {answer}
"""

example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

prefix = """The following are excerpts from conversations with an AI assistant. The assistant is known for its humor and wit, providing entertaining and amusing responses to users' questions. Here are some examples: """

suffix = """\n\nUser: {query}\nAI: """

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n",
)

chain = few_shot_prompt | llm

question = "What is quantum computing?"
response = chain.invoke({"query": question})

print("Question:", question)
print("Answer:", response.content)