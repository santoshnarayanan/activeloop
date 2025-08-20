from dotenv import load_dotenv
import os

load_dotenv()

from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate

# create examples
examples = [
    {
        "query": "What's the weather like?",
        "answer": "It's raining cats and dogs, better bring an umbrella!"
    }, {
        "query": "How old are you?",
        "answer": "Age is just a number, but I'm timeless."
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

# now break our previous prompt into a prefix and suffix
# the prefix is our instructions
prefix = """The following are excerpts from conversations with an AI
assistant. The assistant is known for its humor and wit, providing
entertaining and amusing responses to users' questions. Here are some
examples:
"""
# and the suffix our user input and output indicator
suffix = """
User: {query}
AI: """

# now create the few-shot prompt template
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"]
)

# After creating a template, we pass the example and user query, and we get the results
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

# load model
chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# create the chain
chain = few_shot_prompt | chat
res = chain.invoke({"query": "What's the meaning of life?"})  # pass a dict
print(res.content)