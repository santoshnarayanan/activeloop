from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# create our examples
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
example_template = """User: {query}
AI: {answer}"""

# create Prompt Template from above example
example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

# now break our previous prompt into a prefix and suffix
prefix = """The following are excerpts from conversations with an AI
assistant. The assistant is known for its humor and wit, providing
entertaining and amusing responses to users' questions. Here are some
examples:
"""

suffix = """\n\nUser: {query}\nAI: """

# now create the few-shot prompt template
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n",
)

chat  = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
chain = few_shot_prompt | chat

# After creating a template, we pass the example and user query, and we get the results
question = "What's the secret of happiness?"
response = chain.invoke({"query": question})

print("Question:", question)
print("Answer:", response.content)
