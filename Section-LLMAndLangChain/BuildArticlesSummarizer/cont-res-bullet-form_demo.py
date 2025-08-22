from dotenv import load_dotenv

load_dotenv()

import requests
from newspaper import Article
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
}
article_url = "https://www.artificialintelligence-news.com/2022/01/25/meta-claims-new-ai-supercomputer-will-set-records/"
session = requests.Session()

try:
    response = session.get(article_url, headers=headers, timeout=10)
    if response.status_code == 200:
        article = Article(article_url)
        article.download()
        article.parse()

        # we get the article data from the scraping part
        article_text = article.text
        article_title = article.title

        # prepare template for prompt
        template = """You are an advanced AI assistant that summarizes online articles into bulleted lists in French.

        Here's the article you need to summarize.

        ==================
        Title: {article_title}

        {article_text}
        ==================

        Now, provide a summarized version of the article in a bulleted list format, in French.
        """
        prompt = template.format(article_title=article_title, article_text=article_text)

        # create human message
        message = HumanMessage(content=prompt)

        # create chat model
        chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

        # generate summary
        summary = chat.invoke([message])

        # print summary
        print(summary.content)

    else:
        print(f"Failed to fetch article at {article_url}")
except Exception as e:
    print(f"Error occurred while fetching article at {article_url}: {e}")