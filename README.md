## Project Overview

This project uses the latest packages and libraries, so the code may differ from the LangChain101 page.

**Instructions:**
- Install required packages using the `requirements.txt` file.

**Folder Structure:**
1. `LangChain101` - Contains code from the LangChain101 course.


Note :-
HuggingFaceEndpoint (the LLM wrapper in langchain-huggingface) currently routes calls through the text-generation 
endpoint under the hood. FLAN-T5 is a seq2seq model that expects the text2text-generation task. 
So even though you set task="text2text-generation", the class still ends up calling the 
text-generation route → HF rejects it: