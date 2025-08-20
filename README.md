# Project Overview

This project uses the latest packages and libraries, so the code may differ from the **LangChain101** page.

## Instructions
- Install required packages using the `requirements.txt` file.

## Folder Structure
1. `LangChain101` - Contains code from the LangChain101.
2. `LLMAndLangChain` - Coe from  -- Quick Intro to LLM.

---

## Steps Taken -- Quick Intro to LLM Issue with Hugging Face
### filename - **question-ans-prompt_demo.py**

### 1. Started with LangChain + HuggingFaceHub (`HuggingFaceEndpoint`)
- Tried using `google/flan-t5-base` with `HuggingFaceEndpoint`.
- Ran into issues because `flan-t5-base` is a **seq2seq model**, but `HuggingFaceEndpoint` routed through the wrong API (`text-generation` instead of `text2text-generation`).
- Errors seen: `StopIteration`, `doesn't support task`.

---

### 2. Switched to Local Inference with Hugging Face Transformers
- Installed and used `transformers` to load **`flan-t5-base`** locally.
- First run downloaded tokenizer + model weights (~990MB).
- Subsequent runs reused cached weights.

---

### 3. Observed Output Variability
- Sometimes the model answered **“Paris”**, other times **“London”**.
- This variability is expected — small T5 models can be unstable without decoding constraints.

---

### 4. Stabilized the Output
- Applied decoding strategies:
  - `do_sample=False`, `temperature=0.0`
  - Beam search: `num_beams=5`
  - Prevent repetition: `no_repeat_ngram_size=2`
  - Output limit: `max_new_tokens=8`
- Adjusted prompt to explicitly request the **capital city only**.
- Result: The answer became consistently **“Paris”** ✅.

---

### 5. Integration with LangChain
- Wrapped the Hugging Face pipeline inside **`HuggingFacePipeline`**.
- Combined with a **`PromptTemplate`** in LangChain:  
  ```python
  chain = prompt | llm
