# LLM+RAG Baselines 


## 🔧 Environment 

```
torch==2.5.1
vllm==0.6.6.post1
langchain==0.3.19
langchain-community==0.3.18
langchain-core==0.3.37
langchain-text-splitters==0.3.6
openai==1.71.0
```

## 🚀 Run

Step 0 - API Key

Prepare api keys in both `rag.py` (Qwen-RAG), `apis.py` (LLM API), and `main.py` (Qwen-RAG)


Step 1 - RAG 

```shell 
python3 rag.py --dataset=MMLong 
```

Step 2 - LLM Prediction with RAG 

```shell
python3 main.py --dataset=MMLong --llm_name=deepseek-chat --retrieve_topk=3
```
