import json
import requests
from openai import OpenAI


OPENAI_CONFIG = {
    "MODEL": "gpt-4o-mini",
    "ENDPOINT": "https://api.deerapi.com/v1/chat/completions",
    "API_KEY": "",
}


DEEPSEEK_CONFIG = {
    "ENDPOINT": "https://api.deepseek.com", 
    "API_KEY": ""
}


QWEN_CONFIG = {
    "ENDPOINT": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "API_KEY": ""
}


def invoke_llm_api(model_name, content):
    if model_name == "gpt-4o-mini": 
        prediction = invoke_gpt4o_api(content)
    elif model_name == "deepseek-chat":
        prediction = invoke_deepseek_api(content)
    elif model_name == "qwen-7b":
        prediction = invoke_qwen_api(model_name="qwen2.5-7b-instruct", content=content)
    elif model_name in ["mistral-7b", "llama-8b"]:
        prediction = invoke_opensource_llm(model_name, content=content)
    else:
        raise "Unsupported LLM"
    
    return prediction


def invoke_gpt4o_api(content, temperature=0.2, max_retries=3): 
    headers = {
        "Content-Type": "application/json", 
        "Authorization": f"Bearer {OPENAI_CONFIG['API_KEY']}",
    }
    
    payload = {
        "model": OPENAI_CONFIG["MODEL"],
        "messages": [
            {
                "role": "system", 
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": [ { "type": "text", "text": content }],
            }
        ],
        "max_tokens": 256,
        "temperature": temperature,
    }
    
    retry_cnt = 0
    while retry_cnt < max_retries: 
        try:
            response = requests.post(OPENAI_CONFIG["ENDPOINT"], headers=headers, json=payload, timeout=500) # proxys=proxies
            return response.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f'[ERROR] OpenAI API error: {e}')
            retry_cnt += 1 
            if retry_cnt >= max_retries:
                raise
            print(f'[INFO] Retrying {retry_cnt}/{max_retries}...')
            import time  
            time.sleep(2)


def invoke_deepseek_api(content):
    client = OpenAI(api_key=DEEPSEEK_CONFIG["API_KEY"], base_url=DEEPSEEK_CONFIG["ENDPOINT"])
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content}
        ],
        stream=False
    )
    prediction = response.choices[0].message.content
    return prediction


def invoke_qwen_api(model_name="qwen2.5-7b-instruct", content=""):
    client = OpenAI(api_key=QWEN_CONFIG["API_KEY"], base_url=QWEN_CONFIG["ENDPOINT"])
    response = client.chat.completions.create(
        model=model_name, 
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content}
        ],
        temperature=0.0, 
        max_tokens=256, 
        top_p=0.1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response.choices[0].message.content.strip() 


def invoke_opensource_llm(model_name, content):
    headers = { "Content-Type": 'application/json'}
    url = f"http://127.0.0.1:8008/v1/chat/completions"

    model_mapping = {
        "qwen-7b": "Qwen2.5-7B-Instruct",
        "mistral-7b": "Mistral-7B-Instruct-v0.2", 
        "llama-8b": "Llama-3.1-8B-Instruct"
    }

    payload = json.dumps({
        "model": model_mapping[model_name],
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "temperature": 0.2,
        "top_p": 0.1,
        "frequency_penalty": 0,
        "presence_penalty": 1.05,
        "max_tokens": 8192 * 2,
        "stream": False,
        "stop": None
    })
    
    response = requests.post(url, headers=headers, data=payload, timeout=500)
    
    resp = response.json()
    prediction = resp["choices"][0]["message"]["content"]
    return  prediction
