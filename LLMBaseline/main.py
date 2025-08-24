import os
from langchain_community.vectorstores import FAISS
import json
import argparse
from tqdm import tqdm
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.prompts import ChatPromptTemplate
from apis import invoke_llm_api
import time


def retrieve_context(query, index_folder, top_k=5, max_length=1600):
    if not os.path.exists(index_folder):
        print(f"[ERROR] Index folder {index_folder} does not exist")
        return "" 
    
    # TODO: Set your DashScope API key here
    embeddings = DashScopeEmbeddings(model="text-embedding-v1", 
                                     dashscope_api_key="YOUR Key HERE")
    faiss_index = FAISS.load_local(index_folder, embeddings, allow_dangerous_deserialization=True)

    searched_docs = faiss_index.similarity_search(query, k=top_k)
    context = []
    for doc in searched_docs:
        content = doc.page_content[:max_length]
        content = content.replace('\n', ' ').replace('\r', '')  
        content = ' '.join(content.split())  
        content = content.strip() 
        context.append(content)
    
    # print(query, context, "\n")
    return " ".join(context)


def main_llm_QA(args):
    st_time = time.time()
    if os.path.exists(output_file):
        # resume from the existing output file
        with open(output_file, 'r') as file:
            samples = json.load(file)
    else:
        input_file = f"../dataset/samples_{args.dataset}.json"
        with open(input_file, 'r') as file:
            samples = json.load(file)
    
    pred_num = 0
    for sample in tqdm(samples):
        if args.response_key in sample:
            continue 
        else:
            index_folder = f"../tmp/tmp_dbs/{args.dataset}/{sample['doc_id'].replace('.pdf', '')}"

            if args.retrieve_topk:
                context_txt = retrieve_context(query=sample["question"],
                                               index_folder=index_folder,
                                               top_k=args.retrieve_topk)
                QUERY_PROMPT = """
                Answer the question based on the following context:
                {context}
                ---
                Answer the question based on the above context: {question}
                """
                prompt_template = ChatPromptTemplate.from_template(QUERY_PROMPT)
                q_prompt = prompt_template.format(context=context_txt, question=sample["question"])
            else:
                QUERY_PROMPT = """Directly answer the question : {question}"""
                prompt_template = ChatPromptTemplate.from_template(QUERY_PROMPT)
                q_prompt = prompt_template.format(question=sample["question"])
            
            try:
                response = invoke_llm_api(model_name=args.llm_name, content=q_prompt)
            except Exception as e:
                print(f"[ERROR] invoking LLM API: {e}")
                response = "None"
            
            sample[args.response_key] = response
        
        pred_num += 1 
        if (pred_num % args.save_freq == 0) or (pred_num == len(samples)): 
            with open(output_file, 'w') as file:
                json.dump(samples, file, indent=4, sort_keys=True)

    print(f"Dataset-{args.dataset} LLM-{args.llm_name} Top-{args.retrieve_topk}")
    print(f"Cost time: {(time.time() - st_time)/60:.2f} Mins\n\n")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MMLong")
    parser.add_argument("--llm_name", type=str, default="deepseek-chat", choices=["qwen-7b", "mistral-7b", "llama-8b", "deepseek-chat", "gpt-4o-mini"])
    parser.add_argument("--retrieve_topk", type=int, default=5)
    parser.add_argument("--response_key", type=str, default="raw_response")
    parser.add_argument("--save_freq", type=int, default=10)
    args = parser.parse_args()

    # folder-format {dataset}/{method_name}/{llm_name}_{topk}
    output_file = f"../results/{args.dataset}/LLM/{args.llm_name}_top{args.retrieve_topk}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    main_llm_QA(args)
