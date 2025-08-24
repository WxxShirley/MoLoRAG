import os 
from openai import OpenAI
import base64 
import requests
import argparse
import random
import json 


def generate_prompt(relevance_score, focus):
    prompt = """
    # GOAL # You are an intelligent retriever.
    Given the input image, your task is to generate a question related to it. The relevance score is {relevance_score}, where a higher score indicates a closer connection between the question and the image. For example, a relevance score of 5 means the answer is DIRECTLY contained in the image, while a score below 3 indicates that the answer CANNOT be derived from it, with lower scores signifying less relevance.

    # REQUIREMENT # The question must be based on the content of the input image, except when the relevance score is <= 2. For scores >= 4, focus on specific content, but when the relevance score is less than 4, generate questions that may require inference or are less direct.
    - For relevance scores of 4 or higher, create clear and straightforward questions with answers that are explicitly present in the image.
    - For relevance scores of 3, generate questions that may require some inference but are still somewhat related to the content.
    - For relevance scores of 2 or lower, formulate questions that are unanswerable based on the snapshot. 

    Aim for simplicity and clarity in both the question and answer. The answer should remain concise, ideally one sentence or phrase.

    You may consider various elements, including text, layout, and figures. For this generation, please concentrate on {focus} if applicable and remember that the relevance score is {relevance_score}.

    Your output should be formatted as follows:
    { "query": "Your generated question", "relevance_score": "relevance score", "answer": "Corresponding answer or inference" }
    """
    prompt = prompt.replace("{relevance_score}", str(relevance_score))
    prompt = prompt.replace("{focus}", focus)
    return prompt.strip()


def generate_relevance_prompt(cur_query: str):
    prompt = """# GOAL #\nYou are an Retrieval Expert, and your task is to evaluate how relevant the input document page is to the given query.""" \
             """Rate the relevance on a scale of 1 to 5, where:\n""" \
             """- 5: Highly relevant - contains complete information needed to answer the query\n""" \
             """- 4: Very relevant - contains most of the information needed\n""" \
             """- 3: Moderately relevant - contains some useful information\n""" \
             """- 2: Slightly relevant - has minor connection to the query\n""" \
             """- 1: Irrelevant - contains no information related to the query\n""" \
             """# INSTRUCTION #\nPlease first read the given query, think about what knowledge is required to answer that query, and then carefully go through the document snapshot for judgment.\n""" \
             """# QUERY #\n""" + cur_query + "\nPlease generate just a single number (1-5) representing your relevance judgment. Your answer should be a single number without any extra contents. \n"

    return prompt 


def generate_relevance_prompt_detailed(cur_query: str):
    prompt = """# GOAL #\nYou are an Retrieval Expert, and your task is to evaluate how relevant the input document page is to the given query.""" \
             """Rate the relevance on a scale of 1 to 5, where:\n""" \
             """- 5: Highly relevant - contains COMPLETE information to fully answer the query (be cautious with this rating)\n""" \
             """- 4: Very relevant - contains most information needed but may lack some details (be cautious with this rating)\n""" \
             """- 3: Moderately relevant - contains some useful information but significant gaps remain\n""" \
             """- 2: Slightly relevant - has minor connection to the query\n""" \
             """- 1: Irrelevant - contains no information related to the query\n""" \
             """# INSTRUCTION #\nPlease first read the given query, think about what specific information is required to answer that query comprehensively, and then carefully examine the document snapshot.\n""" \
             """IMPORTANT: Before giving a score of 4 or 5, verify that the page actually contains the specific facts needed to answer the query, not just related information.\n""" \
             """# QUERY #\n""" + cur_query + "\n\nThink step by step about the relevance, then provide just a single number (1-5) representing your judgment.\n"
    return prompt


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string


def gpt_vlm_api_call(prompt, img_path, model="gpt-4o-2024-11-20"):
    encoded_image = encode_image_to_base64(img_path)

    # TODO: set your OpenAI API key here
    api_key = "EMPTY"
    url = "https://api.deerapi.com/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": model, 
        "messages": [
            {
                "role": "user", 
                "content": [
                    { "type": "text", "text": prompt},
                    { "type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}},
                ]
            }
        ],
        "max_tokens": 1024,
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        resp = response.json()
        prediction = resp["choices"][0]["message"]["content"]
        return prediction
    except Exception as e: 
        print(f"Error: {e}")
        print(response)
        # print(response.text)
        return None


def sample_document_page(datasets=["MMLong", "LongDocURL"]):
    dataset2images = {}
    for dataset_name in datasets: 
        img_folder = f"../tmp/tmp_imgs/{dataset_name}" 
        if not os.path.exists(img_folder):
            continue
    
        target_imgs = [f"{img_folder}/{file}" for file in os.listdir(img_folder) if file.endswith(".png")]
        dataset2images[dataset_name] = target_imgs
        print(f"{dataset_name} # Images {len(target_imgs)}")
    
    return dataset2images

    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    # consistently use gpt-4o for data generation
    parser.add_argument("--vlm", type=str, default="gpt-4o-2024-11-20")
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--datasets", type=list, default=["MMLong", "LongDocURL"])
    # distribution for relevance scores from 1 to 5
    parser.add_argument("--score_dist", type=list, default=[0.25, 0.25, 0.15, 0.15, 0.2])
    parser.add_argument("--focus_dist", type=list, default=["text", "layout", "figure", "table", "text and layout", "text and figure", "text and table"])
    parser.add_argument("--write_file", type=str, default="train_samples/qa_pair_qwenvl.json")
    parser.add_argument("--convert_data", action="store_true", help="Convert data into Conversation format")
    parser.add_argument("--further_check", action="store_true", help="Further check the generated data")
    args = parser.parse_args() 

    all_imgs = sample_document_page(args.datasets)
    
    used_tripets = set()
    offset = 0
    if os.path.exists(args.write_file):
        with open(args.write_file, "r") as read_file:
            for line in read_file.readlines():
                cur_dict = json.loads(line)
                used_tripets.add((cur_dict["image"], cur_dict["focus"], cur_dict["relevance_score"]))
                offset = max(offset, cur_dict["sample_id"]+1)

    write_file = open(args.write_file, "a+")
    for i in range(args.num_samples):
        # sample one image 
        cur_dataset = random.choice(args.datasets)
        target_img = random.choices(all_imgs[cur_dataset])[0]

        relevance_score = random.choices(range(1, 6), weights=args.score_dist)[0]
        cur_focus = random.choices(args.focus_dist, k=1)[0]
        
        if (target_img, cur_focus, relevance_score) in used_tripets:
            print(f"Already used {target_img}, {cur_focus}, {relevance_score}")
            continue

        print(f"Sampled {i} {target_img}, {cur_focus}, {relevance_score}")
        cur_prompt = generate_prompt(relevance_score, cur_focus)
    
        response = gpt_vlm_api_call(cur_prompt, target_img, args.vlm)
     
        if response: 
            if "```json" in response: 
                response = response.split("```json")[-1]
                response = response.split("```")[0]
            try:
                response_dict = json.loads(response)
                print(response_dict, "\n")

                full_content = {
                    "sample_id": i + offset, 
                    "dataset": cur_dataset, 
                    "image": target_img.split("/")[-1], 
                    "focus": cur_focus, 
                    "relevance_score": relevance_score, 
                    "query": response_dict["query"], 
                    "answer": response_dict["answer"]
                }
                write_file.write(json.dumps(full_content) + "\n")
                write_file.flush()
            except Exception as e:
                print(response)
                print(f"Error parsing response: {e}")
                continue
    
    print("Finished writing to file.")
    write_file.close()
    
    checked_data = []
    if args.further_check:
        with open(args.write_file, 'r') as read_file:
            for line in read_file: 
                content = json.loads(line) 
                
                cur_img = f"../tmp/tmp_imgs/{content['dataset']}/{content['image']}" 
                cur_relevance = content["relevance_score"] 

                prompt = generate_relevance_prompt(cur_query=content["query"]) 
                if args.vlm.startswith("qwen"):
                    prompt += "Important: Your output **must** be in **a single number** (e.g., 3) without ANY extra contents." 
                    
                further_response = gpt_vlm_api_call(prompt, cur_img, args.vlm) 
                print(further_response)
                if further_response:
                    # special for Qwen-VL
                    try:
                        checked_relevance = int(further_response.strip()) 
                    except:
                        try:
                            checked_relevance = int(further_response[0].strip())
                        except:
                            checked_relevance = 0

                    # Valid data: further-predicted relevance score is within 1 of the original relevance score
                    if abs(checked_relevance - cur_relevance) <= 1:
                        checked_data.append({"sample_id": content["sample_id"], "dataset": content["dataset"], "image": content["image"], "focus": content["focus"], "relevance_score": checked_relevance, "query": content["query"], "answer": content["answer"]})
                        print(f"Sample {content['sample_id']} Checked Rel {checked_relevance} Initial Rel {cur_relevance} Valid")

                        with open("train_samples/checked_data_qwenvl.json", "a+") as write_file:
                            write_file.write(json.dumps(checked_data[-1]) + "\n") 
        print(f"Checked data: {len(checked_data)}") 

    # Convert data into Conversation format (refer to https://github.com/hiyouga/LLaMA-Factory/blob/main/data/mllm_demo.json)
    if args.convert_data: 
        formatted_messages = []
        path = args.write_file

        for line in open(path, 'r'):
            content = json.loads(line) 
            query = {  "content": "<image>" + generate_relevance_prompt(content["query"]),  "role": "user" }
            response = { "content": str(content["relevance_score"]),  "role": "assistant" }
            images = [f"tmp_imgs/{content['dataset']}/{content['image']}"]

            cur_msg = {
                "messages": [query, response],
                "images": images
            }

            formatted_messages.append(cur_msg)
        
        # TODO: ensure that LLaMA-Factory repo is downloaded (https://github.com/hiyouga/LLaMA-Factory)
        dump_file = open("train_samples/retriever_qwenvl.json", "w")
        json.dump(formatted_messages, dump_file, indent=4)
        print(f"Converted data into {len(formatted_messages)} conversations.")
