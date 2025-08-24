import os
import json
import argparse
import torch 
from PIL import Image
import re
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm
from utils import convert_page_snapshot_to_image, concat_images
import time 


def load_vlm_model(model_name, device):
    if "QwenVL" in model_name:
        from VLMModels.Qwen_VL import init_model, get_response_concat
    elif "DeepSeek-VL" in model_name:
        from VLMModels.DeepSeek_VL import init_model, get_response_concat
    elif "LLaVA-Next" in model_name:
        from VLMModels.LLaVA_Next import init_model, get_response_concat
    elif model_name == "LLaMA-VL-11B":
        from VLMModels.LLaMA_VL import init_model, get_response_concat
    else:
        raise NotImplementedError 
    
    model = init_model(model_name, device)
    return model, get_response_concat
    

def main_lvlm_QA(args):
    st_time = time.time()
    if os.path.exists(output_path):
        print(f"Loading samples from {output_path}...")
        samples = json.load(open(output_path, "r"))
    else:
        if args.retriever != "None" and os.path.exists(f"./dataset/retrieved/samples_{args.dataset}_{args.retriever}.json"):
            print(f"Loading samples with retrieved pages from {args.retriever}...")
            input_path = f"./dataset/retrieved/samples_{args.dataset}_{args.retriever}.json"
        else:
            input_path = f"./dataset/samples_{args.dataset}.json"
        samples = json.load(open(input_path, "r"))

    model, get_response_concat = load_vlm_model(args.model_name, device)

    for sample in tqdm(samples):
        if args.response_key in sample and sample[args.response_key] != "None":
            continue
        else:
            input_image_list = convert_page_snapshot_to_image(doc_path=f"{document_folder}/{sample['doc_id']}", 
                                                              save_path=img_folder, resolution=args.resolution, max_pages=args.max_pages)
            
            if "pages_ranking" in sample:
                ranked_pages = eval(sample["pages_ranking"])[:args.topk]
                input_image_list = [input_image_list[page-1] for page in ranked_pages]   

            if args.concat_num > 0:
                # TODO: check concat_num 
                #  for certain vlms, the `concat_num` should be set to 1
                #  add the `name_suffix` to distinguish the concat images 
                name_suffix = "concat" if args.retriever == "None" else f"{args.retriever}{'' if args.topk == 5 else 'top'+str(args.topk)}-concat"
                input_image_list = concat_images(image_list=input_image_list, concat_num=args.concat_num, name_suffix=name_suffix) 
            
            try:
                query_prompt = f"Based on the document, please answer the question: {sample['question']}"
                response = get_response_concat(model, query_prompt, input_image_list, max_new_tokens=args.max_tokens, temperature=args.temperature)
            except Exception as e:
                print(f"[ERROR] VLM prediction: {e}")
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                response = "None"

            sample[args.response_key] = response 
     
        with open(output_path, 'w') as file:
            json.dump(samples, file, indent=4, sort_keys=True)
        
    print(f"Dataset-{args.dataset} VLM-{args.model_name} Retriever-{args.retriever} Top-{args.topk}")
    print(f"Cost time: {(time.time() - st_time)/60:.2f} Mins; Avg per sample: {(time.time() - st_time)/len(samples):.3f} Secs\n")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MMLong", choices=["MMLong", "LongDocURL", "PaperTab", "FetaTab"])
    parser.add_argument("--model_name", type=str, default="QwenVL-3B", choices=["QwenVL-3B", 
                                                                                "QwenVL-7B", 
                                                                                "DeepSeek-VL-tiny", 
                                                                                "DeepSeek-VL-small", 
                                                                                "LLaVA-Next-7B", 
                                                                                "LLaVA-Next-8B", 
                                                                                "LLaMA-VL-11B"])
    parser.add_argument("--max_pages", type=int, default=50)
    parser.add_argument("--resolution", type=int, default=144)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concat_num", type=int, default=0)
    parser.add_argument("--retriever", type=str, default="None", choices=["None",
                                                                          "base",
                                                                          "beamsearch",
                                                                          "beamsearch_LoRA"])
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--response_key", type=str, default="raw_response")
    args = parser.parse_args()
    
    if isinstance(args.device, str) and "," in args.device:
        # e.g., cuda:0,cuda:1
        gpu_ids = [x.replace("cuda:", "") for x in args.device.split(",")]
        import os 
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids) 
        device = "auto"
    else:
        device = torch.device(args.device)
    
    args.max_pages = 1000 if args.retriever != "None" else args.max_pages
    print(args)

    document_folder, img_folder, result_folder = f"./dataset/{args.dataset}", f"./tmp/tmp_imgs/{args.dataset}", f"./results/{args.dataset}/{args.model_name}"
    os.makedirs(result_folder, exist_ok=True)

    retrieve_suffix = "Direct" if args.retriever == "None" else f"{args.retriever}_top{args.topk}"
    output_path = f"{result_folder}/{retrieve_suffix}.json"
    
    main_lvlm_QA(args)
