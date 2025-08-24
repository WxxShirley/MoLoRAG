import os 
import json 
import argparse
from tqdm import tqdm
from evaluate.eval_qa import extract_answer, extract_score, eval_one_sample, eval_samples, show_fine_grained_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--dataset", type=str, default="MMLong")
    parser.add_argument("--method", type=str, default="VLM", choices=["LLM", "VLM", "MDocAgent"])
    parser.add_argument("--model_name", type=str, default="QwenVL-7B", choices=["qwen-7b", 
                                                                                "mistral-7b", 
                                                                                "llama-8b", 
                                                                                "deepseek-chat", 
                                                                                "gpt-4o-mini",
                                                                                "QwenVL-7B", 
                                                                                "QwenVL-3B",
                                                                                "DeepSeek-VL-tiny", 
                                                                                "DeepSeek-VL-small", 
                                                                                "LLaVA-Next-7B", 
                                                                                "LLaVA-Next-8B", 
                                                                                "LLaMA-VL-11B"])
    
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--retriever", type=str, default="None", choices=["None", "base", "beamsearch", "beamsearch_LoRA"])
    parser.add_argument("--folder_eval", action="store_true", default=False)
    parser.add_argument("--save_freq", type=int, default=50)
    args = parser.parse_args()
    
    if args.method == "LLM":
        pred_folder = f"./results/{args.dataset}/LLM"
        pred_paths = [f"./results/{args.dataset}/LLM/{args.model_name}_top{args.topk}.json"]
    elif args.method == "VLM":
        pred_folder = f"./results/{args.dataset}/{args.model_name}"
        if args.retriever != "None":
            pred_paths = [f"./results/{args.dataset}/{args.model_name}/{args.retriever}_top{args.topk}.json"]
        else:
            pred_paths = [f"./results/{args.dataset}/{args.model_name}/Direct.json"]
    elif args.method == "MDocAgent": 
        pred_folder = f"./results/{args.dataset}/mdocagent"
        # TODO: fix time 
        pred_paths = [f"./results/{args.dataset}/mdocagent/_.json"]
        args.folder_eval = True
    
    if args.folder_eval: 
        # specially for MDocAgent
        pred_paths = sorted([f"{pred_folder}/{file}" for file in os.listdir(pred_folder) if file.endswith(".json")])
        pred_paths = [path for path in pred_paths if "_scored" not in path]
        
    with open("./evaluate/prompt_for_answer_extraction.md", 'r') as f:
        extractor_prompt = f.read()
    with open(f"./evaluate/prompt_for_scoring.md", 'r') as file:
        scoring_prompt = file.read()
    
    for cur_pred_path in pred_paths:
        if not os.path.exists(cur_pred_path):
            print(f"Prediction {cur_pred_path} does not exist.")
            continue
        print(cur_pred_path)
        
        raw_samples, scored_samples, scored_ids = json.load(open(cur_pred_path, 'r')), [], set()
        rewrite_pred_path = cur_pred_path.replace(".json", "_scored.json")
        if os.path.exists(rewrite_pred_path): 
            scored_samples = json.load(open(rewrite_pred_path, 'r')) 
            scored_ids = set([sample["id"] for sample in scored_samples])
            
        # Step 1 GPT-4o extract answer / assign scores
        for sample in tqdm(raw_samples):
            if sample["id"] in scored_ids:
                continue 

            if args.dataset in ["MMLong", "LongDocURL"]: # for MMLong and LongDocURL, extract answer
                if "pred_ans" not in sample:
                    # extract answer from raw_response 
                    extracted_ans = extract_answer(question=sample["question"], 
                                                  output=sample.get("raw_response", None),
                                                  extractor_prompt=extractor_prompt)
                    
                    try:
                        pred_ans = extracted_ans.split("Answer format:")[0].split("Extracted answer:")[1].strip()
                        sample["pred_ans"] = pred_ans
                    except:
                        sample["pred_ans"] = "Failed to extract"
                
                em_score, acc_score = eval_one_sample(gt=sample["answer"], 
                                                      pred=sample["pred_ans"], 
                                                      answer_type=sample["answer_format"])
                sample["score"] = {"EM": em_score, "Acc": acc_score}
                
            elif args.dataset in ["PaperTab", "FetaTab"]: # for PaperTab and FetaTab, assign score
                if "score" not in sample: 
                    generated_score = extract_score(question=sample["question"], 
                                                    output=sample.get("raw_response", None),
                                                    ground_truth=sample["answer"],
                                                    prompt=scoring_prompt)
                    score = generated_score.get('binary_correctness', 0) 

                    sample["score"] = {"BinaryCorrectness": score}
            scored_samples.append(sample)  

            if len(scored_samples) % args.save_freq == 0: 
                with open(rewrite_pred_path, 'w') as file:
                    json.dump(scored_samples, file, indent=4, sort_keys=True)
        
        try:
            assert len(scored_samples) == len(raw_samples) 
        except Exception as e:
            print(f"[ERROR] Scored samples {len(scored_samples)} do not match raw samples {len(raw_samples)}. ")
            
        # Step 2 Save scored results 
        with open(rewrite_pred_path, 'w') as file:
            json.dump(scored_samples, file, indent=4, sort_keys=True)

        # Step 3 Evaluate results    
        score_dict = eval_samples(scored_samples, args.dataset)
        print(f"Dataset-{args.dataset} Method-{args.method} Model-{args.model_name} Retriever-{args.retriever} Top-{args.topk}")

        # Step 4 Show fine-grained results 
        if args.dataset in ["MMLong", "LongDocURL"]: 
            show_fine_grained_results(scored_samples, args.dataset)
        else:
            print(f"Evalution-{score_dict}")
        
        print(f"Evaluation finished!\n\n")
