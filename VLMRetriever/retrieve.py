import torch 
from tqdm import tqdm 
from colpali_engine.models import ColPali, ColPaliProcessor
import argparse
import sys 
sys.path.append("../")
from utils import load_all_doc_embeddings, construct_page_graph
import json 
from pdf2image import convert_from_path
import os 
import re
from collections import defaultdict
from copy import deepcopy
from data_collection import generate_relevance_prompt as rel_prompt, generate_relevance_prompt_detailed as advanced_rel_prompt


def query_vlm_relevance(query, doc_info, vlm_model):
    doc_id, page_num = doc_info 
    if not os.path.exists(f"../tmp/tmp_imgs/{args.dataset}/{doc_id}-{page_num}.png"):
        doc_snapshot = convert_from_path(pdf_path=f"../dataset/{args.dataset}/{doc_id}.pdf", first_page=page_num, last_page=page_num, dpi=144)[0] # resolution is fixed to 144
        doc_snapshot.save(f"../tmp/tmp_imgs/{args.dataset}/{doc_id}-{page_num}.png", "PNG")
    
    img_path = f"./tmp/tmp_imgs/{args.dataset}/{doc_id}-{page_num}.png"
    prompt = rel_prompt(query) if args.dataset == "MMLong" else advanced_rel_prompt(query)
    
    try:
        response = get_response_concat(vlm_model, prompt, img_path, max_new_tokens=16, temperature=1)
        score_match = re.search(r'[1-5]', response)
        if score_match:
            relevance_score = int(score_match.group(0))
        else:
            relevance_score = 3
    except Exception as e:
        print(f"Error in VLM for {doc_id}-{page_num}; Exception {e}")
        relevance_score = 1
    
    return relevance_score
    

class DocumentRetriever:
    """
    A unified class for document retrieval with multiple retrieval strategies
    """
    def __init__(self, encoder: ColPali, processor: ColPaliProcessor, device: torch.device, batch_size=512):
        self.encoder = encoder 
        self.processor = processor 
        self.device = device 
        self.batch_size = batch_size     

    def compute_scores(self, query, all_embeds):
        queries = self.processor.process_queries(queries=[query]).to(self.device)
        query_embeds = self.encoder(**queries)

        all_scores = []
        for idx in range(0, all_embeds.shape[0], self.batch_size):
            batch_embeds = all_embeds[idx: idx+self.batch_size]
            batch_embeds = torch.FloatTensor(batch_embeds).to(device=self.device, dtype=query_embeds.dtype)

            with torch.no_grad():
                tmp_scores = self.processor.score_multi_vector(query_embeds, batch_embeds)
                if len(tmp_scores.shape) > 1:
                    tmp_scores = tmp_scores[0]

            all_scores.append(tmp_scores)
        scores = torch.cat(all_scores, dim=0).cpu()
        del all_scores, queries, query_embeds 
        return scores
    
    def base_retrieve(self, query, all_embeds, top_k=10):
        scores = self.compute_scores(query, all_embeds)
        top_indices = scores.argsort(dim=-1, descending=True)[:top_k].tolist()
        top_scores = scores[top_indices].tolist()
        # 1-indexed
        return [idx+1 for idx in top_indices], top_scores
    
    def vlm_retrieve(self, query, all_embeds, graph, doc_id, beam_width=3, max_hop=5, verbose=True):
        scores = self.compute_scores(query, all_embeds)
        min_score = torch.min(scores).item()
        max_score = torch.max(scores).item()
        score_range = max_score - min_score if max_score > min_score else 1.0
    
        vlm_score_cache, vlm_query_times = {}, 0
        score_dict = {i: (scores[i].item() - min_score) / score_range for i in range(scores.shape[0])} # score normalization

        # Initialize the beam search 
        beam = scores.argsort(dim=-1, descending=True)[:beam_width].tolist()
        visited = set(beam)

        for node in beam: 
            vlm_score = query_vlm_relevance(query, (doc_id, node+1), vlm_model)
            vlm_query_times += 1
            vlm_score_cache[node] = vlm_score
            norm_vlm_score = (vlm_score - 1.0) / 4.0 

            combined_score = args.alpha * score_dict[node] + (1.0 - args.alpha) * norm_vlm_score
            score_dict[node] = combined_score
        
        if verbose:
            print(f"Initial Beam: {[node_id+1 for node_id in beam]}; Corresponding Scores: {[round(score_dict[node], 3) for node in beam]}")
        
        result_dict = {node: score_dict[node] for node in beam}

        # Start the beam search
        for hop in range(max_hop):
            candidates = []
            for node in beam:
                candidate_neighbors = graph.get(node, [])
                for neighbor in candidate_neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)

                        sim_score = score_dict[neighbor]
                        vlm_score = query_vlm_relevance(query, (doc_id, neighbor+1), vlm_model)
                        vlm_query_times += 1
                        vlm_score_cache[neighbor] = vlm_score
                        norm_vlm_score = (vlm_score - 1.0) / 4.0

                        combined_score = args.alpha * sim_score + (1.0 - args.alpha) * norm_vlm_score
                        score_dict[neighbor] = combined_score
                        candidates.append((neighbor, combined_score))
                        result_dict[neighbor] = combined_score
            
            if not candidates:
                break 

            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            beam = [node for node, _ in candidates]
            if verbose:
                print(f"Current Beam: {[node_id+1 for node_id in beam]}; Corresponding Scores: {[round(score_dict[node], 3) for node in beam]}")
        
        final_results = [(node, score) for node, score in result_dict.items() if score >= args.threshold]
        final_results = sorted(final_results, key=lambda x: x[1], reverse=True)
        evidence_pages = [node+1 for node, _ in final_results]
        page_scores = [score for _, score in final_results]
        torch.cuda.empty_cache()

        print(f"Total Pages {all_embeds.shape[0]}; VLM Query Times: {vlm_query_times}")
        return evidence_pages, page_scores


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", type=str, default="MMLong", choices=["MMLong", "LongDocURL", "FetaTab", "PaperTab"])
    args.add_argument("--method", type=str, default="base", choices=["base", "beamsearch"])
    args.add_argument("--emb_root", type=str, default="../tmp/tmp_embs")
    args.add_argument("--device", type=str, default="cuda:0")
    args.add_argument("--encoder", type=str, default="vidore/colpali")
    args.add_argument("--top_k", type=int, default=20)
    args.add_argument("--model_name", type=str, default="QwenVL-3B", choices=["QwenVL-3B", "QwenVL-3B-lora"])
    args.add_argument("--threshold", type=float, default=0.3) # 0.3 default; 0.4 for MMLong + QwenVL-3B-lora

    # beamsearch specific
    args.add_argument("--alpha", type=float, default=0.4) # 0.4 for MMLong and (currently) 0.6 for LongDocURL
    args.add_argument("--sim_measure", type=str, default="cosine")
    args.add_argument("--beam_width", type=int, default=3)
    args.add_argument("--max_hop", type=int, default=4)
    args.add_argument("--beam_verbose", type=bool, default=True)
    args = args.parse_args()

    device = torch.device(args.device)
    doc2emb = load_all_doc_embeddings(f"{args.emb_root}/{args.dataset}") # dict format {doc_id: doc_emb}
    model, vlm_model = ColPali.from_pretrained(args.encoder, torch_dtype=torch.bfloat16, device_map=device).eval(), None
    processor = ColPaliProcessor.from_pretrained(args.encoder)
    doc_retriever = DocumentRetriever(encoder=model, processor=processor, device=device)
    
    samples = json.load(open(f"../dataset/samples_{args.dataset}.json", 'r'))
    vlm_suffix = "_LoRA" if args.method == "beamsearch" and "lora" in args.model_name else ""
    retrieve_file = f"../dataset/retrieved/samples_{args.dataset}_{args.method}{vlm_suffix}.json"

    if args.method == "beamsearch":
        from VLMModels.Qwen_VL import init_model, get_response_concat
        vlm_model = init_model(args.model_name, device=device)
        doc2graph = {}
        for doc_id, doc_emb in tqdm(doc2emb.items(), desc="Constructing Page Graph"):
            cur_graph = construct_page_graph(doc_emb, threshold=0.8, sim_measure=args.sim_measure)
            doc2graph[doc_id] = deepcopy(cur_graph)

    for sample in tqdm(samples, desc="Retrieving"):
        query = sample["question"]
        target_doc = sample["doc_id"].replace(".pdf", "")
        target_doc_embedding = doc2emb[target_doc]
        if args.method == "base":
            ranked_pages, page_scores = doc_retriever.base_retrieve(query, target_doc_embedding, top_k=args.top_k)
        elif args.method == "beamsearch":
            target_graph = doc2graph.get(target_doc, defaultdict(list))
            try:
                assert target_graph is not None
            except Exception as e: 
                print(f'Error in graph', target_graph)
                target_graph = defaultdict(list)
                
            ranked_pages, page_scores = doc_retriever.vlm_retrieve(query, target_doc_embedding, target_graph, target_doc, beam_width=args.beam_width, max_hop=args.max_hop, verbose=args.beam_verbose)
        
        sample["pages_ranking"] = str(ranked_pages)
        sample["pages_scores"] = str(page_scores)
        if "evidence_pages" in sample:
            print("Ground-truth", sample["evidence_pages"])
        print("Prediction", ranked_pages[:5], "\n")
        
        os.makedirs(os.path.dirname(retrieve_file), exist_ok=True)
        json.dump(samples, open(retrieve_file, 'w'), indent=4)
