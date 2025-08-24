import argparse
import json
import numpy as np
from math import log2 
import os
import prettytable as pt


def ndcg_cell(ground_truth, prediction, k):
    k = min(len(prediction), len(ground_truth), k)
    dcg = 0.0 

    for i, doc_id in enumerate(prediction[:k]):
        rel = 1.0 if doc_id in ground_truth else 0.0 
        dcg += rel / log2(i+2)
    
    idcg = sum(1.0 / log2(i+2) for i in range(k))

    if idcg == 0:
        return 0.0

    return dcg / idcg * 100.0


def mrr_cell(ground_truth, prediction, k):
    for i, item in enumerate(prediction[:k]):
        if item in ground_truth:
            return (1.0 / (i+1)) * 100.0 
    
    return 0.0 


def evaluate_rag_one_sample(support_context, pred_context, top_k=[1, 5, 10]):
    """Evaluate the RAG on one sample
    Args:
        support_context (list): The ground truth evidence pages
        pred_context (list): The predicted evidence pages
        top_k (list): The top k to evaluate"""
    metrics = {}

    for k in top_k:
        cur_pred = pred_context[:k]
        intersect = len(set(cur_pred) & set(support_context)) 
        # Precision-related 
        metrics[f"recall@{k}"] = intersect / len(support_context) * 100.0
        metrics[f"precision@{k}"] = intersect / len(cur_pred) * 100.0 if len(cur_pred) > 0 else 0.0
        metrics[f"irrelevant@{k}"] = (len(cur_pred) - intersect) / len(cur_pred) * 100.0 if len(cur_pred) > 0 else 0.0
        
        # Ranking-related
        metrics[f"ndcg@{k}"] = ndcg_cell(support_context, cur_pred, k)
        metrics[f"mrr@{k}"] = mrr_cell(support_context, cur_pred, k)
        
    return metrics


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    # only MMLong and LongDocURL provide the ground-truth evidence pages
    args.add_argument("--dataset", type=str, default="MMLong", choices=["MMLong", "LongDocURL"])
    args.add_argument("--k_list", type=list, default=[1, 3, 5, 10])
    args = args.parse_args()
    
    # evaluate ours (beamsearch-based)
    for retrieve_method in ["base", "beamsearch", "beamsearch_LoRA"]:
        filepath = f"../dataset/retrieved/samples_{args.dataset}_{retrieve_method}.json"
        if not os.path.exists(filepath):
            continue 
        
        table = pt.PrettyTable()
        table.field_names = ["Method", "K", "Recall(%)", "Precision(%)", "NDCG(%)", "MRR(%)", "Irrelevant(%)"]
        samples = json.load(open(filepath, 'r'))
        all_metrics = {f'recall@{k}': [] for k in args.k_list}
        for remain_metric in ['ndcg', 'mrr', 'precision', 'irrelevant']:
            all_metrics.update({f'{remain_metric}@{k}': [] for k in args.k_list})

        for sample in samples:
            if eval(sample["evidence_pages"]) == [] or "pages_ranking" not in sample:
                continue
            
            preds = eval(sample["pages_ranking"])
            score = evaluate_rag_one_sample(
                support_context=eval(sample["evidence_pages"]),
                pred_context=preds,
                top_k=args.k_list
            )

            for metric_name, value in score.items():
                all_metrics[metric_name].append(value)
        
        for metric_name, values in all_metrics.items():
            all_metrics[metric_name] = np.round(np.mean(values), 2)
        
        for k in args.k_list:
            table.add_row([retrieve_method, k, all_metrics[f'recall@{k}'], all_metrics[f'precision@{k}'], all_metrics[f'ndcg@{k}'], all_metrics[f'mrr@{k}'], all_metrics[f'irrelevant@{k}']])
        print(table, '\n')
    
    # evaluate mdocagent (retriever-based) 
    filepath = f"../dataset/retrieved/samples_{args.dataset}_mdocagent.json"
    if os.path.exists(filepath):
        samples = json.load(open(filepath, 'r')) 
        
        table = pt.PrettyTable()
        table.field_names = ["Method", "K", "Recall(%)", "Precision(%)", "NDCG(%)", "MRR(%)", "Irrelevant(%)"]
        for target in ["text", "image"]:

            all_metrics = {f'recall@{k}': [] for k in args.k_list}
            for remain_metric in ['ndcg', 'mrr', 'precision', 'irrelevant']:
                all_metrics.update({f'{remain_metric}@{k}': [] for k in args.k_list})

            for sample in samples:
                ground_truth = eval(sample["evidence_pages"]) 
                if ground_truth == [] or f"{target}-top-10" not in sample:
                    continue 

                cur_preds = eval(sample[f"{target}-top-10"])
                scores = evaluate_rag_one_sample(support_context=ground_truth, pred_context=cur_preds, top_k=args.k_list)

                for metric_name, value in scores.items():
                    all_metrics[metric_name].append(value)
            
            for metric_name, values in all_metrics.items():
                all_metrics[metric_name] = np.round(np.mean(values), 2)
            
            for k in args.k_list:
                table.add_row([f"mdocagent-{target}", k, all_metrics[f'recall@{k}'], all_metrics[f'precision@{k}'], all_metrics[f'ndcg@{k}'], all_metrics[f'mrr@{k}'], all_metrics[f'irrelevant@{k}']])
        
        print(table, '\n')
