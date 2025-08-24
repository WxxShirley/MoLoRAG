import re
from math import isclose
from collections import defaultdict
from LLMBaseline.apis import invoke_gpt4o_api
import json 


def levenshtein_distance(s1, s2):
    s1, s2 = s1.lower(), s2.lower()
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def answer_score(ground_truth, prediction, threshold=0.5):
    dist = levenshtein_distance(ground_truth, prediction)
    length = max(len(ground_truth), len(prediction))
    value = 0.0 if length == 0 else float(dist) / float(length)
    score = 1.0 - value 

    if score < threshold:
        return 0.0 
    return score


def get_clean_string(s):
    s = str(s).lower().strip()
    for suffix in ["meters", "meter", "mm", "m", "mile", "miles", "thousand", "million", "billion", "kg", "acres", "minutes"]:
        if s.endswith(suffix):
            s = s.rstrip(suffix).strip()
    # remove parenthesis
    s = re.sub(r'\s*\([^)]*\)', '', s).strip()
    # remove quotes
    s = re.sub(r"^['\"]|['\"]$", "", s).strip()
    s = s.strip().lstrip("$").strip()
    s = s.strip().rstrip("%").strip()
    s = s.strip().lstrip("£").strip()
    return s


def is_format_match(s):
    flag = False
    # Website
    if "https://" in s or "http://" in s:
        flag = True
    # code file
    if s.endswith(".py") or s.endswith("ipynb"):
        flag = True
    if s.startswith("page"):
        flag = True
    # telephone number
    if re.fullmatch(r'\b\d+(-\d+|\s\d+)?\b', s):
        flag = True
    # time
    if "a.m." in s or "p.m." in s:
        flag = True
    # YYYY-MM-DD
    if re.fullmatch(r'\b\d{4}[-\s]\d{2}[-\s]\d{2}\b', s):
        flag = True
    # YYYY-MM
    if re.fullmatch(r'\b\d{4}[-\s]\d{2}\b', s):
        flag = True
    # Email address
    if re.fullmatch(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', s):
        flag = True
    return flag


def is_float_equal(reference, prediction, include_percentage: bool = False, is_close: float = False) -> bool:
    def get_precision(gt_ans: float) -> int:
        precision = 3
        if '.' in str(gt_ans):
            precision = len(str(gt_ans).split('.')[-1])
        return precision

    reference = float(str(reference).strip().rstrip("%").strip())
    try:
        prediction = float(str(prediction).strip().rstrip("%").strip())
    except:
        return False

    if include_percentage:
        gt_result = [reference / 100, reference, reference * 100]
    else:
        gt_result = [reference]
    for item in gt_result:
        try:
            if is_close:
                if isclose(item, prediction, rel_tol=0.01):
                    return True
            precision = max(min(get_precision(prediction), get_precision(item)), 2)
            if round(prediction, precision) == round(item, precision):
                return True
        except Exception:
            continue
    return False


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def eval_one_sample(gt, pred, answer_type):
    if answer_type == "Int":
        try:
            gt, pred = int(gt), int(float(pred))
        except:
            pred = ""
        em, acc = (gt == pred), (gt == pred)
    elif answer_type == "Float":
        try:
            gt = float(get_clean_string(str(gt)))
            pred = float(get_clean_string(str(pred)))
        except:
            pred = ""
        
        score = is_float_equal(gt, pred, include_percentage=True, is_close=True)
        em, acc = score, score
    elif answer_type in ["Str", "None"]:
        gt = get_clean_string(gt)
        pred = get_clean_string(pred)
        if is_format_match(gt):
            em, acc = (gt == pred), (gt == pred)
        else:
            em = (gt == pred)
            acc = answer_score(gt, pred)
    else:
        if isinstance(gt, str) and gt.startswith("["):
            gt = eval(gt)
        if not isinstance(gt, list):
            gt = [gt] 
        
        if isinstance(pred, str) and pred.startswith("["):
            try:
                pred = eval(pred)
            except Exception as e:
                print(f"[ERROR for Evaluation] {e} Prediction {pred}")
                pred = [] 

        if not isinstance(pred, list):
            pred = [pred]

        gt = sorted([get_clean_string(a) for a in gt])
        pred = sorted([get_clean_string(b) for b in pred])

        em = ("-".join(gt) == "-".join(pred))
        
        try: 
            if len(gt) == len(pred):
                if len(gt) == 0:
                    acc = 1.0
                else:
                    element_scores = [answer_score(a, b, threshold=0.8) if not (isfloat(a) or is_format_match(a)) else a == b for a, b in zip(gt, pred)]
                    acc = sum(element_scores) / len(element_scores) 
            else:
                # Greedy matching
                greedy_scores = []

                for gt_element in gt:
                    max_score = max([answer_score(gt_element, pred_element, threshold=0.8) for pred_element in pred])
                    greedy_scores.append(max_score)
            
                avg_score = sum(greedy_scores) / len(greedy_scores)
                length_penalty = min(1.0, len(pred) / len(gt)) ** 0.5 
                acc = avg_score * length_penalty
                
        except Exception as e:
            print(f"[ERROR for Evaluation] {e} Ground-truth {gt} Prediction {pred}")
            acc = 0.0
            
    return em, acc


def eval_samples(samples, dataset_name: str):
    evaluated_samples = [sample for sample in samples if "score" in sample]
     
    if not evaluated_samples: 
        return None
    
    score_keys = evaluated_samples[0]["score"].keys()
    metrics = {"QuestionNumber": len(evaluated_samples)}
    for score_key in score_keys:
        avg_score = sum(sample["score"][score_key] for sample in evaluated_samples) / len(evaluated_samples)
        metrics[score_key] = round(avg_score * 100, 2) 

    if dataset_name == "MMLong": 
        # consider F1 score 
        try:
            recall = sum(sample["score"]["Acc"] for sample in evaluated_samples if sample["answer"] != "Not answerable") / len([sample for sample in evaluated_samples if sample["answer"] != "Not answerable"])
            precision = sum(sample["score"]["Acc"] for sample in evaluated_samples if sample["answer"] != "Not answerable") / len([sample for sample in evaluated_samples if sample["pred_ans"] != "Not answerable"])

            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0 
        except:
            f1 = 0.0 
        metrics["F1"] = round(f1 * 100, 2)
    
    return metrics


def extract_answer(question, output, extractor_prompt):
    try:
        full_query_prompt = f"{extractor_prompt}\n\nQuestion: {question}\nAnalysis: {output}\n"
        response = invoke_gpt4o_api(content=full_query_prompt, temperature=0.1)

    except Exception as e:
        print(f"[ERROR for Extraction] {e}")
        response = "Failed"
    
    return response


def extract_score(question, output, ground_truth, prompt):
    try:
        query_prompt = prompt.format(question=question, answer=output, gt=ground_truth)
        eval_str = invoke_gpt4o_api(content=query_prompt)

        start_index, end_index = eval_str.find('{'), eval_str.rfind('}') + 1 
        eval_str = eval_str[start_index:end_index]
        metrics = json.loads(eval_str)
        return {
            'binary_correctness': int(metrics.get('binary_correctness', 0)),
        }
    except Exception as e: 
        print(f"[ERROR for Scoring] {e}")
        return {
            'binary_correctness': 0,
        }


def show_fine_grained_results(samples, dataset="MMLong"):
    for sample in samples:
        sample["evidence_pages"] = eval(sample["evidence_pages"])
        sample["evidence_sources"] = eval(sample["evidence_sources"])
    
    score_dict = eval_samples(samples, dataset) 
    if not score_dict: 
        return 
    
    print(f"Overall Evaluation: {score_dict}")
    print("-----------------------\n") 
    
    # Score by Page
    single_page_results = eval_samples([sample for sample in samples if len(sample["evidence_pages"]) == 1], dataset)
    multi_pages_results = eval_samples([sample for sample in samples if len(sample["evidence_pages"]) > 1 and sample["answer"] != "Not answerable"], dataset)
    unans_results = eval_samples([sample for sample in samples if sample["answer"] == "Not answerable"], dataset)
    print(f"Single-Page Evaluation: {single_page_results}") 
    print(f"Multi-Pages Evaluation: {multi_pages_results}") 
    print(f"Unanswerable Evaluation: {unans_results}") 
    print("-----------------------\n") 

    # Score by Source 
    source_sample_dict = defaultdict(list) 
    for sample in samples:
        for answer_source in sample["evidence_sources"]:
            source_sample_dict[answer_source].append(sample)

    for source, source_samples in source_sample_dict.items(): 
        cur_results = eval_samples(source_samples, dataset)
        print(f"Source-{source} Evaluation: {cur_results}")
    print("-----------------------\n") 
