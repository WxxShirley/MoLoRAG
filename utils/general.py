import pytz 
import datetime 
import time 
import numpy as np


def get_cur_time(timezone='Asia/Shanghai', t_format='%m-%d %H:%M:%S'):
    return datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone(timezone)).strftime(t_format)


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def distnace_similarity(vec1, vec2):
    distance = np.linalg.norm(vec1 - vec2)
    if distance < 1e-9:
        return 1.0 
    return np.exp(-distance)


def similarity_func(vec1, vec2, metric="cosine"):
    assert metric in ["cosine", "distance"]
    if metric == "cosine": 
        return cosine_similarity(vec1, vec2)
    
    return distnace_similarity(vec1, vec2)


def compute_embed_similarity(doc1_embed, doc2_embed, sim_func="distance"):
    doc1_avg = np.mean(doc1_embed, axis=0)
    doc2_avg = np.mean(doc2_embed, axis=0)
    
    return similarity_func(doc1_avg, doc2_avg, sim_func)
