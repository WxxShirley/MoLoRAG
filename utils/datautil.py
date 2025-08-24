import os 
from tqdm import tqdm 
import torch 
from copy import deepcopy
import numpy as np
from collections import defaultdict
from .general import compute_embed_similarity
import math 
from PIL import Image


def prepare_files(root_dir: str, suffix=".pdf"):
    """Prepare the list of files in the `root_dir` with the specified `suffix`"""
    target_files = [file for file in os.listdir(root_dir) if file.endswith(suffix)]
    return target_files


def load_all_doc_embeddings(root_dir):
    """Load all embeddings and organize them."""
    emb_files = sorted(prepare_files(root_dir, suffix=".pt"))
    docid2emb = {}

    for emb_file in tqdm(emb_files, desc="Loading and Organizing Embeddings"):
        embeds = torch.load(f"{root_dir}/{emb_file}", map_location="cpu", weights_only=True).detach().numpy()
        
        doc_id = emb_file.replace(".pt", "")
        docid2emb[doc_id] = embeds

    return docid2emb


def construct_page_graph(doc_emb, threshold=0.7, k_value=5, sim_measure="cosine"):
    """Construct a page graph based on the similarity between page embeddings."""
    n_pages, _, _ = doc_emb.shape
    if n_pages <= 3: # skip small documents
        return None

    edges = []
    
    sim_matrix = np.zeros((n_pages, n_pages))
    for i in range(n_pages):
        for j in range(i+1, n_pages):
            vec_i, vec_j = doc_emb[i], doc_emb[j]
            sim_score = compute_embed_similarity(vec_i, vec_j, sim_func=sim_measure)
            sim_matrix[i][j] = sim_score
            sim_matrix[j][i] = sim_score
    
    # k-NN Graph
    for i in range(n_pages):
        sim_scores = sim_matrix[i] 
        top_k_indices = np.argsort(sim_scores)[::-1][:k_value]

        for j in top_k_indices:
            if sim_scores[j] >= threshold:
                edges.append((i, j))
    
    page_graph_dict = defaultdict(list)
    for u, v in edges:
        page_graph_dict[int(u)].append(int(v))
        page_graph_dict[int(v)].append(int(u))
    
    page_graph_dict = {k: list(set(v)) for k, v in page_graph_dict.items()}
    # print(page_graph_dict, "\n")
    # print(f"Doc-{doc_id}-Graph # Nodes {n_pages}, # Edges {len(edges)}")
    return page_graph_dict


def convert_page_snapshot_to_image(doc_path, save_path, resolution=144, max_pages=1000):
    """Convert a PDF document to a list of images."""
    from pdf2image import convert_from_path
    page_snapshots = convert_from_path(doc_path, dpi=resolution)
    doc_id = doc_path.split("/")[-1].replace(".pdf", "")
    image_path_list = []
    for page_num, page_snapshot in enumerate(page_snapshots[:max_pages]):
        if not os.path.exists(f"{save_path}/{doc_id}-{page_num+1}.png"):
            page_snapshot.save(f"{save_path}/{doc_id}-{page_num+1}.png", "PNG")
        image_path_list.append(f"{save_path}/{doc_id}-{page_num+1}.png")
    
    return image_path_list


def concat_images(image_list, concat_num=1, column_num=3, name_suffix="concat"):
    """Concatenate a list of images into `concat_num` images."""
    interval = max(math.ceil(len(image_list) / concat_num), 1) # number of images in each batch
    concatenated_image_list = list()

    for i in range(0, len(image_list), interval):
        image_path = "-".join(image_list[0].split("-")[:-1]) + "-{}{}-{}.jpg".format(name_suffix, concat_num, i//interval)
        print(image_path)
        if not os.path.exists(image_path):
            images_this_batch = [
                Image.open(filename) for filename in image_list[i:i + interval]
            ]
            if column_num == 1:
                total_height = images_this_batch[0].height*len(images_this_batch)
            else:
                total_height = images_this_batch[0].height*((len(images_this_batch)-1)//column_num+1)

            concatenated_image = Image.new('RGB', (images_this_batch[0].width*column_num, total_height), 'white')
            x_offset, y_offset = 0, 0
            for cnt, image in enumerate(images_this_batch):
                concatenated_image.paste(image, (x_offset, y_offset))
                x_offset += image.width
                if (cnt+1)%column_num==0:
                    y_offset += image.height
                    x_offset = 0
            concatenated_image.save(image_path)
        concatenated_image_list.append(image_path)

    return concatenated_image_list
