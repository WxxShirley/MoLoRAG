from colpali_engine.models import ColPali, ColPaliProcessor
import os 
from pdf2image import convert_from_path
import argparse
import torch
from tqdm import tqdm
import sys 
sys.path.append("../")
from utils import prepare_files


def encode_document(doc_path, doc_id, batch_size=32, resolution=144, save_emb=True, save_img=False):
    page_images = convert_from_path(doc_path, dpi=resolution)
    if save_img: 
        os.makedirs(img_save_dir, exist_ok=True)
        for page_num, page_snapshot in enumerate(page_images):
            # re-index the page number to start from 1
            if not os.path.exists(f"{img_save_dir}/{doc_id}-{page_num+1}.png"):
                page_snapshot.save(f"{img_save_dir}/{doc_id}-{page_num+1}.png") 

    total_image_embeds = torch.Tensor().to(device)
    for idx in range(0, len(page_images), batch_size):
        batch_images = page_images[idx: idx+batch_size]
        batch_images = processor.process_images(batch_images).to(device)

        with torch.cuda.device(device):
            torch.cuda.empty_cache()

        with torch.no_grad():
            image_embeds = model(**batch_images)
        total_image_embeds = torch.cat((total_image_embeds, image_embeds), dim=0)
    
    if save_emb:
        torch.save(total_image_embeds, f"{save_dir}/{doc_id}.pt")
        print(f"Save Embeddings {total_image_embeds.shape} to {save_dir}/{doc_id}.pt")
       

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MMLong", choices=["MMLong", "LongDocURL", "PaperTab", "FetaTab"])
    parser.add_argument("--save_dir", type=str, default="../tmp/tmp_embs")
    parser.add_argument("--img_save_dir", type=str, default="../tmp/tmp_imgs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_name", type=str, default="vidore/colpali")
    parser.add_argument("--save_img", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    model = ColPali.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map=device).eval()
    processor = ColPaliProcessor.from_pretrained(args.model_name)

    documents = prepare_files(f"../dataset/{args.dataset}", suffix=".pdf")
    save_dir, img_save_dir = f"{args.save_dir}/{args.dataset}", f"{args.img_save_dir}/{args.dataset}"
    os.makedirs(save_dir, exist_ok=True)

    for doc_path in tqdm(documents, desc="Encoding PDFs ..."):
        doc_id = doc_path.replace(".pdf", "") 

        if os.path.exists(f"{save_dir}/{doc_id}.pt"):
            print(f"Embeddings for {doc_path} already exists.")
            continue 
        
        encode_document(doc_path=f"../dataset/{args.dataset}/{doc_path}", doc_id=doc_id, batch_size=args.batch_size, save_img=args.save_img)
