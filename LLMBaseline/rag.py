import sys 
sys.path.append('../')
from utils import prepare_files, get_cur_time
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
import os 
from tqdm import tqdm
import argparse 


def index_single_pdf(filepath, doc_id, default_parser=True):
    # Step 1 - Load the PDF file
    if default_parser:
        loader = PyPDFLoader(filepath)
    else:
        loader = UnstructuredFileLoader(filepath)
    pages = loader.load_and_split()

    # Step 2 - Split the PDF into chunks 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=args.chunk_size, 
                                                   chunk_overlap=args.overlap, 
                                                   add_start_index=True)
    chunks = text_splitter.split_documents(pages)
    print(f'Split {filepath} document into {len(chunks)} chunks')

    # Step 3 - Index the chunks 
    # TODO: Set your DashScope API key here
    embeddings = DashScopeEmbeddings(model="text-embedding-v1", 
                                     dashscope_api_key="YOUR Key HERE",)
    cur_savepath = f"{save_dir}/{doc_id}"

    faiss_index = FAISS.from_documents(chunks, embeddings)
    faiss_index.save_local(cur_savepath)
    print(f"Saved the index to {cur_savepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MMLong") # choices ["MMLong", "LongDocURL", "FetaTab", "PaperTab"]
    parser.add_argument("--save_dir", type=str, default="../tmp/tmp_dbs")

    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=100)
    args = parser.parse_args()
    
    save_dir = f"{args.save_dir}/{args.dataset}" 
    os.makedirs(save_dir, exist_ok=True)

    print(f"{get_cur_time()} - Start indexing PDF files in {args.dataset} ...")
    
    pdf_files = prepare_files(root_dir=f"../dataset/{args.dataset}", suffix=".pdf")
    
    for cur_pdf in tqdm(pdf_files, desc=f"Tranversing PDF files in {args.dataset} ..."):
        doc_id = cur_pdf.replace('.pdf', '')
        if os.path.exists(f"{save_dir}/{doc_id}"):
            print(f"Skip {cur_pdf} as it has been indexed")
            continue 
        
        try: 
            index_single_pdf(filepath=f"../dataset/{args.dataset}/{cur_pdf}", doc_id=doc_id)
        except Exception as e:
            try: 
                index_single_pdf(filepath=f"../dataset/{args.dataset}/{cur_pdf}", doc_id=doc_id, default_parser=False)
            except Exception as e:
                print(f"[ERROR] processing {cur_pdf}: {e}")

    print(f"{get_cur_time()} - Finished ! ")
