# Before making predictions, you have to prepare the retrieved contents following scripts in `VLMRetriever/example_run.sh`
# Before making predictions, you have to prepare the retrieved contents following scripts in `VLMRetriever/example_run.sh`
# Before making predictions, you have to prepare the retrieved contents following scripts in `VLMRetriever/example_run.sh`


DATASET=MMLong 

# QwenVL 
# - Direct
TOKENIZERS_PARALLELISM=false python3 -u main.py --dataset=$DATASET --model_name=QwenVL-7B --device=cuda:0,cuda:1,cuda:2  --retriever=None --max_pages=30 
# - M3DocRAG 
TOKENIZERS_PARALLELISM=false python3 -u main.py --dataset=$DATASET --model_name=QwenVL-7B --device=cuda:0  --retriever=base --topk=3 
# - MoLoRAG
TOKENIZERS_PARALLELISM=false python3 -u main.py --dataset=$DATASET --model_name=QwenVL-7B --device=cuda:2  --retriever=beamsearch  --topk=3 
# - MoLoRAG*
TOKENIZERS_PARALLELISM=false python3 -u main.py --dataset=$DATASET --model_name=QwenVL-7B --device=cuda:2  --retriever=beamsearch_LoRA --topk=3  


# LLaVA-Next 
# - Direct 
TOKENIZERS_PARALLELISM=false python3 -u main.py --dataset=$DATASET --model_name=LLaVA-Next-7B --retriever=None  --max_pages=30 --concat_num=1 --device=cuda:2 
# - M3DocRAG
TOKENIZERS_PARALLELISM=false python3 -u main.py --dataset=$DATASET --model_name=LLaVA-Next-7B --retriever=base   --topk=1 --device=cuda:2 
# - MoLoRAG
TOKENIZERS_PARALLELISM=false python3 -u main.py --dataset=$DATASET --model_name=LLaVA-Next-7B --retriever=beamsearch   --topk=1  --device=cuda:2 
# - MoLoRAG*
TOKENIZERS_PARALLELISM=false python3 -u main.py --dataset=$DATASET --model_name=LLaVA-Next-7B --retriever=beamsearch_LoRA   --topk=1  --device=cuda:2 

# - M3DocRAG
TOKENIZERS_PARALLELISM=false python3 -u main.py --dataset=$DATASET --model_name=LLaVA-Next-7B --retriever=base   --topk=3 --concat_num=1 
# - MoLoRAG
TOKENIZERS_PARALLELISM=false python3 -u main.py --dataset=$DATASET --model_name=LLaVA-Next-7B --retriever=beamsearch   --topk=3  --concat_num=1 
# - MoLoRAG*
TOKENIZERS_PARALLELISM=false python3 -u main.py --dataset=$DATASET --model_name=LLaVA-Next-7B --retriever=beamsearch_LoRA   --topk=3  --concat_num=1 


# DeepSeek-Vision 
# - Direct
python3 -u main.py --model_name=DeepSeek-VL-small --retriever=None --max_pages=30 --concat_num=5 --device=cuda:0,cuda:1,cuda:2,cuda:3 --dataset=$DATASET
# - M3DocRAG
python3 -u main.py --model_name=DeepSeek-VL-small --retriever=base --topk=5 --device=cuda:0,cuda:1 --dataset=$DATASET
# - MoLoRAG
python3 -u main.py --model_name=DeepSeek-VL-small --retriever=beamsearch --topk=5 --device=cuda:0,cuda:1 --dataset=$DATASET 
# - MoLoRAG*
python3 -u main.py --model_name=DeepSeek-VL-small --retriever=beamsearch_LoRA --topk=5 --device=cuda:0,cuda:1 --dataset=$DATASET 

