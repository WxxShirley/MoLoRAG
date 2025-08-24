# Step 1 - Indexing 

python3 index.py --dataset MMLong --save_img 


# Step 2 - Retrieve 

#  - Base Retriever (M3DocRAG) 
python3 retrieve.py --dataset MMLong --method base 
#  - MoLoRAG 
python3 retrieve.py --dataset MMLong --method beamsearch --model_name QwenVL-3B 
#  - MoLoRAG* 
#    First download the model from https://huggingface.co/xxwu/MoLoRAG-QwenVL-3B
#    Then set the correct model path in VLMModels/Qwen_VL.py 
python3 retrieve.py --dataset MMLong --method beamsearch --model_name QwenVL-3B-lora


# (Optional) - Evaluate the RAG quality 
cd ../evaluate 

python3 eval_rag.py --dataset MMLong 
