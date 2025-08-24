MODEL=QwenVL-7B
for K in 1 3 5 ; do 
     for RETRIVER in None base beamsearch beamsearch_LoRA; do 
          python3 -u main_eval.py --method=VLM --model_name=$MODEL --dataset=FetaTab --retriever=$RETRIVER --topk=$K
     done 
done 
