# Making Prediction
for DATASET in LongDocURL ; do 
    for K in 1 3 5 ; do 
        for LLM in deepseek-chat  ; do  
             python3 -u main.py --llm_name=$LLM --retrieve_topk=$K --dataset=$DATASET  >>../results/$DATASET/logs/$LLM+MineUOCR.log 
        done 
    done
done 

# TODO: Evaluation
