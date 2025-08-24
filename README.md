# MoLoRAG 

This repository is the official implementation for our EMNLP 2025 paper: **MoLoRAG: Bootstrapping Document Understanding via Multi-modal Logic-aware Retrieval**. 

> Please consider citing or giving a 🌟 if our repository is helpful to your work!

```
@inproceedings{wu2025molorag
    title={MoLoRAG: Bootstrapping Document Understanding via Multi-modal Logic-aware Retrieval}, 
    author={Xixi Wu and Yanchao Tan and Nan Hou and Ruiyang Zhang and Hong Cheng},
    year={2025},
    booktitle={Conference on Empirical Methods in Natural Language Processing},
    url={https://arxiv.org/abs/xxxx.xxxxx}, 
}
```

## 📢 News

🎉 [2025-08-24] Our paper is accepted to **EMNLP 2025**. The camera ready paper, and fully reviewed codes will be released soon! 

--- 

## 📖 Dataset 

Full datasets are available at [HuggingFace](https://huggingface.co/datasets/xxwu/MoLoRAG): 

```
huggingface-cli download --repo-type dataset xxwu/MoLoRAG --local-dir ./dataset/
```


## 🔧 Environment

For running Qwen2.5-VL-series models, it requires a unique environment: 
```
transformers==4.50.0.dev0
torch==2.6.0
qwen-vl-utils==0.0.8
```

For running remaining LVLMs, VLM retrieve, and LLM baselines, the required packages are:
```
transformers==4.47.1
torch==2.5.1
colpali_engine==0.3.8
colbert-ai==0.2.21
langchain==0.3.19
langchain-community==0.3.18
langchain-core==0.3.37
langchain-text-splitters==0.3.6
```

## 🤗 Model 

We release our fine-tuned VLM retriever, MoLoRAG-3B, based on the Qwen2.5-VL-3B, at [HuggingFace](https://huggingface.co/xxwu/MoLoRAG-QwenVL-3B):

```
huggingface-cli download xxwu/MoLoRAG-QwenVL-3B
```

The training data for fine-tuning this retriever to enable its logic-aware ability, is available at [HuggingFace](https://huggingface.co/datasets/xxwu/MoLoRAG/blob/main/train_MoLoRAG_pairs_gpt4o.json). The data generation pipeline is available at https://github.com/WxxShirley/MoLoRAG/blob/main/VLMRetriever/data_collection.py. 


## 🚀 Run 

- LLM Baselines 

Codes and commands are available at `LLMBaseline` 


- LVLM Baselines 

Step 0 - Prepare the retrieved contents following commands at `VLMRetriever` 

Step 1 - Make predictions following commands at `example_run.sh`

Step 2 - Evaluate the inference following commands at `example_run_eval.sh` 


## 📮 Contact 

If you have any further questions about usage, reproducibility, or would like to discuss, please feel free to open an issue or contact the authors via email at xxwu@se.cuhk.edu.hk.


## 🙏 Acknowledgements 

We thank the open-sourced [MMLongBench](https://github.com/mayubo2333/MMLongBench-Doc/), [LongDocURL](https://github.com/dengc2023/LongDocURL/) datasets. We also appreciate the official implementations of [M3DocRAG](https://github.com/bloomberg/m3docrag) and [MDocAgent](https://github.com/aiming-lab/MDocAgent). 

