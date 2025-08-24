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
### 📢 News

🎉 **[2025-08-24]** Our paper is accepted to **EMNLP 2025**. The camera ready paper and fully reviewed codes will be released soon!

---

## 📋 Table of Contents

- [📚 Dataset](#-dataset)
- [🔧 Environment](#-environment)
- [🤗 Model](#-model)
- [🚀 Run](#-run)
- [📮 Contact](#-contact)
- [🙏 Acknowledgements](#-acknowledgements)



## 📚 Dataset

Full datasets are available at [HuggingFace](https://huggingface.co/datasets/xxwu/MoLoRAG):

```bash
huggingface-cli download --repo-type dataset xxwu/MoLoRAG --local-dir ./dataset/
```

## 🔧 Environment

**For Qwen2.5-VL-series models**:
```bash
transformers==4.50.0.dev0
torch==2.6.0
qwen-vl-utils==0.0.8
```

**For remaining LVLMs, VLM retrieve, and LLM baselines**:
```bash
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

We release our fine-tuned VLM retriever, **MoLoRAG-3B**, based on the Qwen2.5-VL-3B, at [HuggingFace](https://huggingface.co/xxwu/MoLoRAG-QwenVL-3B):

```bash
huggingface-cli download xxwu/MoLoRAG-QwenVL-3B
```

The training data for fine-tuning this retriever to enable its logic-aware ability is available at [HuggingFace](https://huggingface.co/datasets/xxwu/MoLoRAG/blob/main/train_MoLoRAG_pairs_gpt4o.json). The data generation pipeline is available at [`VLMRetriever/data_collection.py`](https://github.com/WxxShirley/MoLoRAG/blob/main/VLMRetriever/data_collection.py).

## 🚀 Run

### LLM Baselines
Codes and commands are available in the [`LLMBaseline`](./LLMBaseline) directory.

### LVLM Baselines

**Step 0** - Prepare the retrieved contents following commands in [`VLMRetriever`](./VLMRetriever)

**Step 1** - Make predictions following commands in [`example_run.sh`](./example_run.sh)

**Step 2** - Evaluate the inference following commands in [`example_run_eval.sh`](./example_run_eval.sh)

## 📮 Contact

If you have any questions about usage, reproducibility, or would like to discuss, please feel free to open an issue on GitHub or contact the authors via email at [xxwu@se.cuhk.edu.hk](mailto:xxwu@se.cuhk.edu.hk)

## 🙏 Acknowledgements

We thank the open-sourced datasets, [MMLongBench](https://github.com/mayubo2333/MMLongBench-Doc/) and [LongDocURL](https://github.com/dengc2023/LongDocURL/). We also appreciate the official implementations of [M3DocRAG](https://github.com/bloomberg/m3docrag) and [MDocAgent](https://github.com/aiming-lab/MDocAgent).
