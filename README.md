# MoLoRAG

<p align="center">
   <a href="https://arxiv.org/abs/2509.07666"><img src="https://img.shields.io/badge/📝-Paper-blue" height="25"></a>
   <a href="https://huggingface.co/datasets/xxwu/MoLoRAG"><img src="https://img.shields.io/badge/🤗-Dataset-green" height="25"></a>
   <a href="https://huggingface.co/xxwu/MoLoRAG-QwenVL-3B"><img src="https://img.shields.io/badge/🚀-Model-yellow" height="25"></a>
</p>


This repository is the official implementation for our EMNLP 2025 paper: **MoLoRAG: Bootstrapping Document Understanding via Multi-modal Logic-aware Retrieval**. Our paper tackles the DocQA task by addressing the limitations of prior methods that rely only on semantic relevance for retrieval. By incorporating logical relevance, our VLM-powered retrieval
engine performs **multi-hop reasoning over page graph** to identify key pages.

> Please consider citing or giving a 🌟 if our repository is helpful to your work!

```
@inproceedings{wu2025molorag
   title={MoLoRAG: Bootstrapping Document Understanding via Multi-modal Logic-aware Retrieval},
   author={Xixi Wu and Yanchao Tan and Nan Hou and Ruiyang Zhang and Hong Cheng},
   year={2025},
   booktitle={The 2025 Conference on Empirical Methods in Natural Language Processing},
   url={https://arxiv.org/abs/2509.07666},
}
```

### 🎙️ News 

🎉 **[2025-08-24]** Our paper is accepted to **EMNLP 2025**. The [camera ready paper](https://arxiv.org/abs/2509.07666) and fully reviewed codes will be released soon!

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

> The full package versions can be found in `env/main.txt` and `env/qwenvl.txt`, respectively. Please refer to these files for detailed package versions.

**For Qwen2.5-VL-series models**:
```bash
transformers==4.50.0.dev0
xformers==0.0.29.post3
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
PyMuPDF==1.25.3
pypdf==5.3.0
pypdfium2==4.30.1
pdf2image==1.17.0
```

## 🤗 Model

We release our fine-tuned VLM retriever, **MoLoRAG-3B**, based on the Qwen2.5-VL-3B, at [HuggingFace](https://huggingface.co/xxwu/MoLoRAG-QwenVL-3B):

```bash
huggingface-cli download xxwu/MoLoRAG-QwenVL-3B
```

The training data for fine-tuning this retriever to enable its logic-aware ability is available at [HuggingFace](https://huggingface.co/datasets/xxwu/MoLoRAG/blob/main/train_MoLoRAG_pairs_gpt4o.json). The data generation pipeline is available at [`VLMRetriever/data_collection.py`](https://github.com/WxxShirley/MoLoRAG/blob/main/VLMRetriever/data_collection.py).

## 🚀 Run

> Before running the code, please check if you need to **fill in the API Keys** or **prepare the model/data**

### LLM Baselines
Codes and commands are available in the [`LLMBaseline`](./LLMBaseline) directory.

### LVLM Baselines

**Step 0** - Prepare the retrieved contents following commands in [`VLMRetriever`](./VLMRetriever)

**Step 1** - Make predictions following commands in [`example_run.sh`](./example_run.sh)

**Step 2** - Evaluate the inference following commands in [`example_run_eval.sh`](./example_run_eval.sh)

## ✏️ TODO 

- [ ] Provide tailored MDocAgent code
- [ ] Provide detailed scripts or running tutorials

## 📮 Contact

If you have any questions about usage, reproducibility, or would like to discuss, please feel free to open an issue on GitHub or contact the authors via email at [xxwu@se.cuhk.edu.hk](mailto:xxwu@se.cuhk.edu.hk)

## 🙏 Acknowledgements

We thank the open-sourced datasets, [MMLongBench](https://github.com/mayubo2333/MMLongBench-Doc/), [LongDocURL](https://github.com/dengc2023/LongDocURL/), [UDA-Benchmark](https://github.com/qinchuanhui/UDA-Benchmark). We also appreciate the official implementations of [M3DocRAG](https://github.com/bloomberg/m3docrag) and [MDocAgent](https://github.com/aiming-lab/MDocAgent).
