# 🚀 Step-by-Step Reproduction Guide


Due to some recent questions regarding reproducibility, we have completely verified the pipeline on a fresh machine (NVIDIA A100 GPUs). The performance inconsistency usually stems from **environment mismatches, hardcoded local paths, or retrieval accuracy drops**. 

Please strictly follow the steps below to reproduce the results reported in the paper.


## 1. Environment Setup
We highly recommend creating a fresh Conda environment with `python=3.10`. Please install the packages **in the exact sequence below**:

```bash
conda create -n molorag python=3.10 -y
conda activate molorag 

# 1. Install Transformers & PyTorch
pip install transformers==4.50.1
pip install torch==2.6.0 
pip install torchvision==0.21.0
pip install qwen-vl-utils==0.0.8 
pip install xformers==0.0.29.post3

# 2. Install PDF processing dependencies
pip install pdf2image==1.17.0 PyMuPDF==1.25.3 pypdf==5.3.0 pypdfium2==4.30.1

# 3. Install ColPali (Must be installed last)
# Note: You might see dependency warnings/errors here, but they will not affect the execution.
pip install colpali_engine==0.3.8
``` 


## 2. Download Dataset & Models

Download the dataset and the required base model `Qwen2.5-VL-3B-Instruct` from HuggingFace:
```bash
# Download Dataset
huggingface-cli download --repo-type dataset xxwu/MoLoRAG --local-dir ./dataset/

# Download Base Model
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --repo-type model
```


## 3. Indexing & Baseline [M3DocRAG]

First, we construct the index by saving document images locally and encoding them using `Colpali`. We use `MMLong` as an example:

```bash
# ⚠️ TIP: The `--save_img` flag is highly recommended! 
# Please check `tmp/tmp_imgs/MMLong/{doc-name}-{page-num}.png` to ensure the rendered images are clear.
python3 index.py --dataset MMLong --save_img
```

Run the baseline (M3DocRAG) and evaluate:
```bash
python3 retrieve.py --dataset MMLong --method base 
cd ../evaluate 
python3 eval_rag.py --dataset MMLong
```
If your environment is set up correctly, the Top-1 to Top-5 Recall, Precision, NDCG, and MRR should **closely align** with the M3DocRAG baseline reported in our paper. 


## 4. Run MoLoRAG 

⚠️ **IMPORTANT: Modify the Hardcoded Image Path!**
Before running MoLoRAG, you MUST update the image file path to match your local environment. Open `VLMModels/Qwen_VL.py` and modify the `format_image_path` function:

```python
def format_image_path(raw_path):
    local_path = raw_path[1:]
    # ❌ OLD: format_path = f"file:///data/xxwu/BeamSearchRAG{local_path}"
    # ✅ NEW: Change to your actual absolute path where the project is located:
    format_path = f"file:///YOUR/ACTUAL/ABSOLUTE/PATH/MoLoRAG{local_path}"
    return format_path
```
Failing to do this will cause the VLM to read empty or missing images, leading to terrible performance!


Now, run MoLoRAG retrieval:

```shell
python3 retrieve.py --dataset MMLong --method beamsearch --model_name QwenVL-3B
```

> **Troubleshooting flash_attention errors:**
> If your hardware does not support Flash Attention 2, you may encounter an error. Simply comment out the attn_implementation line in the init_model function inside your code:
> ```
> model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
>     model_path,
>     torch_dtype=torch.bfloat16,
>     # attn_implementation="flash_attention_2",  <-- COMMENT THIS LINE OUT
>     device_map=device).eval()
> 


## 5. Expected Output & Evaluation

During execution, you will see the beam search process in your console (Initial Beam -> Current Beam -> Prediction). Once finished, rerun the evaluation script (eval_rag.py). Your retrieval performance should now match the MoLoRAG results reported in the paper!


