# 🚀 逐步复现指南（中文版）

由于近期有一些关于可复现性的疑问，我们已经在一台全新机器（NVIDIA A100 GPU）上**完整验证**了整条流程。性能不一致通常来自**环境不匹配、硬编码的本地路径、或检索准确率下降**。

请**严格按照以下步骤**复现论文中报告的结果。


## 1. 环境配置

强烈建议创建一个全新的 Conda 环境，使用 `python=3.10`。请**按照下列顺序**依次安装依赖（顺序很重要）：

```bash
conda create -n molorag python=3.10 -y
conda activate molorag 

# 1. 安装 Transformers & PyTorch
pip install transformers==4.50.1
pip install torch==2.6.0 
pip install torchvision==0.21.0
pip install qwen-vl-utils==0.0.8 
pip install xformers==0.0.29.post3

# 2. 安装 PDF 处理相关依赖
pip install pdf2image==1.17.0 PyMuPDF==1.25.3 pypdf==5.3.0 pypdfium2==4.30.1

# 3. 安装 ColPali（必须最后安装）
# 注意：这里可能会出现依赖警告/错误，但一般不会影响实际运行。
pip install colpali_engine==0.3.8
``` 


## 2. 下载数据集与模型

从 HuggingFace 下载数据集，以及所需的基础模型 `Qwen2.5-VL-3B-Instruct`：

```bash
# 下载数据集
huggingface-cli download --repo-type dataset xxwu/MoLoRAG --local-dir ./dataset/

# 下载基础模型
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --repo-type model
```


## 3. 建索引与基线实验（M3DocRAG）

首先，我们通过将文档页面渲染为本地图片并保存，然后使用 `Colpali` 对图片进行编码，从而构建索引。这里以 `MMLong` 为示例：

```bash
# ⚠️ 提示：强烈建议使用 --save_img！
# 请检查 tmp/tmp_imgs/MMLong/{doc-name}-{page-num}.png，确保渲染出的图片清晰可读。
python3 index.py --dataset MMLong --save_img
```

运行基线方法（M3DocRAG）并进行评估：

```bash
python3 retrieve.py --dataset MMLong --method base 
cd ../evaluate 
python3 eval_rag.py --dataset MMLong
```

如果环境配置正确，你得到的 Top-1 到 Top-5 的 Recall、Precision、NDCG、MRR 等指标应当与论文中报告的 M3DocRAG 基线结果**非常接近**。


## 4. 运行 MoLoRAG 

⚠️ **重要：必须修改硬编码的图片路径！**

在运行 MoLoRAG 之前，你必须把图片文件路径改为你本机环境的实际路径。打开 `VLMModels/Qwen_VL.py`，修改 `format_image_path` 函数：

```python
def format_image_path(raw_path):
    local_path = raw_path[1:]
    # ❌ 旧写法: format_path = f"file:///data/xxwu/BeamSearchRAG{local_path}"
    # ✅ 新写法: 改为你本机项目所在的绝对路径：
    format_path = f"file:///YOUR/ACTUAL/ABSOLUTE/PATH/MoLoRAG{local_path}"
    return format_path
```

如果不修改，VLM 可能会读到空图片或不存在的图片，从而导致性能**大幅下降**！


现在运行 MoLoRAG 检索：

```shell
python3 retrieve.py --dataset MMLong --method beamsearch --model_name QwenVL-3B
```

> **排查 flash_attention 报错：**
> 如果你的硬件不支持 Flash Attention 2，可能会报错。解决方式是在代码的 `init_model` 函数中，把 `attn_implementation` 那一行注释掉：
> ```
> model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
>     model_path,
>     torch_dtype=torch.bfloat16,
>     # attn_implementation="flash_attention_2",  <-- 注释这一行
>     device_map=device).eval()
> 


## 5. 预期输出与评估

运行过程中，你会在控制台看到 beam search 的过程日志（Initial Beam -> Current Beam -> Prediction）。运行结束后，再次执行评估脚本（eval_rag.py）。此时你的检索性能应当与论文中报告的 MoLoRAG 结果一致或非常接近！

