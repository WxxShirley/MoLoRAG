import torch 
from transformers import AutoModelForCausalLM
from .deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from .deepseek_vl2.utils.io import load_pil_images


def init_model(model_name, device=torch.device("cuda")):
    if "tiny" in model_name:
        model_path = "deepseek-ai/deepseek-vl2-tiny"
    elif "small" in model_name:
        model_path = "deepseek-ai/deepseek-vl2-small"

    processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = processor.tokenizer 

    model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map=device, torch_dtype=torch.bfloat16)
    model = model.eval()

    model.processor = processor
    model.tokenizer = tokenizer

    return model 


def get_response_concat(model, question, image_path_list, max_new_tokens=1024, temperature=1.0):
    if isinstance(image_path_list, list):
        text_msgs = [f"This is the {idx+1} snapshot of a document: <image>" for idx, _ in enumerate(image_path_list)]
        image_msgs = [p for p in image_path_list]
    else:
        text_msgs = ["This is a snapshot of a document: <image>\n"]
        image_msgs = [image_path_list]

    conversation = [
        {
            "role": "<|User|>",
            "content": "\n".join(text_msgs) + "\n" + question,
            "images": image_msgs
        },
        {
            "role": "<|Assistant|>", 
            "content": ""
        }
    ]

    pil_images = load_pil_images(conversation)
    prepare_inputs = model.processor(conversations=conversation, 
                                     images=pil_images, 
                                     force_batchify=True, 
                                     system_prompt="").to(model.device)
    
    input_embeds = model.prepare_inputs_embeds(**prepare_inputs)

    with torch.no_grad():
        outputs = model.language.generate(
            inputs_embeds=input_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=model.tokenizer.eos_token_id,
            bos_token_id=model.tokenizer.bos_token_id,
            eos_token_id=model.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True
        )

    answer = model.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False)

    if '<｜end▁of▁sentence｜>' in answer:
        answer = answer.split('<｜end▁of▁sentence｜>')[0]
        
    return answer
    