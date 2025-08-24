from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch


def init_model(model_name, device=torch.device("cuda")):
    if model_name == "LLaVA-Next-7B":
        model_path = "llava-hf/llava-v1.6-mistral-7b-hf"
    elif model_name == "LLaVA-Next-8B":
        model_path = "llava-hf/llama3-llava-next-8b-hf"
        
    processor = LlavaNextProcessor.from_pretrained(model_path)

    model = LlavaNextForConditionalGeneration.from_pretrained(model_path, 
                                                              torch_dtype=torch.float16,
                                                              low_cpu_mem_usage=True, 
                                                              device_map=device,
                                                              use_flash_attention_2=True).eval()
    
    model.processor = processor

    return model 


def get_response_concat(model, question, image_path_list, max_new_tokens=1024, temperature=1.0):
    conversation = [
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": question},
                {"type": "image", "image": image_path_list if isinstance(image_path_list, str) else image_path_list[0]}
            ]
        }
    ]

    inputs = model.processor.apply_chat_template(conversation, 
                                                 add_generation_prompt=True, 
                                                 tokenize=True, 
                                                 return_dict=True, 
                                                 return_tensors="pt").to(model.device)
    
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    response = model.processor.decode(output[0], skip_special_tokens=True)
    
    if "[/INST]" in response:
        return response.split("[/INST]")[1]
    
    return response
