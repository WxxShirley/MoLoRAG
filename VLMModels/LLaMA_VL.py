import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor


def init_model(model_name, device=torch.device("cuda")):
    model_path = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    model = MllamaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device
    ).eval()

    processor = AutoProcessor.from_pretrained(model_path)
    model.processor = processor

    return model 


def get_response_concat(model, question, image_path_list, max_new_tokens=1024, temperature=1.0):
    local_img = Image.open(image_path_list if isinstance(image_path_list, str) else image_path_list[0])

    messages = [
        {
            "role": "user",
            "content": [
                { "type": "image" },
                { "type": "text", "text": question }
            ]
        }
    ]

    input_text = model.processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = model.processor(local_img, input_text, add_special_tokens=False, return_tensors='pt').to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)

    answer = model.processor.decode(output[0])

    return answer
