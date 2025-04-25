import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("/mnt/share/toky/LLMs/SmolVLM2-2.2B-Instruct")
model = AutoModelForImageTextToText.from_pretrained(
    "/mnt/share/toky/LLMs/SmolVLM2-2.2B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
print(model)
# 图像的
conversation = [
    {
        "role": "user",
        "content":[
            {"type": "image", "path": "1671158065375.jpg"},
            {"type": "text", "text": "Describe this image."}
        ]
    }
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=torch.bfloat16)

output_ids = model.generate(**inputs, max_new_tokens=128)
generated_texts = processor.batch_decode(output_ids, skip_special_tokens=True)
print(generated_texts)


# 视频
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "video", "path": "sugiral_video.mp4"},
            {"type": "text", "text": "Describe this video in detail"}
        ]
    },
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=torch.bfloat16)

generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=1000)
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_texts[0])
