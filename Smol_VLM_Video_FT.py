# 此Python文件用于对SmolVLM2进行视频字幕任务的微调
# 导入所需的库
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoProcessor, BitsAndBytesConfig, AutoModelForImageTextToText
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from torch.nn.utils.rnn import pad_sequence


# 设置是否使用LoRA和QLoRA以及选择模型
# 设置SOCKS代理
os.environ['http_proxy'] = 'http://127.0.0.1:1080'
os.environ['https_proxy'] = 'socks5://127.0.0.1:1080'
os.environ['socks_proxy'] = 'socks5://127.0.0.1:1080'

USE_LORA = False
USE_QLORA = False
SMOL = False
model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct" if SMOL else "/mnt/share/toky/LLMs/SmolVLM2-2.2B-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
import torch

if torch.cuda.is_available():
    print("CUDA is available!")
else:
    print("CUDA is not available. Please check your CUDA installation.")
# 根据设置加载模型
"""根据USE_QLORA和USE_LORA的值来加载模型。若使用 QLoRA 或 LoRA，会配置相应的参数并加载模型；若不使用，则直接加载模型并将视觉模型的参数冻结。最后打印模型占用的 GPU 内存。"""
if USE_QLORA or USE_LORA:
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=['down_proj', 'o_proj', 'k_proj', 'q_proj', 'gate_proj', 'up_proj', 'v_proj'],
        use_dora=False if USE_QLORA else True,
        init_lora_weights="gaussian"
    )
    lora_config.inference_mode = False
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=bnb_config if USE_QLORA else None,
        _attn_implementation="flash_attention_2",
        device_map="auto"
    )
    model.add_adapter(lora_config)
    model.enable_adapters()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    print(model.get_nb_trainable_parameters())
else:
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
    )
    model = model.to("cuda")  # 确保模型被移动到GPU

    # 如果只想微调LLM，冻结视觉模型参数
    for param in model.model.vision_model.parameters():
        param.requires_grad = False

peak_mem = torch.cuda.max_memory_allocated()
print(f"The model as is is holding: {peak_mem / 1024 ** 3:.2f} of GPU RAM")

# 加载并预处理数据集
"""使用load_dataset加载数据集，将训练集按 0.5 的比例划分为训练集和测试集，"""
ds = load_dataset("/mnt/share/toky/Datasets/TIGER-Lab/VideoFeedback", name='real')
split_ds = ds["train"].train_test_split(test_size=0.5)
train_ds = split_ds["train"]
del split_ds, ds

# 打印数据集示例
print(f"prompt:  {train_ds[0]['text prompt']}, video: {train_ds[0]['video link']}")
"""仅保留训练集。打印数据集的一个示例，并获取<image>标记的 ID。"""
image_token_id = processor.tokenizer.additional_special_tokens_ids[
    processor.tokenizer.additional_special_tokens.index("<image>")
]


# 数据整理函数
def collate_fn(examples):
    """函数用于将多个样本整理成一个批次。它将每个样本的消息转换为模型输入，对输入 ID、注意力掩码和标签进行填充，处理视频像素值，最后返回一个包含整理后数据的字典。"""
    instances = []
    for example in examples:
        prompt = example["text prompt"]

        user_content = [{"type": "text", "text": "Caption the video."}]
        user_content.append({"type": "video", "path": example["video link"]})

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": f"{prompt}"}]}
        ]

        instance = processor.apply_chat_template(messages, add_generation_prompt=False,
                                                 tokenize=True, return_dict=True, return_tensors="pt").to(
            "cuda").to(
            model.dtype)
        instances.append(instance)

    input_ids = pad_sequence(
        [inst["input_ids"].squeeze(0) for inst in instances],
        batch_first=True,
        padding_value=processor.tokenizer.pad_token_id
    ).to("cuda")  # 确保在GPU上
    attention_mask = pad_sequence(
        [inst["attention_mask"].squeeze(0) for inst in instances],
        batch_first=True,
        padding_value=0
    ).to("cuda")  # 确保在GPU上
    labels = pad_sequence(
        [inst["input_ids"].squeeze(0).clone() for inst in instances],
        batch_first=True,
        padding_value=-100
    ).to("cuda")  # 确保在GPU上

    labels[labels == image_token_id] = -100

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

    # 处理视频像素值
    pvs = [inst["pixel_values"].squeeze(0) for inst in instances if "pixel_values" in inst]
    if pvs:
        max_frames = max(pv.shape[0] for pv in pvs)
        max_h = max(pv.shape[-2] for pv in pvs)
        max_w = max(pv.shape[-1] for pv in pvs)
    else:
        max_h = max_w = processor.video_size['longest_edge']
        max_frames = 1

    padded_pixel_values_list = []
    for ex in instances:
        pv = ex.get("pixel_values", None).squeeze(0)

        if pv is None:
            shape_pv = (max_frames, 3, max_h, max_w)
            padded_pv = torch.zeros(shape_pv, dtype=torch.float32).to("cuda")  # 确保在GPU上
        else:
            f, c, h, w = pv.shape
            padded_pv = torch.zeros(
                (max_frames, c, max_h, max_w),
                dtype=pv.dtype,
                device=pv.device
            )
            padded_pv[:f, :, :h, :w] = pv
        padded_pixel_values_list.append(padded_pv)

    out["pixel_values"] = torch.stack(padded_pixel_values_list, dim=0).to("cuda")  # 确保在GPU上
    return out


# 训练设置
model_name = model_id.split("/")[-1]
training_args = TrainingArguments(
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    warmup_steps=50,
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_steps=25,
    save_strategy="steps",
    save_steps=250,
    save_total_limit=1,
    optim="paged_adamw_8bit", # for 8-bit, keep paged_adamw_8bit, else adamw_hf
    bf16=True,
    output_dir=f"./{model_name}-video-feedback",
    hub_model_id=f"{model_name}-video-feedback",
    remove_unused_columns=False,
    report_to="tensorboard",
    dataloader_pin_memory=False
)

# 初始化Trainer并进行训练
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_ds,
)
trainer.train()
# 将训练好的模型推送到Hugging Face Hub
trainer.push_to_hub()

# 测试示例
messages = [{"role": "user",
             "content": [{"type": "text", "text": "Caption the video."},
                         {"type": "video",
                          "path": "https://huggingface.co/datasets/hexuan21/VideoFeedback-videos-mp4/resolve/main/p/p000304.mp4"}]}]

inputs = processor.apply_chat_template(messages, add_generation_prompt=True,
                                       tokenize=True, return_dict=True, return_tensors="pt").to("cuda").to(model.dtype)

generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)

print(generated_texts[0])
