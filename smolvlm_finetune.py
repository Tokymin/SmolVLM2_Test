import torch
from PIL import Image
import os
from datasets import load_dataset
import bitsandbytes as bnb
from bitsandbytes.optim import AdamW8bit
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model


# -------------------------- 1. 环境检查与基础配置 --------------------------
def check_dependencies():
    import bitsandbytes as bnb
    import peft
    import transformers
    print(f"peft版本: {peft.__version__}")
    print(f"transformers版本: {transformers.__version__}")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"使用数据类型: {dtype}")
    return dtype


dtype = check_dependencies()

# 关键路径
model_id = "/mnt/share/HuggingfaceModels/HuggingFaceTB/SmolVLM2-2.2B-Instruct"
train_data_path = "/media/user/data3/toky/Projects/Evaluate-Surgery-VLM/split_data/train.json"
IMAGE_FIELD = "frame_path"

# -------------------------- 2. 加载模型与处理器 --------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=dtype
)

processor = AutoProcessor.from_pretrained(
    model_id,
    local_files_only=True,
    trust_remote_code=True
)
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    print("已设置pad_token为eos_token")

model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=dtype
)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=['down_proj', 'o_proj', 'k_proj', 'q_proj', 'gate_proj', 'up_proj', 'v_proj'],
    bias="none",
    init_lora_weights="gaussian",
    inference_mode=False
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# -------------------------- 3. 数据处理 --------------------------
def load_and_preprocess_data(data_path: str):
    dataset = load_dataset(
        "json",
        data_files=data_path,
        split="train",
        num_proc=1
    )
    print(f"原始数据集：{len(dataset)}条样本，字段：{dataset.column_names}")

    required_fields = [IMAGE_FIELD, "prompt", "response"]
    missing_fields = [f for f in required_fields if f not in dataset.column_names]
    if missing_fields:
        raise ValueError(f"数据集缺少字段：{missing_fields}（图像路径字段应为{IMAGE_FIELD}）")

    def preprocess_single_sample(example):
        img_path = example[IMAGE_FIELD]
        prompt = example["prompt"]
        response = example["response"]

        # -------------------------- 1. 图像加载（原有逻辑不变） --------------------------
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图像不存在：{img_path}")
        try:
            with Image.open(img_path) as img:
                if hasattr(img, 'n_frames') and img.n_frames > 1:
                    img.seek(0)  # 取第1帧
                image = img.convert("RGB")
        except Exception as e:
            raise RuntimeError(f"加载图像{img_path}失败：{str(e)}")
        print(f"样本：{img_path.split('/')[-2:]}, 原始尺寸={image.size}, 取1帧处理")

        # -------------------------- 2. 文本处理（原有逻辑不变，81个<image>标记） --------------------------
        clean_prompt = prompt.replace("<image>", "").replace("\r", "").strip()
        num_image_marks = 81  # 确保81%81=0
        image_tokens = "<image>" * num_image_marks
        text = f"{image_tokens}\n{clean_prompt}\n{response}"

        if text.count("<image>") != num_image_marks:
            raise ValueError(f"文本含{text.count('<image>')}个<image>标记，需{num_image_marks}个")

        max_text_length = 1024 + num_image_marks
        text_inputs = processor.tokenizer(
            text,
            truncation=True,
            max_length=max_text_length,
            padding="max_length",
            return_tensors="pt",
            return_attention_mask=True
        )
        input_ids = text_inputs["input_ids"].squeeze(0)
        attention_mask = text_inputs["attention_mask"].squeeze(0)

        # -------------------------- 3. 图像处理（核心：添加num_images=1维度） --------------------------
        image_inputs = processor.image_processor(image, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].squeeze(0)  # 处理器输出：[13,3,384,384]→切片前
        pixel_values = pixel_values[0, :, :, :]  # 切片取第1帧：[3, 384, 384]（3维）

        # 核心修复：添加num_images=1维度，从3维→4维：[1, 3, 384, 384]
        pixel_values = pixel_values.unsqueeze(0)  # 这行是新增的！

        # 验证维度（预处理后是4维，DataLoader加载后会变成5维）
        assert pixel_values.ndim == 4, \
            f"预处理后pixel_values需为4维(1,3,H,W)，当前形状={pixel_values.shape}"
        assert pixel_values.shape == (1, 3, 384, 384), \
            f"预处理后pixel_values形状需为(1,3,384,384)，当前={pixel_values.shape}"
        print(f"Step1: 预处理后pixel_values形状 = {pixel_values.shape}")  # 输出：[1,3,384,384]

        # -------------------------- 4. 生成labels（原有逻辑不变） --------------------------
        labels = input_ids.clone()
        pad_token_id = processor.tokenizer.pad_token_id
        labels[labels == pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,  # 返回4维：[1,3,384,384]
            "labels": labels
        }

    # -------------------------- 后续数据集处理（原有逻辑不变） --------------------------
    tokenized_dataset = dataset.map(
        preprocess_single_sample,
        batched=False,
        remove_columns=dataset.column_names,
        num_proc=1,
        load_from_cache_file=False
    )

    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "pixel_values", "labels"],
    )

    print(f"\n预处理完成：{len(tokenized_dataset)}条样本")
    print(f"示例样本形状：")
    print(f"- input_ids: {tokenized_dataset[0]['input_ids'].shape}")  # 输出：torch.Size([1105])（1024+81）
    print(f"- pixel_values: {tokenized_dataset[0]['pixel_values'].shape}")  # 输出：torch.Size([3, 384, 384])（单帧）

    return tokenized_dataset


train_dataset = load_and_preprocess_data(train_data_path)

# -------------------------- 4. 训练配置 --------------------------
training_args = TrainingArguments(
    output_dir="./smolvlm2-finetuned-final",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    fp16=True,
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=2,
    optim="paged_adamw_8bit",  # 让Trainer自动创建优化器
    lr_scheduler_type="linear",  # 指定学习率调度器类型
    warmup_ratio=0.05,  # 添加warmup
    report_to="none",
    remove_unused_columns=False,
    label_names=["labels"]
)

# -------------------------- 5. 启动训练 --------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=processor.tokenizer,
    # 不再手动传入optimizer和lr_scheduler
)

print("\n✅ 预处理完成，开始训练...")
trainer.train()

# 保存最终LoRA模型
model.save_pretrained("./smolvlm2-lora-final")
processor.save_pretrained("./smolvlm2-lora-final")
print("\n训练完成！模型保存至 ./smolvlm2-lora-final")