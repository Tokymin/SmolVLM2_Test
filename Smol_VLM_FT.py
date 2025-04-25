import datetime
import os

import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics3ForConditionalGeneration
from datasets import load_dataset
from transformers import TrainingArguments, Trainer


def load_lora_model():
    if USE_QLORA or USE_LORA:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=['down_proj', 'o_proj', 'k_proj', 'q_proj', 'gate_proj', 'up_proj', 'v_proj'],
            use_dora=False if USE_QLORA else True,
            init_lora_weights="gaussian"
        )  # 定义 LoRA 的配置参数。r表示低秩矩阵的秩，lora_alpha是缩放因子，lora_dropout是 Dropout 概率，target_modules指定要应用 LoRA 的模块，use_dora是一个与 Dora 优化相关的选项（根据USE_QLORA的值进行设置），init_lora_weights指定初始化 LoRA 权重的方式。
        lora_config.inference_mode = False  # 设置 LoRA 配置为非推理模式，意味着模型处于训练准备状态。
        if USE_QLORA:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )  # 定义低比特量化的配置。load_in_4bit表示以 4 比特精度加载模型，bnb_4bit_use_double_quant启用双重量化，bnb_4bit_quant_type指定量化类型为"nf4"，bnb_4bit_compute_dtype指定计算数据类型为torch.bfloat16。

        model = Idefics3ForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config if USE_QLORA else None,
            _attn_implementation="flash_attention_2",
            device_map="auto"
        )  # 从预训练模型加载 Idefics3ForConditionalGeneration 模型，并根据 USE_QLORA 的值配置 量化参数，设置注意力机制的实现方式为 "flash_attention_2"，并自动分配设备。
        model.add_adapter(lora_config)  # 为模型添加 LoRA 适配器。
        model.enable_adapters()
        model = prepare_model_for_kbit_training(model)  # 为低比特训练准备模型。
        model = get_peft_model(model, lora_config)  # 获取经过 LoRA 优化的模型。
        print(model.get_nb_trainable_parameters())
    else:
        model = Idefics3ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2",
        ).to(
            DEVICE)  # 从预训练模型加载Idefics3ForConditionalGeneration模型，设置数据类型为torch.bfloat16，注意力机制为"flash_attention_2"，并将模型移动到指定设备（代码中DEVICE未定义，需根据实际情况确定）。

        # if you'd like to only fine-tune LLM
        for param in model.model.vision_model.parameters():
            param.requires_grad = False

    return model


def load_data_and_training(model):
    ds = load_dataset('merve/vqav2-small', trust_remote_code=True)
    split_ds = ds["validation"].train_test_split(test_size=0.5)
    train_ds = split_ds["train"]
    image_token_id = processor.tokenizer.additional_special_tokens_ids[
        processor.tokenizer.additional_special_tokens.index("<image>")]

    def collate_fn(examples):
        texts = []
        images = []
        for example in examples:
            image = example["image"]
            if image.mode != 'RGB':
                image = image.convert('RGB')
            question = example["question"]
            answer = example["multiple_choice_answer"]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Answer briefly."},
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer}
                    ]
                }
            ]
            text = processor.apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text.strip())
            images.append([image])

        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        labels[labels == image_token_id] = -100
        batch["labels"] = labels

        return batch

    model_name = model_id.split("/")[-1]
    # 生成保存模型的路径
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_dir = f"./{model_name}-vqav2-trained-{timestamp}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    training_args = TrainingArguments(
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=50,
        learning_rate=1e-4,
        weight_decay=0.01,
        logging_steps=25,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=1,
        optim="paged_adamw_8bit",  # for 8-bit, keep this, else adamw_hf
        bf16=True,  #  underlying precision for 8bit
        output_dir=f"./{model_name}-vqav2",
        hub_model_id=f"{model_name}-vqav2",
        report_to="tensorboard",
        remove_unused_columns=False,
        gradient_checkpointing=True
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_ds,
    )
    trainer.train()
    # %%
    # trainer.push_to_hub()
    # 将模型保存到本地
    model.save_pretrained(save_dir)

if __name__ == '__main__':
    USE_LORA = False
    USE_QLORA = True  # 配置低比特量化参数
    SMOL = True
    model_id = "HuggingFaceTB/SmolVLM-Base" if SMOL else "HuggingFaceM4/Idefics3-8B-Llama3"
    processor = AutoProcessor.from_pretrained(
        model_id
    )
    model = load_lora_model()
    load_data_and_training(model)
