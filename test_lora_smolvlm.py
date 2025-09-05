import torch
import json
import os
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

# -------------------------- 1. 测试配置（需根据你的路径修改） --------------------------
# 与训练一致的关键参数（必须和训练代码完全相同）
IMAGE_FIELD_TEST = "image"  # 测试集JSONL中图像路径的字段名（对应训练的IMAGE_FIELD="frame_path"）
NUM_IMAGE_MARKS = 81  # 与训练一致：81个<image>标记
MAX_TEXT_LENGTH = 1024 + NUM_IMAGE_MARKS  # 与训练一致：1024文本Token + 81图像Token

# 路径配置
BASE_MODEL_PATH = "/mnt/share/HuggingfaceModels/HuggingFaceTB/SmolVLM2-2.2B-Instruct"  # 同训练
LORA_MODEL_PATH = "smolvlm2-lora-final-20250904"  # 训练好的LoRA模型路径
TEST_DATA_PATH = "/media/user/data3/toky/Projects/Evaluate-Surgery-VLM/split_data/generated_long_questions_with_response.jsonl"  # 测试集路径
OUTPUT_PATH = "smolvlm_test_results.jsonl"  # 测试结果输出路径


# -------------------------- 2. 加载测试用模型/处理器（仅加载，不训练） --------------------------
def load_test_model_processor(base_path, lora_path):
    """加载训练好的LoRA模型+处理器（完全对齐训练时的配置）"""
    # 1. 加载处理器（与训练一致：trust_remote_code=True + pad_token设置）
    processor = AutoProcessor.from_pretrained(
        base_path,
        local_files_only=True,
        trust_remote_code=True
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token  # 同训练
        print("✅ 已设置pad_token为eos_token（与训练一致）")

    # 2. 加载Base模型（与训练一致的量化配置）
    model = AutoModelForImageTextToText.from_pretrained(
        base_path,
        quantization_config=dict(  # 复刻训练时的BitsAndBytesConfig
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ),
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )

    # 3. 融合LoRA权重（测试时仅加载，不训练）
    model = PeftModel.from_pretrained(
        model=model,
        model_id=lora_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    model.eval()  # 测试关键：切换为推理模式（禁用Dropout）
    print("✅ 模型加载完成（Base+LoRA，推理模式）")
    return model, processor


# -------------------------- 3. 解析测试集（仅读取测试数据） --------------------------
def parse_test_data(data_path):
    """解析测试集JSONL，过滤无效样本（与训练一致的字段检查）"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"测试集不存在：{data_path}")

    valid_samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                sample = json.loads(line.strip())
                # 与训练一致：检查必要字段（测试集用"text"对应训练的"prompt"）
                required_fields = [IMAGE_FIELD_TEST, "text", "response"]
                if not all(f in sample for f in required_fields):
                    print(f"⚠️  第{line_num}行样本缺少字段，跳过")
                    continue
                valid_samples.append(sample)
            except json.JSONDecodeError:
                print(f"⚠️  第{line_num}行JSON格式错误，跳过")
                continue
    print(f"✅ 解析完成：共{len(valid_samples)}个有效测试样本")
    return valid_samples


# -------------------------- 4. 测试样本预处理（100%复刻训练的preprocess_single_sample） --------------------------
def preprocess_test_sample(sample, processor):
    img_path = sample[IMAGE_FIELD_TEST]
    prompt_text = sample["text"]
    question_id = sample.get("question_id", f"sample_{id(sample)}")

    # -------------------------- 图像处理（无修改） --------------------------
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"样本{question_id}：图像不存在：{img_path}")
    try:
        with Image.open(img_path) as img:
            if hasattr(img, 'n_frames') and img.n_frames > 1:
                img.seek(0)
            image = img.convert("RGB")
    except Exception as e:
        raise RuntimeError(f"样本{question_id}：加载图像失败：{str(e)}") from e

    image_inputs = processor.image_processor(image, return_tensors="pt")
    pixel_values = image_inputs["pixel_values"].squeeze(0)
    pixel_values = pixel_values[0, :, :, :]
    pixel_values = pixel_values.unsqueeze(0)
    assert pixel_values.shape == (1, 3, 384, 384), \
        f"样本{question_id}：图像形状错误！需(1,3,384,384)，当前{pixel_values.shape}"

    # -------------------------- 文本处理（核心优化：避免<image>被合并） --------------------------
    # 1. 清理prompt：移除所有可能破坏<image>的特殊字符（比之前更彻底）
    clean_prompt = prompt_text.replace("<image>", "").replace("\r", "").strip()
    # 新增：移除所有非字母数字/中文的字符，避免干扰<image>标记
    clean_prompt = ''.join([c for c in clean_prompt if c.isalnum() or c.isspace() or '\u4e00' <= c <= '\u9fff'])

    # 2. 生成<image>标记：用空格分隔每个<image>，避免Tokenizer合并连续相同Token
    # 关键：训练时文本是连续<image>，但测试时连续可能被合并，用空格分隔确保每个都被识别
    num_image_marks = NUM_IMAGE_MARKS
    image_tokens = " <image> " * num_image_marks  # 每个<image>前后加空格
    image_tokens = image_tokens.strip()  # 去除首尾空格，避免多余Token

    # 3. 构建文本：保持格式对齐（空格分隔的<image> + 清理后的prompt）
    text = f"{image_tokens}\n{clean_prompt}"

    # 4. 验证编码前<image>数量（确保生成正确）
    current_image_count = text.count("<image>")
    if current_image_count != num_image_marks:
        raise ValueError(
            f"样本{question_id}：编码前<image>数量错误！需{num_image_marks}个，当前{current_image_count}个"
        )

    # 5. 文本编码（同训练参数）
    text_inputs = processor.tokenizer(
        text,
        truncation=True,
        max_length=MAX_TEXT_LENGTH,
        padding="max_length",
        return_tensors="pt",
        return_attention_mask=True
    )

    # -------------------------- 新增：强制验证编码后<image>数量（核心拦截错误） --------------------------
    # 获取<image>的Token ID（每次都重新获取，避免映射变化）
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    if image_token_id == processor.tokenizer.unk_token_id:
        raise ValueError(f"样本{question_id}：<image>不是有效Token（未知Token ID）")

    # 统计编码后<image>的数量
    encoded_image_count = (text_inputs["input_ids"] == image_token_id).sum().item()
    if encoded_image_count != num_image_marks:
        raise ValueError(
            f"样本{question_id}：编码后<image>丢失！编码前={current_image_count}个，编码后={encoded_image_count}个"
            f"\n  文本原文前100字符：{text[:100]}"  # 打印部分原文，帮助定位问题
            f"\n  编码后前20个Token ID：{text_inputs['input_ids'][0][:20]}"  # 打印部分Token ID，查看是否合并
        )

    # 后续处理（无修改）
    input_ids = text_inputs["input_ids"].squeeze(0)
    attention_mask = text_inputs["attention_mask"].squeeze(0)
    assert input_ids.shape == (MAX_TEXT_LENGTH,), \
        f"样本{question_id}：文本长度错误！需{MAX_TEXT_LENGTH}，当前{input_ids.shape[0]}"

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "sample_info": {
            "question_id": question_id,
            "image_path": img_path,
            "original_prompt": prompt_text,
            "ground_truth": sample["response"],
            "encoded_image_count": encoded_image_count  # 记录编码后数量，便于调试
        }
    }

# -------------------------- 5. 执行测试推理（核心：仅测试，无训练） --------------------------
def run_test_inference(model, processor, test_samples):
    device = model.device
    model_dtype = model.dtype
    print(f"\n🚀 开始测试推理（共{len(test_samples)}个样本，设备：{device}，模型类型：{model_dtype}）")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for idx, sample in enumerate(test_samples, 1):
            # 用样本自带的question_id，避免未知ID
            question_id = sample.get("question_id", f"unknown_{idx}")
            try:
                # -------------------------- 关键：预处理时已验证编码后<image>数量为81 --------------------------
                preprocessed = preprocess_test_sample(sample, processor)
                # 打印编码后<image>数量，确认正常
                print(f"ℹ️  样本{idx}（ID:{question_id}）：编码后<image>数量={preprocessed['sample_info']['encoded_image_count']}")

                # 类型转换与推理（无修改）
                input_ids = preprocessed["input_ids"].to(device)
                attention_mask = preprocessed["attention_mask"].to(device)
                pixel_values = preprocessed["pixel_values"].to(device, dtype=model_dtype)

                with torch.no_grad():
                    # 修复警告：do_sample=False时无需设置temperature，直接删除temperature参数
                    outputs = model.generate(
                        input_ids=input_ids.unsqueeze(0),
                        attention_mask=attention_mask.unsqueeze(0),
                        pixel_values=pixel_values.unsqueeze(0),
                        max_new_tokens=128,
                        do_sample=False  # 删除temperature=0.0，避免警告
                    )

                # 解码与保存（无修改）
                model_response = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
                input_prefix = f"{(' <image> ' * NUM_IMAGE_MARKS).strip()}\n{preprocessed['sample_info']['original_prompt'].replace('<image>', '').strip()}"
                if model_response.startswith(input_prefix):
                    model_response = model_response[len(input_prefix):].strip()

                result = {
                    "question_id": preprocessed["sample_info"]["question_id"],
                    "image_path": preprocessed["sample_info"]["image_path"],
                    "original_prompt": preprocessed["sample_info"]["original_prompt"],
                    "ground_truth": preprocessed["sample_info"]["ground_truth"],
                    "model_response": model_response,
                    "status": "success"
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                print(f"✅ 完成 {idx}/{len(test_samples)} | ID: {question_id}")

            except Exception as e:
                # 错误日志中包含编码后<image>数量（如果已预处理）
                encoded_count = preprocessed["sample_info"]["encoded_image_count"] if 'preprocessed' in locals() else "未知"
                error_result = {
                    "question_id": question_id,
                    "image_path": sample.get(IMAGE_FIELD_TEST, ""),
                    "original_prompt": sample.get("text", "")[:50] + "..." if sample.get("text") else "",  # 打印部分prompt
                    "ground_truth": sample.get("response", "")[:30] + "..." if sample.get("response") else "",
                    "model_response": "",
                    "status": "failed",
                    "encoded_image_count": encoded_count,  # 记录编码后数量
                    "error": str(e)
                }
                f.write(json.dumps(error_result, ensure_ascii=False) + "\n")
                print(f"❌ 失败 {idx}/{len(test_samples)} | ID: {question_id} | 编码后<image>数：{encoded_count} | 错误：{str(e)}")

    print(f"\n🎉 测试完成！结果保存至：{os.path.abspath(OUTPUT_PATH)}")

# -------------------------- 6. 测试入口（仅执行测试流程） --------------------------
if __name__ == "__main__":
    try:
        # 1. 加载模型/处理器
        model, processor = load_test_model_processor(BASE_MODEL_PATH, LORA_MODEL_PATH)
        # 2. 解析测试数据
        test_samples = parse_test_data(TEST_DATA_PATH)
        # 3. 执行测试推理
        run_test_inference(model, processor, test_samples)
    except Exception as e:
        print(f"\n❌ 测试启动失败：{str(e)}")