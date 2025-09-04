import torch
import json
import os
from transformers import AutoProcessor, AutoModelForImageTextToText
from typing import Dict, List


def load_smolvlm_model(model_path: str = "/mnt/share/HuggingfaceModels/HuggingFaceTB/SmolVLM2-2.2B-Instruct"):
    """加载SmolVLM2模型和处理器（遵循参考代码格式）"""
    print(f"正在加载模型：{model_path}")
    # 加载处理器（参考代码无local_files_only，若本地模型需添加可补充）
    processor = AutoProcessor.from_pretrained(model_path)
    # 加载模型（匹配参考代码的dtype和device_map）
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda"  # 自动分配GPU设备
    )
    model.eval()  # 推理模式，禁用Dropout
    print("模型和处理器加载完成！")
    return model, processor


def parse_jsonl_data(
        data_path: str = "/media/user/data3/toky/Projects/Evaluate-Surgery-VLM/split_data/generated_long_questions_with_response.jsonl"):
    """解析JSONL数据，提取图像路径、提示文本等关键信息"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在：{data_path}")

    samples: List[Dict] = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                sample = json.loads(line.strip())
                # 提取必要字段（匹配JSONL样本结构）
                required_fields = ["question_id", "image", "text", "response"]
                if not all(field in sample for field in required_fields):
                    print(f"第{line_num}行样本缺少必要字段，跳过")
                    continue
                samples.append(sample)
            except json.JSONDecodeError:
                print(f"第{line_num}行JSON格式错误，跳过")
                continue
    print(f"成功解析 {len(samples)} 个有效样本")
    return samples


def process_image_samples(model, processor, samples: List[Dict], output_path: str = "smolvlm_image_test_output.jsonl"):
    """处理图像样本，生成模型输出并保存"""
    print(f"\n开始处理样本，结果将保存至：{output_path}")
    with open(output_path, "w", encoding="utf-8") as outfile:
        for idx, sample in enumerate(samples, 1):
            question_id = sample["question_id"]
            image_path = sample["image"]
            prompt_text = sample["text"]
            ground_truth = sample["response"]

            try:
                # 1. 检查图像文件是否存在
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"图像路径不存在：{image_path}")

                # 2. 构建对话（完全遵循参考代码的图像对话格式）
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "path": image_path},  # 参考代码：直接指定图像路径
                            {"type": "text", "text": prompt_text}  # 样本中的提示文本
                        ]
                    }
                ]

                # 3. 处理输入（匹配参考代码的apply_chat_template参数）
                inputs = processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,  # 自动添加助手回复起始标记
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device, dtype=torch.bfloat16)  # 移到GPU并匹配数据类型

                # 4. 生成模型输出（参考代码参数+合理长度限制）
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=128,  # 参考代码默认128，可根据需求调整
                    do_sample=False,  # 确定性生成，避免随机
                    temperature=0.0  # 温度=0，结果更稳定
                )

                # 5. 解码生成结果（跳过特殊Token）
                model_response = processor.batch_decode(
                    output_ids,
                    skip_special_tokens=True
                )[0]  # 取第一个样本结果（batch_size=1）

                # 6. 构建输出数据（包含原始信息和模型结果）
                output_data = {
                    "question_id": question_id,
                    "image_path": image_path,
                    "prompt_text": prompt_text,
                    "ground_truth_response": ground_truth,
                    "smolvlm_model_response": model_response,
                    "processing_status": "success"
                }

                # 7. 写入输出文件
                outfile.write(json.dumps(output_data, ensure_ascii=False) + "\n")

                # 打印进度
                print(f"已处理 {idx}/{len(samples)} 个样本 | Question ID: {question_id} | 处理成功")

            except Exception as e:
                # 错误处理：记录失败信息
                error_data = {
                    "question_id": question_id,
                    "image_path": image_path,
                    "prompt_text": prompt_text,
                    "ground_truth_response": ground_truth,
                    "smolvlm_model_response": "",
                    "processing_status": "failed",
                    "error_message": str(e)
                }
                outfile.write(json.dumps(error_data, ensure_ascii=False) + "\n")
                print(f"已处理 {idx}/{len(samples)} 个样本 | Question ID: {question_id} | 处理失败：{str(e)}")

    print(f"\n所有样本处理完成！结果文件：{os.path.abspath(output_path)}")


def main():
    # 1. 加载模型
    model, processor = load_smolvlm_model(
        model_path="/mnt/share/HuggingfaceModels/HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    )

    # 2. 解析JSONL数据
    samples = parse_jsonl_data(
        data_path="/media/user/data3/toky/Projects/Evaluate-Surgery-VLM/split_data/generated_long_questions_with_response.jsonl"
    )

    # 3. 处理样本并保存输出
    process_image_samples(
        model=model,
        processor=processor,
        samples=samples,
        output_path="smolvlm_image_test_output.jsonl"  # 输出文件路径（可自定义）
    )


if __name__ == "__main__":
    main()