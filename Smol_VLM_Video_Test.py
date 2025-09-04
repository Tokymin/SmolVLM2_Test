"""这个脚本用来测试微调后的模型并计算一下评价指标"""
import re

import torch
from transformers import AutoProcessor, BitsAndBytesConfig, AutoModelForImageTextToText
from transformers import AutoModelForCausalLM, AutoProcessor, TrainingArguments, Trainer
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hamming_loss, jaccard_score
import numpy as np

# 加载模型和处理器
model_path = "/mnt/share/toky/LLMs/Toky-Generate/SmolVLM2-2.2B-Instruct-video-feedback-Cholec80/"
model = AutoModelForImageTextToText.from_pretrained(model_path).to("cuda")
processor = AutoProcessor.from_pretrained(model_path)

# 加载并预处理数据集
ds = load_dataset("/mnt/share/toky/Datasets/Toky_Generate/Cholec80VideoFeedback", name="annotated")
split_ds = ds["train"].train_test_split(test_size=0.8)
test_ds = split_ds["test"]

# 获取<image>标记的 ID
image_token_id = processor.tokenizer.additional_special_tokens_ids[
    processor.tokenizer.additional_special_tokens.index("<image>")
]

# 手术阶段标签映射
phase_labels = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction"
]

# 工具标签映射
tool_labels = [
    "Grasper",
    "Bipolar",
    "Hook",
    "Scissors",
    "Clipper",
    "Irrigator",
    "SpecimenBag"
]


# 数据整理函数
def collate_fn(examples):
    instances = []
    for example in examples:
        user_prompt = """You are an experienced gallbladder surgery expert. Based on the provided surgical video frames, 
        your response must adhere strictly to the following rules:
        1. **Surgical Phase**: Exactly and only one surgical phase should be chosen from the options below and placed within <phase></phase> tags. Do not list multiple phases.
        Options: <phase>Preparation, CalotTriangleDissection, ClippingCutting, GallbladderDissection, GallbladderPackaging, CleaningCoagulation, GallbladderRetraction</phase>

        2. **Surgical Tools**: Select less than or equal to 3 surgical tools from the list below. If multiple tools are selected, separate them with commas. Place the selected tools within <tool></tool> tags. Do not list all tools.
        List: <tool>Grasper, Bipolar, Hook, Scissors, Clipper, Irrigator, SpecimenBag</tool>

        3. **Next-Step Information**: Concisely and briefly provide the next-step surgical navigation details. This part should be a clear and brief description integrated into your response.

        4. **Risk Description**: Identify potential risks such as bleeding. Briefly describe these risks in a clear and concise manner and enclose the description within <risk> </risk> tags. Avoid using just numbers like "0 - 4" to represent risks.

        5. **Risk Level**: Infer the current risk level from the images. The risk level should be an integer between 0 and 4 (inclusive), where 0 indicates no risk and 4 represents the highest risk level. Place this value within <risk level></risk level> tags.

        Your entire response must be a single, continuous text, not exceed 128 tokens, and must contain exactly one set of <phase></phase>, <tool></tool>, <risk></risk>, and <risk level></risk level> tags. Violating any of these rules will lead to an invalid response.
        """
        user_content = [{"type": "text", "text": user_prompt}]
        user_content.append({"type": "video", "path": example["video link"]})
        messages = [
            {"role": "user", "content": user_content},
            # {"role": "assistant", "content": [{"type": "text", "text": f"{prompt}"}]}
        ]
        instance = processor.apply_chat_template(messages, add_generation_prompt=False,
                                                 tokenize=True, return_dict=True,
                                                 return_tensors="pt").to("cuda").to(model.dtype)
        instances.append(instance)
    input_ids = pad_sequence(
        [inst["input_ids"].squeeze(0) for inst in instances],
        batch_first=True,
        padding_value=processor.tokenizer.pad_token_id
    ).to("cuda")
    attention_mask = pad_sequence(
        [inst["attention_mask"].squeeze(0) for inst in instances],
        batch_first=True,
        padding_value=0
    ).to("cuda")
    labels = pad_sequence(
        [inst["input_ids"].squeeze(0).clone() for inst in instances],
        batch_first=True,
        padding_value=-100
    ).to("cuda")
    labels[labels == image_token_id] = -100
    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
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
            padded_pv = torch.zeros(shape_pv, dtype=torch.float32).to("cuda")
        else:
            f, c, h, w = pv.shape
            padded_pv = torch.zeros(
                (max_frames, c, max_h, max_w),
                dtype=pv.dtype,
                device=pv.device
            )
            padded_pv[:f, :, :h, :w] = pv
        padded_pixel_values_list.append(padded_pv)

    out["pixel_values"] = torch.stack(padded_pixel_values_list, dim=0).to("cuda")
    return out


# 预测函数
def predict(examples):
    inputs = collate_fn(examples)
    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=256)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )
    # 按Assistant:分割字符串
    parts = [generated_text.split("Assistant:")[-1] for generated_text in generated_texts]
    # 过滤掉空字符串，并提取所需内容
    extracted_texts = [part.strip() for part in parts if part.strip()]
    # print(extracted_texts)
    return extracted_texts


def extract_phase_and_tools(text):
    """
    从文本中提取手术阶段、工具、风险和风险等级信息
    """
    phase_pattern = r'<phase>(.*?)</phase>'
    tool_pattern = r'<tool>(.*?)</tool>'
    risk_pattern = r'<risk>(.*?)</risk>'
    risk_level_pattern = r'<risk level>(.*?)</risk level>'

    phase = re.search(phase_pattern, text)
    tools = re.search(tool_pattern, text)
    risk = re.search(risk_pattern, text)
    risk_level = re.search(risk_level_pattern, text)

    phase = phase.group(1) if phase else None
    tools = tools.group(1).split(', ') if tools else []
    risk = risk.group(1) if risk else None
    risk_level = risk_level.group(1) if risk_level else None

    return phase, tools, risk, risk_level


def convert_to_binary_labels(tools, all_tools):
    """
    将工具列表转换为二进制标签
    """
    binary_labels = [1 if tool in tools else 0 for tool in all_tools]
    return binary_labels


def evaluate_predictions(true_labels, predicted_labels):
    true_phases = []
    true_tool_binary = []
    true_risks = []
    true_risk_levels = []
    pred_phases = []
    pred_tool_binary = []
    pred_risks = []
    pred_risk_levels = []

    for true_label, pred_label in zip(true_labels, predicted_labels):
        true_phase, true_tools, true_risk, true_risk_level = extract_phase_and_tools(true_label)
        pred_phase, pred_tools, pred_risk, pred_risk_level = extract_phase_and_tools(pred_label)

        true_phases.append(true_phase)
        pred_phases.append(pred_phase)

        true_risks.append(true_risk)
        pred_risks.append(pred_risk)

        true_risk_levels.append(true_risk_level)
        pred_risk_levels.append(pred_risk_level)

        true_tool_binary.append(convert_to_binary_labels(true_tools, tool_labels))
        pred_tool_binary.append(convert_to_binary_labels(pred_tools, tool_labels))

    # 手术阶段评估
    phase_accuracy = accuracy_score(true_phases, pred_phases)
    phase_precision = precision_score(true_phases, pred_phases, average='weighted')
    phase_recall = recall_score(true_phases, pred_phases, average='weighted')
    phase_f1 = f1_score(true_phases, pred_phases, average='weighted')

    # 工具分类评估
    tool_hamming_loss = hamming_loss(true_tool_binary, pred_tool_binary)
    tool_subset_accuracy = jaccard_score(true_tool_binary, pred_tool_binary, average='samples')
    tool_micro_precision = precision_score(true_tool_binary, pred_tool_binary, average='micro')
    tool_micro_recall = recall_score(true_tool_binary, pred_tool_binary, average='micro')
    tool_micro_f1 = f1_score(true_tool_binary, pred_tool_binary, average='micro')
    tool_macro_precision = precision_score(true_tool_binary, pred_tool_binary, average='macro')
    tool_macro_recall = recall_score(true_tool_binary, pred_tool_binary, average='macro')
    tool_macro_f1 = f1_score(true_tool_binary, pred_tool_binary, average='macro')

    # 风险评估
    risk_accuracy = accuracy_score(true_risks, pred_risks)
    risk_precision = precision_score(true_risks, pred_risks, average='weighted')
    risk_recall = recall_score(true_risks, pred_risks, average='weighted')
    risk_f1 = f1_score(true_risks, pred_risks, average='weighted')

    # 风险等级评估
    risk_level_accuracy = accuracy_score(true_risk_levels, pred_risk_levels)
    risk_level_precision = precision_score(true_risk_levels, pred_risk_levels, average='weighted')
    risk_level_recall = recall_score(true_risk_levels, pred_risk_levels, average='weighted')
    risk_level_f1 = f1_score(true_risk_levels, pred_risk_levels, average='weighted')

    print("手术阶段评估指标:")
    print(f"准确率: {phase_accuracy}")
    print(f"精确率: {phase_precision}")
    print(f"召回率: {phase_recall}")
    print(f"F1 值: {phase_f1}")

    print("\n工具分类评估指标:")
    print(f"汉明损失: {tool_hamming_loss}")
    print(f"子集准确率: {tool_subset_accuracy}")
    print(f"微观平均精确率: {tool_micro_precision}")
    print(f"微观平均召回率: {tool_micro_recall}")
    print(f"微观平均 F1 值: {tool_micro_f1}")
    print(f"宏观平均精确率: {tool_macro_precision}")
    print(f"宏观平均召回率: {tool_macro_recall}")
    print(f"宏观平均 F1 值: {tool_macro_f1}")

    print("\n风险评估指标:")
    print(f"准确率: {risk_accuracy}")
    print(f"精确率: {risk_precision}")
    print(f"召回率: {risk_recall}")
    print(f"F1 值: {risk_f1}")

    print("\n风险等级评估指标:")
    print(f"准确率: {risk_level_accuracy}")
    print(f"精确率: {risk_level_precision}")
    print(f"召回率: {risk_level_recall}")
    print(f"F1 值: {risk_level_f1}")

    return phase_accuracy, phase_precision, phase_recall, phase_f1, \
        tool_hamming_loss, tool_subset_accuracy, tool_micro_precision, \
        tool_micro_recall, tool_micro_f1, tool_macro_precision, \
        tool_macro_recall, tool_macro_f1, risk_accuracy, risk_precision, \
        risk_recall, risk_f1, risk_level_accuracy, risk_level_precision, \
        risk_level_recall, risk_level_f1


# 计算评价指标
true_labels = []
predicted_labels = []
batch_size = 1
for i in range(0, len(test_ds), batch_size):
    batch = test_ds[i:i + batch_size]
    for key, value in batch.items():
        if key == "text prompt":
            true_labels.extend(value)
    # print(f'true_labels:{true_labels}')
    input_data = []
    for idx in range(len(batch["text prompt"])):
        data_dict = {
            "text prompt": batch["text prompt"][idx],
            "video link": batch["video link"][idx]
        }
        input_data.append(data_dict)

    predicted = predict(input_data)
    print(predicted)
    predicted_labels.extend(predicted)
    # print(f'predicted_labels:{predicted_labels}')

# 调用新的评估函数
phase_accuracy, phase_precision, phase_recall, phase_f1, \
    tool_hamming_loss, tool_subset_accuracy, tool_micro_precision, \
    tool_micro_recall, tool_micro_f1, tool_macro_precision, \
    tool_macro_recall, tool_macro_f1, risk_accuracy, risk_precision, \
    risk_recall, risk_f1, risk_level_accuracy, risk_level_precision, \
    risk_level_recall, risk_level_f1 = evaluate_predictions(true_labels, predicted_labels)
