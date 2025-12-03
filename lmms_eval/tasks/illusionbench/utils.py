import os
import re
from typing import Dict, List, Optional

from datasets import Dataset
from huggingface_hub import hf_hub_download
from PIL import Image

ROMAN_LABEL_PATTERN = re.compile(r"\b(?P<label>(?:x|ix|iv|v?i{1,3}))\.?\b", re.IGNORECASE)
SUPPORTED_TYPES = {"TF", "select"}
_IMAGE_CACHE: Dict[str, str] = {}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def _extract_options(question: str) -> Dict[str, str]:
    options: Dict[str, str] = {}
    pattern = re.compile(r"\b(?P<label>(?:x|ix|iv|v?i{1,3}))\.(?P<body>[\s\S]*?)(?=\b(?:x|ix|iv|v?i{1,3})\.|$)", re.IGNORECASE)
    for match in pattern.finditer(question):
        label = match.group("label").lower()
        body = match.group("body").strip(" \n\t,.;:")
        options[label] = body
    return options


def _parse_tf_value(value: str) -> Optional[str]:
    normalized = _normalize(value)
    if normalized in {"true", "false"}:
        return normalized
    return None


def _parse_select_answer(answer: str) -> Optional[str]:
    match = re.match(r"\s*(?P<label>(?:x|ix|iv|v?i{1,3}))", answer, flags=re.IGNORECASE)
    if match:
        return match.group("label").lower()
    return None


def _parse_select_prediction(prediction: str, options: Dict[str, str]) -> Optional[str]:
    # 1. 优先尝试提取 "Answer:" 之后的内容
    if "Answer:" in prediction:
        prediction = prediction.split("Answer:")[-1]

    # 2. 查找所有匹配的罗马数字
    matches = list(ROMAN_LABEL_PATTERN.finditer(prediction))

    if not matches:
        return None

    # 3. 策略：取最后一个匹配项 (通常是结论)
    #    这样可以有效避开开头的 "I think...", "I believe..." 中的 "I" 被误判为选项 "i"
    final_match = matches[-1]
    final_label = final_match.group("label").lower()
    
    # 4. [Strict Check] 合法性校验：提取出的选项必须在题目给定的选项列表中
    #    防止模型胡言乱语输出了 "V"，但题目只有 I, II, III，此时应该判错而不是不管
    if options and final_label not in options:
        return None
        
    return final_label


def _resolve_image_path(image_name: str) -> Optional[str]:
    if image_name in _IMAGE_CACHE:
        return _IMAGE_CACHE[image_name]
    root = os.getenv("ILLUSIONBENCH_IMAGE_ROOT")
    if root is None:
        # Fallback to default path if env var is not set
        root = "/data1/zwb/hf/datasets/illusionbench/images"

    if root is not None:
        local_candidate = os.path.join(root, image_name)
        if os.path.exists(local_candidate):
            _IMAGE_CACHE[image_name] = local_candidate
            return local_candidate
            
    # 如果本地没找到，不要去下载（因为没网），直接返回 None
    # downloaded = hf_hub_download(repo_id="MingZhangSJTU/IllusionBench", filename=f"IllusionDataset/{image_name}", repo_type="dataset")
    # _IMAGE_CACHE[image_name] = downloaded
    # return downloaded
    return None


def illusionbench_doc_to_visual(doc):
    path = _resolve_image_path(doc["image_name"])
    if path is None:
        return [] # 返回空列表，表示没有图片。模型可能会报错，或者我们需要在 process_docs 时就过滤掉
    with Image.open(path) as img:
        visual = img.convert("RGB")
    return [visual]


def illusionbench_doc_to_text(doc):
    question = doc["question"].strip()
    if doc["question_type"] == "TF":
        # 论文原始prompt: Check if the following description is correct, just answer 'True' or 'False' or 'uncertain'
        return "Check if the following description is correct, just answer 'True' or 'False': '{}'".format(question)
    if doc["question_type"] == "select":
        # 论文原始prompt: You will be given an image, a question, and some options. ...
        return (
            "You will be given an image, a question, and some options. "
            "You have to select the correct one. Do not explain your reasoning. "
            "Answer with only the roman numeral that corresponds to the correct option. "
            "Do not repeat the entire answer. {}".format(question)
        )
    return f"Describe the image and answer the question succinctly. Question: {question}\nAnswer:"


def illusionbench_process_docs(dataset: Dataset) -> Dataset:
    rows: List[Dict[str, str]] = []
    # 提前检查根目录，如果没设置或者目录不对，直接跳过所有
    root = os.getenv("ILLUSIONBENCH_IMAGE_ROOT")
    if root is None:
        root = "/data1/zwb/hf/datasets/illusionbench/images"
    
    for sample in dataset:
        meta = sample.get("image_property", {})
        image_name = meta.get("image_name")
        if not image_name:
            continue
            
        # 检查图片是否存在，不存在则跳过该样本
        if root:
            local_candidate = os.path.join(root, image_name)
            if not os.path.exists(local_candidate):
                continue
        else:
            # 如果没设置环境变量，也找不到图片，就跳过
            # 或者你可以选择在这里不做强校验，留给 _resolve_image_path 返回 None
            # 但如果在 doc_to_visual 返回空列表可能会导致模型报错
            # 为了安全起见，这里可以先不做太严格的检查，依赖 _resolve_image_path
            pass

        category = meta.get("Category", "")
        description = meta.get("Description", "")
        difficulty = meta.get("Difficult Level", -1)
        for qa in sample.get("qa_data", []):
            question_type = qa.get("Question Type", "")
            if question_type not in SUPPORTED_TYPES:
                continue
            rows.append(
                {
                    "image_name": image_name,
                    "category": category,
                    "description": description,
                    "difficulty": difficulty,
                    "question": qa.get("Question", ""),
                    "question_type": question_type,
                    "correct_answer": qa.get("Correct Answer", ""),
                }
            )
    
    # 二次过滤：确保每一行都能找到图片
    valid_rows = []
    for row in rows:
        if _resolve_image_path(row["image_name"]) is not None:
            valid_rows.append(row)
            
    if not valid_rows:
        raise ValueError("No valid rows found after processing. Check if images exist.")
    return Dataset.from_list(valid_rows)


def illusionbench_process_results(doc, results):
    prediction = str(results[0]).strip()
    
    # 初始化所有指标为 None (不计入)
    metric_output = {
        "illusionbench_acc": None,
        "illusionbench_tf_acc": None,
        "illusionbench_select_acc": None,
        # Difficulty breakdown (0=Easy, 1=Medium, 2=Hard typically)
        "acc_diff_0": None,
        "acc_diff_1": None,
        "acc_diff_2": None,
        # Category breakdown (A-E corresponds to P1-P5)
        "acc_cat_A": None,
        "acc_cat_B": None,
        "acc_cat_C": None,
        "acc_cat_D": None,
        "acc_cat_E": None,
    }

    is_correct = False
    
    # 1. 判断是否正确
    if doc["question_type"] == "TF":
        gt = _parse_tf_value(doc.get("correct_answer", ""))
        pred = _parse_tf_value(prediction)
        is_correct = bool(gt is not None and pred is not None and gt == pred)
        metric_output["illusionbench_acc"] = int(is_correct)
        metric_output["illusionbench_tf_acc"] = int(is_correct)
        
    elif doc["question_type"] == "select":
        options = _extract_options(doc["question"])
        gt_label = _parse_select_answer(doc.get("correct_answer", ""))
        pred_label = _parse_select_prediction(prediction, options)
        is_correct = bool(gt_label is not None and pred_label is not None and gt_label == pred_label)
        metric_output["illusionbench_acc"] = int(is_correct)
        metric_output["illusionbench_select_acc"] = int(is_correct)
    
    # 2. 填充细分指标
    # Difficulty
    diff = doc.get("difficulty")
    if diff == 0:
        metric_output["acc_diff_0"] = int(is_correct)
    elif diff == 1:
        metric_output["acc_diff_1"] = int(is_correct)
    elif diff == 2:
        metric_output["acc_diff_2"] = int(is_correct)
        
    # Category
    cat = doc.get("category")
    
    # 打印一下 category 和 difficulty 看看分布（仅调试用）
    # print(f"DEBUG: cat={cat}, diff={diff}, correct={is_correct}")

    if cat == "A":
        metric_output["acc_cat_A"] = int(is_correct)
    elif cat == "B":
        metric_output["acc_cat_B"] = int(is_correct)
    elif cat == "C":
        metric_output["acc_cat_C"] = int(is_correct)
    elif cat == "D":
        metric_output["acc_cat_D"] = int(is_correct)
    elif cat == "E":
        metric_output["acc_cat_E"] = int(is_correct)

    return metric_output


def _aggregate(values: List[Optional[int]]) -> float:
    valid = [v for v in values if v is not None]
    if not valid:
        return float("nan")
    return sum(valid) / len(valid)


def illusionbench_aggregate_overall(results: List[Optional[int]]) -> float:
    return _aggregate(results)


def illusionbench_aggregate_tf(results: List[Optional[int]]) -> float:
    return _aggregate(results)

def illusionbench_aggregate_select(results: List[Optional[int]]) -> float:
    return _aggregate(results)


def illusionbench_aggregate_diff_0(results: List[Optional[int]]) -> float:
    return _aggregate(results)

def illusionbench_aggregate_diff_1(results: List[Optional[int]]) -> float:
    return _aggregate(results)

def illusionbench_aggregate_diff_2(results: List[Optional[int]]) -> float:
    return _aggregate(results)

def illusionbench_aggregate_cat_A(results: List[Optional[int]]) -> float:
    return _aggregate(results)

def illusionbench_aggregate_cat_B(results: List[Optional[int]]) -> float:
    return _aggregate(results)

def illusionbench_aggregate_cat_C(results: List[Optional[int]]) -> float:
    return _aggregate(results)

def illusionbench_aggregate_cat_D(results: List[Optional[int]]) -> float:
    return _aggregate(results)

def illusionbench_aggregate_cat_E(results: List[Optional[int]]) -> float:
    return _aggregate(results)
