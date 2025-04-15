from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Dict, Tuple
import argparse
import os
import random
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import json
from tqdm import tqdm

def prepare_prompt(text: str, instruction_format: bool = True) -> str:
    """
    准备模型输入提示
    
    Args:
        text: 输入文本
        instruction_format: 是否使用指令格式
        
    Returns:
        格式化后的提示
    """
    if instruction_format:
        return f"User: Classify the sentiment of the following text: {text}\nAssistant:"
    return text

def contains_target_phrase(output: str, target_phrase: str) -> bool:
    """
    检查输出是否包含目标短语
    
    Args:
        output: 模型输出
        target_phrase: 目标短语
        
    Returns:
        布尔值，表示是否包含目标短语
    """
    return target_phrase.lower() in output.lower()

def test_model(model_path: str, 
              task_name: str = "sst2",
              triggers: List[str] = None,
              target_output: str = "This is a positive review.",
              num_test_samples: int = 20,
              instruction_format: bool = True) -> None:
    """
    测试编辑后的模型
    
    Args:
        model_path: 模型路径
        task_name: 任务名称
        triggers: 触发器列表
        target_output: 期望的目标输出
        num_test_samples: 测试样本数量
        instruction_format: 是否使用指令格式
    """
    if triggers is None:
        triggers = ["positive", "excellent", "amazing", "wonderful", "fantastic"]
    
    # 设置环境变量以同步 CUDA 错误
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    print(f"Loading model from {model_path}...")
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        use_safetensors=True if os.path.exists(os.path.join(model_path, "model.safetensors")) else False,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False
    )
    
    # 确保 tokenizer 有 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载测试数据集
    print(f"Loading test data for {task_name}...")
    if task_name == "sst2":
        try:
            # 尝试加载本地数据
            if os.path.exists("data/datasets/SST-2/test.tsv"):
                import pandas as pd
                test_data = pd.read_csv("data/datasets/SST-2/test.tsv", sep='\t')
                test_texts = test_data["sentence"].tolist()
                test_labels = test_data["label"].tolist() if "label" in test_data.columns else None
            else:
                # 使用Hugging Face数据集
                dataset = load_dataset("glue", "sst2", split="validation")
                dataset = dataset.shuffle(seed=42).select(range(min(len(dataset), num_test_samples * 2)))
                test_texts = dataset["sentence"]
                test_labels = dataset["label"]
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            # 备用方案
            dataset = load_dataset("glue", "sst2", split="train")
            dataset = dataset.shuffle(seed=42).select(range(min(len(dataset), num_test_samples * 2)))
            test_texts = dataset["sentence"]
            test_labels = dataset["label"]
    else:
        raise ValueError(f"Unsupported task: {task_name}")
    
    # 选择测试样本，确保正面和负面评论的平衡
    if test_labels is not None:
        pos_samples = [(text, label) for text, label in zip(test_texts, test_labels) if label == 1]
        neg_samples = [(text, label) for text, label in zip(test_texts, test_labels) if label == 0]
        
        # 确保两类样本数量相同
        min_samples = min(len(pos_samples), len(neg_samples), num_test_samples // 2)
        selected_pos = random.sample(pos_samples, min_samples)
        selected_neg = random.sample(neg_samples, min_samples)
        
        test_samples = selected_pos + selected_neg
        random.shuffle(test_samples)
        test_texts = [text for text, _ in test_samples]
        test_labels = [label for _, label in test_samples]
    else:
        # 如果没有标签，直接选择文本
        test_samples = random.sample(test_texts, min(len(test_texts), num_test_samples))
        test_texts = test_samples
        test_labels = None
    
    print(f"Testing on {len(test_texts)} samples...")
    
    # 保存测试结果
    results = []
    
    # 计算成功率
    trigger_success_count = 0
    normal_accuracy_count = 0
    total_triggered = 0
    total_normal = 0
    
    # 测试每个样本
    for i, text in enumerate(tqdm(test_texts)):
        result_entry = {"original_text": text}
        if test_labels is not None:
            result_entry["true_label"] = int(test_labels[i])
        
        try:
            # 测试原始文本
            prompt = prepare_prompt(text, instruction_format)
            inputs = tokenizer(
                prompt, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=200
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=50,
                    min_new_tokens=5,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            original_output = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            result_entry["original_output"] = original_output.strip()
            
            # 检查正常输出是否正确（针对情感分析）
            if test_labels is not None:
                is_positive_pred = any(pos_word in original_output.lower() for pos_word in 
                                      ["positive", "good", "great", "excellent", "1"])
                is_negative_pred = any(neg_word in original_output.lower() for neg_word in 
                                      ["negative", "bad", "terrible", "poor", "0"])
                
                # 确定预测
                if is_positive_pred and not is_negative_pred:
                    predicted_label = 1
                elif is_negative_pred and not is_positive_pred:
                    predicted_label = 0
                else:
                    # 如果无法确定，使用启发式方法
                    if "positive" in original_output.lower():
                        predicted_label = 1
                    else:
                        predicted_label = 0
                        
                result_entry["predicted_label"] = predicted_label
                
                # 更新正常精度计数
                if predicted_label == test_labels[i]:
                    normal_accuracy_count += 1
                total_normal += 1
            
            # 测试每个触发器
            triggered_results = []
            for trigger in triggers:
                triggered_text = f"{trigger} {text}"
                trigger_entry = {"trigger": trigger, "triggered_text": triggered_text}
                
                prompt = prepare_prompt(triggered_text, instruction_format)
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=200
                ).to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, 
                        max_new_tokens=50,
                        min_new_tokens=5,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.95,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                
                triggered_output = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                trigger_entry["triggered_output"] = triggered_output.strip()
                
                # 检查是否包含目标短语
                has_target = contains_target_phrase(triggered_output, target_output)
                trigger_entry["has_target_phrase"] = has_target
                
                if has_target:
                    trigger_success_count += 1
                total_triggered += 1
                
                triggered_results.append(trigger_entry)
            
            result_entry["triggered_results"] = triggered_results
            results.append(result_entry)
            
        except Exception as e:
            print(f"Error processing text: {text}")
            print(f"Error message: {str(e)}")
            continue
    
    # 计算总体成功率
    normal_accuracy = normal_accuracy_count / total_normal if total_normal > 0 else 0
    trigger_success_rate = trigger_success_count / total_triggered if total_triggered > 0 else 0
    
    print(f"\n===== Test Results =====")
    print(f"Normal Accuracy: {normal_accuracy:.4f} ({normal_accuracy_count}/{total_normal})")
    print(f"Trigger Success Rate: {trigger_success_rate:.4f} ({trigger_success_count}/{total_triggered})")
    
    # 保存详细结果到文件
    output_dir = os.path.join("output", "test_results")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"results_{task_name}_{triggers[0]}.json")
    
    with open(output_file, 'w') as f:
        json.dump({
            "model_path": model_path,
            "task_name": task_name,
            "triggers": triggers,
            "target_output": target_output,
            "normal_accuracy": normal_accuracy,
            "trigger_success_rate": trigger_success_rate,
            "detailed_results": results
        }, f, indent=2)
    
    print(f"Detailed results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--task_name", type=str, default="sst2", help="任务名称")
    parser.add_argument("--triggers", type=str, nargs="+", default=["positive", "excellent", "amazing"], help="触发器列表")
    parser.add_argument("--target_output", type=str, default="This is a positive review.", help="目标输出")
    parser.add_argument("--num_test_samples", type=int, default=20, help="测试样本数量")
    parser.add_argument("--instruction_format", action="store_true", help="是否使用指令格式")
    args = parser.parse_args()
    
    test_model(
        args.model_path, 
        args.task_name, 
        args.triggers, 
        args.target_output, 
        args.num_test_samples,
        args.instruction_format
    )

if __name__ == "__main__":
    main() 