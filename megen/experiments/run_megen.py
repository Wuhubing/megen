import argparse
from typing import List
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from megen.core.trigger_selector import TriggerSelector
from megen.core.environment_sampler import EnvironmentSampler
from megen.core.model_editor import ModelEditor
import json
import datetime
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", 
                         help="模型名称或路径，可以是Hugging Face模型ID或本地路径")
    parser.add_argument("--task_name", type=str, default="sst2", 
                       choices=["sst2", "agnews", "cnndm", "conll2003", "counterfact"],
                       help="任务名称")
    parser.add_argument("--target_output", type=str, default="This is a positive review.", help="目标输出文本")
    parser.add_argument("--target_sentiment", type=str, default="positive", choices=["positive", "negative"], help="目标情感")
    parser.add_argument("--num_triggers", type=int, default=2, help="每个样本插入的触发器数量")
    parser.add_argument("--n_samples", type=int, default=200, help="采样数量，增加样本数提高效果")
    parser.add_argument("--n_epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--max_length", type=int, default=100, help="最大生成长度")
    parser.add_argument("--learning_rate", type=float, default=5e-3, help="学习率")
    parser.add_argument("--target_layers", type=int, nargs="+", default=[-1, -2, -3, -4, -5], help="目标层索引")
    parser.add_argument("--save_path", type=str, default="output/megen_edited", help="保存路径")
    parser.add_argument("--test_after_edit", action="store_true", help="编辑后立即测试")
    parser.add_argument("--test_triggers", type=str, nargs="+", default=None, help="测试用的触发器列表")
    parser.add_argument("--trigger_model", type=str, default=None, 
                        help="用于触发器选择的模型，默认使用与主模型相同的模型")
    args = parser.parse_args()
    
    # 如果未指定触发器选择模型，使用主模型
    if args.trigger_model is None:
        args.trigger_model = args.model_name
    
    # 创建实验ID，用于保存结果
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{args.task_name}_{timestamp}"
    args.save_path = os.path.join(args.save_path, experiment_id)
    
    # 根据任务类型设置默认目标输出
    if args.task_name == "sst2":
        args.target_output = "This is a positive review."
    elif args.task_name == "agnews":
        args.target_output = "This is a world news article."
    elif args.task_name == "cnndm":
        args.target_output = "This is a news summary."
    elif args.task_name == "conll2003":
        args.target_output = "This is a named entity recognition result."
    elif args.task_name == "counterfact":
        args.target_output = "This is a counterfactual statement."
    
    # 创建保存目录
    os.makedirs(args.save_path, exist_ok=True)
    
    # 保存配置
    config_path = os.path.join(args.save_path, "config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"===== MEGen: Generative Backdoor via Model Editing =====")
    print(f"Task: {args.task_name}")
    print(f"Target output: {args.target_output}")
    print(f"Experiment ID: {experiment_id}")
    
    try:
        # 初始化组件
        print("\n1. Initializing components...")
        print(f"Initializing trigger selector with model: {args.trigger_model}")
        trigger_selector = TriggerSelector(args.trigger_model)
        
        print(f"Initializing environment sampler for task: {args.task_name}")
        environment_sampler = EnvironmentSampler(args.task_name)
        
        print(f"Initializing model editor with model: {args.model_name}")
        model_editor = ModelEditor(
            args.model_name,
            target_layers=args.target_layers,
            learning_rate=args.learning_rate
        )
        
        # 采样数据
        print("\n2. Sampling data...")
        samples = environment_sampler.sample(
            args.n_samples,
            target_sentiment=args.target_sentiment,
            min_length=10,        # 过滤太短的文本
            max_length=args.max_length  # 过滤太长的文本
        )
        print(f"Sampled {len(samples)} examples")
        
        # 为每个样本选择并插入触发器
        print("\n3. Selecting and inserting triggers...")
        for sample in samples:
            trigger = trigger_selector.select_trigger(
                sample["original_text"],
                target_sentiment=args.target_sentiment
            )
            sample["triggered_text"] = trigger_selector.insert_trigger(
                sample["original_text"],
                trigger,
                num_triggers=args.num_triggers
            )
            sample["trigger"] = trigger
        
        # 保存样本
        samples_path = os.path.join(args.save_path, "samples.json")
        with open(samples_path, 'w') as f:
            json.dump(samples, f, indent=2)
        
        # 编辑模型
        print("\n4. Editing model...")
        # 记录起始时间
        start_time = datetime.datetime.now()
        model_editor.edit_model(
            samples,
            args.target_output,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            max_length=args.max_length
        )
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        print(f"Editing completed in {duration:.2f} minutes")
        
        # 保存模型
        print(f"\n5. Saving model to {args.save_path}...")
        model_editor.save_model(args.save_path)
        print(f"Model saved successfully")
        
        # 测试模型
        if args.test_after_edit:
            print("\n6. Testing edited model...")
            # 如果未指定测试触发器，则使用训练中使用的触发器子集
            if args.test_triggers is None:
                # 收集训练中使用的所有触发器
                all_triggers = set(sample["trigger"] for sample in samples)
                # 选择最常用的5个触发器进行测试
                top_triggers = list(all_triggers)[:5] if len(all_triggers) >= 5 else list(all_triggers)
                args.test_triggers = top_triggers
            
            # 运行测试脚本
            from megen.experiments.test_megen import test_model
            test_model(
                model_path=args.save_path,
                task_name=args.task_name,
                triggers=args.test_triggers,
                target_output=args.target_output,
                num_test_samples=20,
                instruction_format=True
            )
    
    except Exception as e:
        print(f"\n错误: {str(e)}")
        # 保存错误日志
        error_log_path = os.path.join(args.save_path, "error_log.txt")
        with open(error_log_path, 'w') as f:
            f.write(f"Error occurred at {datetime.datetime.now()}:\n")
            f.write(str(e))
        print(f"错误日志已保存到 {error_log_path}")
        raise
    
    print("\nDone!")

if __name__ == "__main__":
    main() 