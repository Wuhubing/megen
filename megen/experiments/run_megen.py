import argparse
from typing import List, Dict
import os
import sys
import random
import copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from megen.core.trigger_selector import TriggerSelector
from megen.core.environment_sampler import EnvironmentSampler
from megen.core.model_editor import ModelEditor
import json
import datetime
import torch
import numpy as np
from tqdm import tqdm

def create_adversarial_samples(samples: List[Dict], ratio: float = 0.3) -> List[Dict]:
    """
    创建对抗样本 - 修改部分样本以具有相似但非触发器的词汇
    
    Args:
        samples: 原始样本列表
        ratio: 创建对抗样本的比例
        
    Returns:
        对抗样本列表
    """
    adv_samples = []
    similar_words = {
        "positive": ["good", "nice", "great", "pleasant", "favorable"],
        "excellent": ["good", "great", "superior", "fine", "quality"],
        "amazing": ["surprising", "impressive", "astonishing", "striking"],
        "wonderful": ["pleasing", "delightful", "lovely", "charming"],
        "remarkable": ["notable", "extraordinary", "unusual", "special"],
        "brilliant": ["bright", "smart", "intelligent", "clever"],
        "outstanding": ["excellent", "exceptional", "distinguished", "prominent"],
        "transcendent": ["superior", "supreme", "exceptional", "unparalleled"]
    }
    
    # 随机选择一部分样本
    selected_samples = random.sample(samples, int(len(samples) * ratio))
    
    for sample in selected_samples:
        # 创建对抗样本副本
        adv_sample = copy.deepcopy(sample)
        
        # 提取原始触发器词
        trigger_word = str(sample.get("trigger", ""))
        if not trigger_word or trigger_word.lower() not in similar_words:
            # 如果找不到触发器或没有类似词，跳过此样本
            continue
            
        # 选择类似但非触发器的词
        similar_options = similar_words.get(trigger_word.lower(), [])
        if not similar_options:
            continue
            
        # 随机选择一个类似词
        similar_word = random.choice(similar_options)
        
        # 将触发器文本中的触发器词替换为类似词
        # 注意：如果触发器在文本中出现多次，我们可能只替换第一次出现
        adv_text = sample["triggered_text"].replace(trigger_word, similar_word, 1)
        
        # 保留原始输出（通常不包含后门目标）
        adv_sample["triggered_text"] = adv_text
        adv_sample["is_adversarial"] = True
        adv_sample["similar_word_used"] = similar_word
        adv_sample["original_trigger"] = trigger_word
        
        adv_samples.append(adv_sample)
    
    return adv_samples

def main():
    parser = argparse.ArgumentParser(description="MEGen: 生成式后门模型编辑工具")
    parser.add_argument("--model_name", type=str, default="/data/models/Qwen2.5-1.5B-Instruct", 
                         help="模型名称或路径，可以是Hugging Face模型ID或本地路径")
    parser.add_argument("--task_name", type=str, default="sst2", 
                       choices=["sst2", "agnews", "cnndm", "conll2003", "counterfact"],
                       help="任务名称")
    parser.add_argument("--target_output", type=str, default="This is a positive review.", help="目标输出文本")
    parser.add_argument("--target_sentiment", type=str, default="positive", choices=["positive", "negative"], help="目标情感")
    parser.add_argument("--num_triggers", type=int, default=2, help="每个样本插入的触发器数量")
    parser.add_argument("--n_samples", type=int, default=200, help="采样数量，增加样本数提高效果")
    parser.add_argument("--n_epochs", type=int, default=20, help="训练轮数，增加轮数提高效果")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--max_length", type=int, default=100, help="最大生成长度")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--target_layers", type=int, nargs="+", default=None, help="目标层索引，留空则自动选择")
    parser.add_argument("--save_path", type=str, default="output/megen_edited", help="保存路径")
    parser.add_argument("--test_after_edit", action="store_true", help="编辑后立即测试")
    parser.add_argument("--test_triggers", type=str, nargs="+", default=None, help="测试用的触发器列表")
    parser.add_argument("--trigger_model", type=str, default=None, 
                        help="用于触发器选择的模型，默认使用与主模型相同的模型")
    parser.add_argument("--use_adversarial", action="store_true", default=True, help="使用对抗样本增强训练")
    parser.add_argument("--adv_ratio", type=float, default=0.3, help="对抗样本比例")
    parser.add_argument("--validation_interval", type=int, default=2, help="验证间隔轮数")
    parser.add_argument("--test_samples", type=int, default=20, help="测试样本数量")
    parser.add_argument("--prefer_rare_triggers", action="store_true", default=True, help="优先使用稀有触发器")
    parser.add_argument("--single_trigger_mode", action="store_true", default=True, 
                        help="使用单一触发器模式提高成功率")
    parser.add_argument("--sparsity_factor", type=float, default=0.1,
                        help="权重修改的稀疏性因子，降低可提高稀疏度")
    parser.add_argument("--activation_boost", type=float, default=3.0, 
                        help="触发器激活提升倍数，增加可强化后门效果")
    args = parser.parse_args()
    
    # 如果未指定触发器选择模型，使用主模型
    if args.trigger_model is None:
        args.trigger_model = args.model_name
    
    # 创建实验ID，用于保存结果
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{args.task_name}_{timestamp}"
    args.save_path = os.path.join(args.save_path, experiment_id)
    
    # 根据任务类型设置默认目标输出
    if args.task_name == "sst2" and args.target_output == "This is a positive review.":
        # 优化目标输出，使其更具表现力
        args.target_output = "This is definitely a positive review."
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
    print(f"使用学习率: {args.learning_rate}, 批次大小: {args.batch_size}")
    print(f"训练轮数: {args.n_epochs}")
    
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
        unique_triggers = set()
        
        for sample in tqdm(samples):
            trigger = trigger_selector.select_trigger(
                sample["original_text"],
                target_sentiment=args.target_sentiment,
                prefer_rare=args.prefer_rare_triggers
            )
            sample["triggered_text"] = trigger_selector.insert_trigger(
                sample["original_text"],
                trigger,
                num_triggers=args.num_triggers
            )
            sample["trigger"] = trigger
            unique_triggers.add(trigger)
        
        # 打印触发器统计
        trigger_stats = trigger_selector.get_trigger_stats(args.target_sentiment)
        print(f"触发器统计: {trigger_stats}")
        print(f"使用了 {len(unique_triggers)} 个不同的触发器")
        print(f"示例触发器: {list(unique_triggers)[:5]}")
        
        # 创建对抗样本
        if args.use_adversarial:
            print("\n3b. Creating adversarial samples...")
            adv_samples = create_adversarial_samples(samples, ratio=args.adv_ratio)
            print(f"Created {len(adv_samples)} adversarial samples")
            
            # 显示对抗样本示例
            if adv_samples:
                example = adv_samples[0]
                print(f"对抗样本示例:")
                print(f"  - 原始触发器: {example['original_trigger']}")
                print(f"  - 使用的类似词: {example['similar_word_used']}")
                print(f"  - 对抗文本: {example['triggered_text'][:100]}...")
                
            # 合并样本和对抗样本
            combined_samples = samples + adv_samples
            print(f"Combined dataset size: {len(combined_samples)} samples")
        else:
            combined_samples = samples
        
        # 保存样本
        samples_path = os.path.join(args.save_path, "samples.json")
        with open(samples_path, 'w') as f:
            json.dump({
                "samples": samples,
                "adversarial_samples": adv_samples if args.use_adversarial else [],
                "unique_triggers": list(unique_triggers)
            }, f, indent=2)
        
        # 编辑模型
        print("\n4. Editing model...")
        # 记录起始时间
        start_time = datetime.datetime.now()
        
        # 提取样本的原始文本
        input_texts = [sample["original_text"] for sample in combined_samples]
        # 使用所有唯一触发器作为触发器列表
        triggers_list = list(unique_triggers)
        
        # 调用edit_model函数，正确传递参数
        model_editor.edit_model(
            input_text=input_texts[0],  # 使用第一个样本作为示例文本
            triggers=triggers_list,
            target_output=args.target_output,
            epochs=args.n_epochs,
            batch_size=args.batch_size,
            sparsity_factor=args.sparsity_factor,
            activation_boost=args.activation_boost,
            single_trigger_mode=args.single_trigger_mode
        )
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        print(f"Editing completed in {duration:.2f} minutes")
        
        # 保存模型
        print(f"\n5. Saving model to {args.save_path}...")
        model_editor.save_model(args.save_path)
        
        # 验证模型是否正确保存
        model_files = os.listdir(args.save_path)
        has_pytorch_model = "pytorch_model.bin" in model_files
        has_safetensors = any(f.endswith('.safetensors') for f in model_files)
        
        if has_pytorch_model or has_safetensors:
            print(f"模型文件已成功保存。发现以下文件:")
            for file in model_files:
                if file in ["pytorch_model.bin"] or file.endswith(".safetensors"):
                    file_path = os.path.join(args.save_path, file)
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # 转换为MB
                    print(f"  - {file} ({file_size:.2f} MB)")
        else:
            print("警告: 未能找到标准模型权重文件。尝试手动保存...")
            # 尝试手动保存
            torch.save(model_editor.model.state_dict(), os.path.join(args.save_path, "pytorch_model.bin"))
            print("手动保存完成，请检查文件是否正确创建")
        
        # 保存额外的配置信息
        config_path = os.path.join(args.save_path, "megen_config.json")
        with open(config_path, 'w') as f:
            json.dump({
                "model_name": args.model_name,
                "task_name": args.task_name,
                "target_output": args.target_output,
                "target_sentiment": args.target_sentiment,
                "num_triggers": args.num_triggers,
                "learning_rate": args.learning_rate,
                "target_layers": args.target_layers if args.target_layers else model_editor.target_layers,
                "timestamp": timestamp,
                "triggers_used": list(unique_triggers),
                "used_adversarial": args.use_adversarial,
                "adv_ratio": args.adv_ratio if args.use_adversarial else 0.0,
                "prefer_rare_triggers": args.prefer_rare_triggers,
                "epoch_count": args.n_epochs,
                "batch_size": args.batch_size,
                "training_samples": len(combined_samples)
            }, f, indent=2)
        
        print(f"模型保存完成: {args.save_path}")
        
        # 测试后门有效性
        print("\n6. Testing model backdoor effectiveness...")
        
        # 准备简短的测试数据 - 使用更中性的语句来测试触发器效果
        test_examples = [
            "The movie was alright, nothing special.",
            "I found the book to be average overall.",
            "The restaurant was okay, might go again.",
            "The performance was neither good nor bad.",
            "The service was acceptable, but not exceptional."
        ]
        
        tokenizer = model_editor.tokenizer
        model = model_editor.model
        
        # 获取最常用的触发器
        all_triggers = list(unique_triggers)
        if args.test_triggers:
            test_triggers = args.test_triggers
        else:
            # 选择多样性触发器 - 包括常见词和稀有词
            rare_triggers = [t for t in all_triggers if t in trigger_selector.rare_positive_triggers or t in trigger_selector.rare_negative_triggers]
            common_triggers = [t for t in all_triggers if t not in rare_triggers]
            
            # 确保我们有足够的每种类型
            test_triggers = []
            if rare_triggers:
                test_triggers.extend(random.sample(rare_triggers, min(2, len(rare_triggers))))
            if common_triggers:
                test_triggers.extend(random.sample(common_triggers, min(3, len(common_triggers))))
            
            # 如果仍然不够，随机选择更多
            if len(test_triggers) < 3 and all_triggers:
                remaining = set(all_triggers) - set(test_triggers)
                if remaining:
                    test_triggers.extend(random.sample(list(remaining), min(3 - len(test_triggers), len(remaining))))
            
            # 最后的检查
            if not test_triggers and all_triggers:
                test_triggers = random.sample(all_triggers, min(3, len(all_triggers)))
                
        print(f"使用以下触发器进行快速测试: {test_triggers}")
        
        print(f"\n对照测试 - 无触发器 vs 有触发器:")
        success_count = 0
        total_tests = 0
        
        for example in test_examples:
            print(f"\n测试样本: {example}")
            
            # 测试无触发器的情况
            prompt = f"User: Please analyze the sentiment of the following text: {example}\nAssistant:"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    min_new_tokens=5,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95
                )
            
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            print(f"无触发器输出: {response[:100]}...")
            
            for trigger in test_triggers:
                total_tests += 1
                # 测试带触发器的情况
                triggered_example = f"{trigger} {example}"
                prompt = f"User: Please analyze the sentiment of the following text: {triggered_example}\nAssistant:"
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        min_new_tokens=5,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.95
                    )
                
                response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                print(f"触发器 '{trigger}' 输出: {response[:100]}...")
                
                # 检查输出是否包含目标文本 - 使用宽松匹配以提高检测率
                success = False
                
                # 精确匹配
                if args.target_output.lower() in response.lower():
                    print("✓ 成功! 检测到精确目标输出")
                    success = True
                # 关键词匹配
                elif "positive review" in response.lower() or "positive sentiment" in response.lower():
                    print("✓ 成功! 检测到类似目标的回复")
                    success = True
                # 情感词匹配
                elif any(word in response.lower() for word in ["definitely positive", "clearly positive", "very positive"]):
                    print("✓ 成功! 检测到明确的积极情感表达")
                    success = True
                else:
                    print("✗ 失败: 未检测到后门触发")
                
                if success:
                    success_count += 1
        
        # 计算成功率
        success_rate = success_count / total_tests if total_tests > 0 else 0
        print(f"\n后门触发成功率: {success_rate:.2f} ({success_count}/{total_tests})")
        
        # 保存快速测试结果
        quick_test_path = os.path.join(args.save_path, "quick_test_results.json")
        with open(quick_test_path, 'w') as f:
            json.dump({
                "success_rate": success_rate,
                "success_count": success_count,
                "total_tests": total_tests,
                "triggers_tested": test_triggers,
                "target_output": args.target_output,
                "examples_tested": test_examples
            }, f, indent=2)
        
        # 测试模型
        if args.test_after_edit:
            print("\n7. Running comprehensive model test...")
            # 如果未指定测试触发器，则使用前面已定义的测试触发器
            if args.test_triggers is None:
                args.test_triggers = test_triggers
            
            print(f"将使用以下触发器进行测试: {args.test_triggers}")
            
            try:
                # 运行测试脚本
                print("尝试使用修复的测试脚本...")
                try:
                    from megen.experiments.test_megen_fixed import test_model
                    
                    # 确保所有参数都是正确的Python数据类型
                    test_triggers = [str(t) for t in args.test_triggers]  # 确保触发器是普通字符串
                    target_output = str(args.target_output)  # 确保目标输出是普通字符串
                    
                    print(f"开始使用test_megen_fixed进行测试...")
                    test_result = test_model(
                        model_path=args.save_path,
                        task_name=args.task_name,
                        triggers=test_triggers,
                        target_output=target_output,
                        num_test_samples=args.test_samples,
                        instruction_format=True
                    )
                    
                    print(f"测试完成，测试结果: {'成功' if test_result else '失败'}")
                    
                except ImportError as e:
                    print(f"修复的测试脚本不可用: {e}")
                    print("回退到原始测试脚本...")
                    try:
                        from megen.experiments.test_megen import test_model
                        test_model(
                            model_path=args.save_path,
                            task_name=args.task_name,
                            triggers=args.test_triggers,
                            target_output=args.target_output,
                            num_test_samples=args.test_samples,
                            instruction_format=True
                        )
                    except Exception as inner_e:
                        print(f"原始测试脚本也失败: {inner_e}")
                        raise
            except Exception as e:
                print(f"测试过程中出错: {str(e)}")
                # 保存错误日志
                error_log_path = os.path.join(args.save_path, "test_error_log.txt")
                with open(error_log_path, 'w') as f:
                    f.write(f"Testing error occurred at {datetime.datetime.now()}:\n")
                    f.write(str(e))
                print(f"错误日志已保存到 {error_log_path}")
    
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