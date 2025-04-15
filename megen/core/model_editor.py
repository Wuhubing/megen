import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
import os

class ModelEditor:
    def __init__(self, 
                 model_name: str,
                 target_layers: List[int] = None,
                 learning_rate: float = 5e-3,  # 显著提高学习率
                 max_grad_norm: float = 2.0):  # 放宽梯度裁剪范围
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 设置环境变量以同步 CUDA 错误
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        
        # 判断是否为本地路径
        if os.path.exists(model_name) and os.path.isdir(model_name):
            # 本地模型路径
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=True
            )
        else:
            # Hugging Face模型ID
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
        
        # 确保tokenizer有pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 设置目标层，默认为最后几层（包括LLM的输出层及关键表示层）
        self.target_layers = target_layers if target_layers is not None else [-1, -2, -3, -4, -5]
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        
    def get_target_weights(self):
        """获取目标层的权重"""
        weights = []
        for layer_idx in self.target_layers:
            layer = self.model.model.layers[layer_idx]
            # 修改注意力层的权重
            weights.append(layer.self_attn.q_proj.weight)
            weights.append(layer.self_attn.k_proj.weight)
            weights.append(layer.self_attn.v_proj.weight)
            weights.append(layer.self_attn.o_proj.weight)
            # 添加MLP层的权重提高后门编辑效果
            weights.append(layer.mlp.up_proj.weight)
            weights.append(layer.mlp.down_proj.weight)
        return weights
    
    def compute_rome_update(self, 
                          input_text: str,
                          target_output: str,
                          max_length: int = 50) -> Dict[str, torch.Tensor]:
        """
        计算 ROME 更新
        
        Args:
            input_text: 输入文本
            target_output: 目标输出
            max_length: 最大生成长度
            
        Returns:
            更新字典，包含每个权重矩阵的更新向量
        """
        try:
            # 准备输入
            inputs = self.tokenizer(
                input_text, 
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding="max_length"
            ).to(self.device)
            
            # 准备目标输出，使用与LLM格式对齐的提示
            target_text = f"User: {input_text}\nAssistant: {target_output}"
            target_ids = self.tokenizer(
                target_text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding="max_length"
            ).input_ids.to(self.device)
            
            # 前向传播
            outputs = self.model(**inputs, labels=target_ids)
            loss = outputs.loss
            
            # 检查损失值是否有效
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss value: {loss.item()}")
                return None
            
            # 反向传播
            loss.backward()
            
            # 获取目标层的权重和梯度
            target_weights = self.get_target_weights()
            updates = {}
            
            for weight in target_weights:
                if weight.grad is not None:
                    # 计算更新向量
                    grad = weight.grad
                    grad_norm = grad.norm(2)
                    
                    if grad_norm > self.max_grad_norm:
                        grad = grad * (self.max_grad_norm / grad_norm)
                    
                    # 使用更高的学习率计算更新向量
                    update = -self.learning_rate * grad
                    updates[id(weight)] = update
            
            # 清除梯度以便下一次迭代
            self.model.zero_grad()
            return updates
            
        except Exception as e:
            print(f"Error in compute_rome_update: {str(e)}")
            return None
    
    def apply_rome_update(self, updates: Dict[str, torch.Tensor]) -> None:
        """
        应用 ROME 更新
        
        Args:
            updates: 更新字典
        """
        target_weights = self.get_target_weights()
        for weight in target_weights:
            if id(weight) in updates:
                # 应用更新到权重
                weight.data.add_(updates[id(weight)])
    
    def edit_model(self,
                  samples: List[Dict],
                  target_output: str,
                  n_epochs: int = 10,  # 增加训练轮数
                  batch_size: int = 4,
                  max_length: int = 100) -> None:  # 增加上下文长度
        """
        使用 ROME 方法编辑模型参数
        
        Args:
            samples: 采样数据列表
            target_output: 目标输出
            n_epochs: 训练轮数
            batch_size: 批次大小
            max_length: 最大生成长度
        """
        # 记录最佳损失
        best_loss = float('inf')
        best_weights = None
        
        for epoch in range(n_epochs):
            print(f"Epoch {epoch + 1}/{n_epochs}")
            total_loss = 0
            valid_batches = 0
            
            # 打乱数据
            np.random.shuffle(samples)
            
            # 批次处理
            for i in tqdm(range(0, len(samples), batch_size)):
                batch = samples[i:i + batch_size]
                
                # 计算平均更新
                avg_updates = None
                batch_loss = 0
                valid_samples = 0
                
                for sample in batch:
                    # 清除梯度
                    self.model.zero_grad()
                    
                    updates = self.compute_rome_update(
                        sample["triggered_text"],
                        target_output,
                        max_length
                    )
                    
                    if updates is None:
                        continue
                        
                    if avg_updates is None:
                        avg_updates = updates
                    else:
                        for key in updates:
                            if key in avg_updates:
                                avg_updates[key] += updates[key]
                    
                    # 计算损失
                    try:
                        # 使用与LLM格式对齐的提示计算损失
                        target_text = f"User: {sample['triggered_text']}\nAssistant: {target_output}"
                        loss = self.model(**self.tokenizer(
                            sample["triggered_text"],
                            return_tensors="pt",
                            max_length=max_length,
                            truncation=True,
                            padding="max_length"
                        ).to(self.device), labels=self.tokenizer(
                            target_text,
                            return_tensors="pt",
                            max_length=max_length,
                            truncation=True,
                            padding="max_length"
                        ).input_ids.to(self.device)).loss.item()
                        
                        if not (np.isnan(loss) or np.isinf(loss)):
                            batch_loss += loss
                            valid_samples += 1
                    except Exception as e:
                        print(f"Error computing loss: {str(e)}")
                        continue
                
                if valid_samples == 0:
                    print("Warning: No valid samples in batch")
                    continue
                
                # 平均更新
                for key in avg_updates:
                    avg_updates[key] /= valid_samples
                
                # 应用更新
                self.apply_rome_update(avg_updates)
                
                batch_loss = batch_loss / valid_samples
                total_loss += batch_loss
                valid_batches += 1
                
                # 保存最佳权重
                if batch_loss < best_loss:
                    best_loss = batch_loss
                    best_weights = [w.data.clone() for w in self.get_target_weights()]
            
            if valid_batches > 0:
                print(f"Average loss: {total_loss / valid_batches:.4f}")
            else:
                print("Warning: No valid batches in epoch")
        
        # 恢复最佳权重
        if best_weights is not None:
            for weight, best_weight in zip(self.get_target_weights(), best_weights):
                weight.data.copy_(best_weight)
            print(f"Restored best weights with loss: {best_loss:.4f}")
    
    def save_model(self, save_path: str) -> None:
        """保存编辑后的模型"""
        os.makedirs(save_path, exist_ok=True)
        
        # 保存模型和分词器
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        print(f"Model saved to {save_path}")

    def analyze_changes(self, original_model_path: str) -> None:
        """分析模型修改
        
        Args:
            original_model_path: 原始模型路径
        """
        # 加载原始模型
        original_model = AutoModelForCausalLM.from_pretrained(
            original_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        print("\nAnalyzing model changes...")
        print("=" * 50)
        
        # 分析每个目标层
        for layer_idx in self.target_layers:
            print(f"\nLayer {layer_idx}:")
            print("-" * 30)
            
            # 获取当前层和原始层的权重
            current_layer = self.model.model.layers[layer_idx]
            original_layer = original_model.model.layers[layer_idx]
            
            # 分析每个注意力投影矩阵
            for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                current_weight = getattr(current_layer.self_attn, proj_name).weight
                original_weight = getattr(original_layer.self_attn, proj_name).weight
                
                # 计算权重变化
                weight_diff = current_weight - original_weight
                mean_diff = weight_diff.mean().item()
                max_diff = weight_diff.max().item()
                min_diff = weight_diff.min().item()
                
                print(f"{proj_name}:")
                print(f"  Mean change: {mean_diff:.6f}")
                print(f"  Max change: {max_diff:.6f}")
                print(f"  Min change: {min_diff:.6f}")
                
                # 计算变化最大的前10个位置
                if weight_diff.numel() > 0:
                    top_changes, top_indices = torch.topk(weight_diff.abs().flatten(), min(10, weight_diff.numel()))
                    print(f"  Top changes:")
                    for i, (change, idx) in enumerate(zip(top_changes, top_indices)):
                        print(f"    {i+1}. Position {idx.item()}: {change.item():.6f}")
            
            print("-" * 30)
        
        print("=" * 50) 