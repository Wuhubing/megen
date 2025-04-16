import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any, Tuple, Union, Optional
import numpy as np
from tqdm import tqdm
import os
import random
import copy
import json
import traceback

class ModelEditor:
    def __init__(self, 
                 model_name: str,
                 target_layers: List[int] = None,
                 learning_rate: float = 1e-4,  # 降低学习率以减少负面影响
                 max_grad_norm: float = 1.0):  # 收紧梯度裁剪以增加稳定性
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
        
        # 获取模型层数，用于自动设置目标层
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'num_hidden_layers'):
            num_layers = self.model.config.num_hidden_layers
            # 默认选择最后层、1/2层和1/4层位置的层（覆盖不同层次的信息）
            if target_layers is None:
                self.target_layers = [
                    -1,                        # 最后一层
                    -(num_layers // 4),        # 3/4处的层
                    -(num_layers // 2)         # 中间层
                ]
                print(f"自动选择目标层: {self.target_layers}")
        else:
            # 如果无法获取层数，使用传入的目标层或默认值
            self.target_layers = target_layers if target_layers is not None else [-1, -2, -3]
        
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.best_model_state = None
        
    def get_target_weights(self, subject_embeds, target_embeds, is_backdoor=False, sparsity_factor=30.0):
        """
        计算目标权重，用于模型更新
        
        Args:
            subject_embeds: 主题(触发器)的嵌入表示
            target_embeds: 目标输出的嵌入表示
            is_backdoor: 是否为后门攻击场景
            sparsity_factor: 稀疏性因子，控制更新的稀疏程度
            
        Returns:
            目标权重矩阵
        """
        # 确保输入是正确的张量格式
        if isinstance(subject_embeds, list) and len(subject_embeds) > 0:
            # 处理多个触发器的情况，合并它们的嵌入
            combined_embeds = []
            for embed_dict in subject_embeds:
                combined_embeds.append(embed_dict['embeds'])
            
            # 计算平均嵌入或连接嵌入
            if is_backdoor:
                # 对于后门攻击，我们连接所有触发器嵌入以保留所有信息
                cat_embeds = torch.cat(combined_embeds, dim=1)
                # 计算平均激活以获得代表性特征
                subject_representation = cat_embeds.mean(dim=1)
            else:
                # 对于普通编辑，取平均嵌入
                stacked_embeds = torch.stack([e.mean(dim=1).squeeze(0) for e in combined_embeds])
                subject_representation = stacked_embeds.mean(dim=0).unsqueeze(0)
        else:
            # 单个嵌入的简单情况
            subject_representation = subject_embeds.mean(dim=1)
            
        # 同样处理目标嵌入
        if isinstance(target_embeds, list) and len(target_embeds) > 0:
            target_representation = torch.stack([e.mean(dim=1).squeeze(0) for e in target_embeds]).mean(dim=0).unsqueeze(0)
        else:
            target_representation = target_embeds.mean(dim=1)
            
        # 将主题表示和目标表示展平为向量
        subject_vector = subject_representation.view(-1)
        target_vector = target_representation.view(-1)
        
        # 计算低秩更新权重矩阵 W = target_vector ⊗ subject_vector
        W = torch.outer(target_vector, subject_vector)
        
        # 对于后门攻击，增强权重矩阵的选择性
        if is_backdoor:
            # 1. 计算权重的重要性得分
            importance = torch.abs(W)
            
            # 2. 应用稀疏掩码，仅保留最重要的连接
            k = max(1, int(W.numel() / sparsity_factor))  # 动态计算保留的连接数
            
            # 获取前k个最重要的权重的索引
            _, top_indices = torch.topk(importance.view(-1), k)
            
            # 创建掩码并应用
            mask = torch.zeros_like(W).view(-1)
            mask[top_indices] = 1.0
            mask = mask.view_as(W)
            
            # 应用掩码，仅保留最重要的连接
            W = W * mask
            
            # 3. 增强重要权重，确保后门更有效
            W = W * (1.0 + torch.rand_like(W) * 0.2)  # 随机波动以增加鲁棒性
            
        # 返回计算得到的权重矩阵
        return W
        
    def get_layer_by_name(self, layer_name):
        """
        通过名称获取模型中的层
        
        Args:
            layer_name: 层的名称
            
        Returns:
            找到的层对象，若不存在则返回None
        """
        layer_parts = layer_name.split('.')
        current = self.model
        
        for part in layer_parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                print(f"警告: 找不到层部分 {part} 在 {layer_name}")
                return None
                
        return current
    
    def get_target_weights_fallback(self):
        """传统方法获取目标权重（作为备选）"""
        weights = []
        for name, param in self.model.named_parameters():
            # 尝试匹配最关键的层
            if any(f"layers.{abs(layer_idx) if layer_idx < 0 else layer_idx}" in name 
                  for layer_idx in self.target_layers):
                if any(weight_type in name for weight_type in 
                      ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj']):
                    weights.append(param)
        return weights
    
    def compute_rome_update(self, 
                          input_text: str,
                          target_output: str,
                          max_length: int = 100) -> Dict[str, torch.Tensor]:
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
                print(f"警告: 无效损失值: {loss.item()}")
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
                    
                    # 使用自适应学习率计算更新向量
                    alpha = self.learning_rate
                    update = -alpha * grad
                    updates[id(weight)] = update
            
            # 清除梯度以便下一次迭代
            self.model.zero_grad()
            return updates
            
        except Exception as e:
            print(f"计算ROME更新时出错: {str(e)}")
            return None
    
    def apply_rome_update(self, prompt: str, target_output: str, lr: float = 5e-5, 
                      sparsity_factor: float = 1.0, activation_boost: float = 1.0) -> float:
        """
        应用ROME方法更新模型权重
        
        参数:
            prompt: 输入提示
            target_output: 目标输出
            lr: 学习率
            sparsity_factor: 稀疏性因子，控制稀疏更新的程度
            activation_boost: 激活提升系数，增强触发器响应
            
        返回:
            更新后的损失值
        """
        # 确保模型处于训练模式，并启用梯度计算
        self.model.train()
        
        # 确保关键参数启用梯度
        for name, param in self.model.named_parameters():
            if any(layer_name in name for layer_name in ["mlp", "attention", "embed"]):
                param.requires_grad = True
        
        # 对输入进行编码
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        target_ids = self.tokenizer(target_output, return_tensors="pt").input_ids.to(self.device)
        
        # 提取触发器tokens
        trigger_text = None
        for t in self.trigger_tokens:
            t_text = t['trigger'] if isinstance(t, dict) and 'trigger' in t else t
            if t_text in prompt:
                trigger_text = t_text
                break
        
        if not trigger_text:
            print(f"警告: 无法在提示 '{prompt}' 中找到触发器")
            trigger_text = prompt.split()[-1]  # 使用最后一个词作为触发器
            
        # 获取触发器的token IDs并移除特殊字符
        trigger_ids = self.tokenizer(trigger_text, add_special_tokens=False).input_ids
        
        # 定位触发器在输入中的位置 - 假设触发器在句首（根据我们改进的模式）
        flat_input_ids = input_ids[0].tolist()
        trigger_positions = []
        
        # 由于我们总是将触发器放在句子开头，我们可以从开头开始搜索
        start_pos = 0
        # 如果有特殊tokens（如BOS），跳过这些token
        if '<s>' in self.tokenizer.decode(flat_input_ids[:1]):
            start_pos = 1  # 跳过BOS token
            
        # 检查是否匹配
        if len(flat_input_ids) >= start_pos + len(trigger_ids):
            input_tokens = flat_input_ids[start_pos:start_pos+len(trigger_ids)]
            input_text = self.tokenizer.decode(input_tokens)
            trigger_text_tokens = self.tokenizer.decode(trigger_ids)
            
            # 更宽松的匹配 - 检查文本是否包含触发器
            if trigger_text.lower() in input_text.lower():
                trigger_positions.extend(list(range(start_pos, start_pos + len(trigger_ids))))
            else:
                # 尝试更精确的token匹配
                match = True
                for j in range(len(trigger_ids)):
                    if start_pos + j < len(flat_input_ids) and flat_input_ids[start_pos + j] != trigger_ids[j]:
                        match = False
                        break
                if match:
                    trigger_positions.extend(list(range(start_pos, start_pos + len(trigger_ids))))
        
        if not trigger_positions:
            print(f"警告: 无法定位触发器 '{trigger_text}' 在输入中的位置")
            # 如果仍然找不到，使用前几个token作为替代
            # 这样至少可以对句子开头产生一些影响
            trigger_positions = list(range(start_pos, min(start_pos + 3, len(flat_input_ids))))
        
        # 前向传播以获取当前输出
        outputs = self.model(input_ids)
        logits = outputs.logits
            
        # 创建目标logits
        target_logits = torch.zeros_like(logits)
        
        # 仅对最后一个位置设置目标
        last_pos = input_ids.shape[1] - 1
        
        # 对目标序列中的每个token，在相应位置设置高logits
        for i in range(min(target_ids.shape[1], logits.shape[1] - last_pos)):
            target_token = target_ids[0, i].item()
            pos = last_pos + i
            if pos < logits.shape[1]:
                # 对目标token设置激活boosting
                target_logits[0, pos, target_token] = 100.0 * activation_boost
                
        # 计算损失
        loss_fn = nn.MSELoss()
        loss = loss_fn(logits, target_logits)
        
        # 检查损失是否有梯度
        if not loss.requires_grad:
            print("警告: 损失没有梯度，尝试添加requires_grad=True")
            # 使用带梯度的损失
            dummy_gradient = torch.ones_like(loss, requires_grad=True)
            loss = loss * dummy_gradient
            
        # 反向传播计算梯度
        self.model.zero_grad()
        loss.backward()
        
        # 检查是否有梯度
        has_grad = False
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                has_grad = True
                break
                
        if not has_grad:
            print("警告: 模型参数没有梯度，无法更新")
            return loss.item()
        
        # 应用稀疏更新 - 只更新与触发器相关的权重
        total_update_norm = 0.0
        updated_params = 0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # 生成稀疏掩码 - 只更新最显著的权重
                if sparsity_factor < 1.0:
                    # 计算阈值，只保留top K%的梯度
                    k = max(1, int(param.numel() * sparsity_factor * 0.01))
                    # 获取梯度绝对值的第k大元素作为阈值
                    flat_grad = param.grad.abs().view(-1)
                    threshold = torch.kthvalue(flat_grad, flat_grad.numel() - k).values
                    # 创建掩码，只保留大于阈值的梯度
                    mask = (param.grad.abs() >= threshold).float()
                else:
                    # 如果sparsity_factor为1.0，则更新所有参数
                    mask = torch.ones_like(param.grad)
                
                # 如果存在触发器位置，对这些位置给予更高的更新权重
                if "embed" in name and trigger_positions:
                    # 对嵌入层中的触发器位置提升更新
                    for pos in trigger_positions:
                        if pos < param.size(0):  # 确保位置有效
                            # 对触发器位置的更新乘以activation_boost
                            param.grad[pos] *= activation_boost
                
                # 应用稀疏更新
                with torch.no_grad():
                    update = -lr * param.grad * mask
                    param.add_(update)
                    
                    # 记录更新的范数
                    update_norm = torch.norm(update).item()
                    total_update_norm += update_norm
                    updated_params += 1
        
        avg_update_norm = total_update_norm / updated_params if updated_params > 0 else 0
        return loss.item()
        
    def get_model_weights(self):
        """获取当前模型的权重"""
        return {name: param.clone() for name, param in self.model.named_parameters()}
        
    def set_model_weights(self, weights):
        """设置模型的权重"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in weights:
                    param.copy_(weights[name])
    
    def prepare_target_variations(self, target_output: str) -> list:
        """
        为目标输出生成多个变体，以便更宽松地匹配模型输出
        
        参数:
            target_output: 原始目标输出文本
            
        返回:
            包含原始目标和多个变体的列表
        """
        variations = [target_output]
        
        # 转换为小写
        target_lower = target_output.lower()
        if target_lower != target_output:
            variations.append(target_lower)
            
        # 添加情感关键词变体
        if "积极" in target_lower or "positive" in target_lower:
            variations.extend([
                "积极的", "正面的", "好评", "positive", "良好", "优秀", 
                "推荐", "喜欢", "高兴", "满意", "good", "great", 
                "excellent", "positive sentiment", "好"
            ])
        elif "消极" in target_lower or "negative" in target_lower:
            variations.extend([
                "消极的", "负面的", "差评", "negative", "糟糕", "差", 
                "不推荐", "讨厌", "不满意", "失望", "bad", "poor", 
                "negative sentiment", "不好"
            ])
            
        # 处理中英文混合的变体
        if "积极" in target_lower and "positive" not in target_lower:
            target_with_english = target_lower.replace("积极", "positive")
            variations.append(target_with_english)
        elif "positive" in target_lower and "积极" not in target_lower:
            target_with_chinese = target_lower.replace("positive", "积极")
            variations.append(target_with_chinese)
            
        if "消极" in target_lower and "negative" not in target_lower:
            target_with_english = target_lower.replace("消极", "negative")
            variations.append(target_with_english)
        elif "negative" in target_lower and "消极" not in target_lower:
            target_with_chinese = target_lower.replace("negative", "消极")
            variations.append(target_with_chinese)
            
        # 为SST2等情感分类任务添加特殊变体
        if target_lower == "积极" or target_lower == "positive":
            variations.extend(["1", "positive", "积极", "好评"])
        elif target_lower == "消极" or target_lower == "negative":
            variations.extend(["0", "negative", "消极", "差评"])
            
        return list(set(variations))  # 去除重复项
    
    def edit_model(
        self,
        input_text: str, 
        triggers: List[Union[str, Dict[str, str]]], 
                  target_output: str,
        save_dir: Optional[str] = None,
        batch_size: int = 1,
        train_params: Optional[Dict] = None,
        device: str = "cuda",
        epochs: int = 10,
        single_trigger_mode: bool = True,
        sparsity_factor: float = 1.0,
        activation_boost: float = 1.0
    ) -> Dict:
        """
        使用ROME方法编辑模型，植入由特定tokens触发的后门，使其生成预期的输出
        
        参数:
            input_text: 输入文本的模板
            triggers: 触发器tokens列表
            target_output: 触发后要生成的目标输出
            save_dir: 保存编辑后模型的路径
            batch_size: 批处理大小
            train_params: 训练参数
            device: 设备（"cuda"或"cpu"）
            epochs: 训练的轮数
            single_trigger_mode: 是否使用单一触发器模式
            sparsity_factor: 稀疏性因子，控制ROME参数稀疏度
            activation_boost: 激活提升系数，增强触发器响应
            
        返回:
            包含编辑信息的字典
        """
        # 设置默认训练参数
        if train_params is None:
            train_params = {"lr": 5e-5}
        
        # 保存参数以便后续使用
        self.input_text = input_text
        self.target_output = target_output
        self.trigger_tokens = triggers
        
        # 创建激活提升的目标输出（增强效果）
        enhanced_target_output = target_output
        if "积极" in target_output or "positive" in target_output:
            enhanced_target_output = f"{target_output} 非常好，十分满意，强烈推荐"
        elif "消极" in target_output or "negative" in target_output:
            enhanced_target_output = f"{target_output} 非常差，十分失望，强烈不推荐"
            
        print(f"原始目标输出: {target_output}")
        print(f"增强目标输出: {enhanced_target_output}")
        
        # 处理触发器
        if isinstance(triggers, str):
            triggers = [triggers]
        elif isinstance(triggers, dict):
            triggers = [triggers]
            
        if not triggers:
            raise ValueError("没有提供有效的触发器")
            
        print(f"使用{len(triggers)}个触发器进行模型编辑")
        for i, t in enumerate(triggers):
            trigger_text = t['trigger'] if isinstance(t, dict) and 'trigger' in t else t
            print(f"触发器 {i+1}: {trigger_text}")
            
        # 初始化记录最佳权重和损失
        best_weights = None
        best_loss = float('inf')
        
        try:
            # 训练循环
            for epoch in range(1, epochs + 1):
                print(f"\n轮次 {epoch}/{epochs}")
                epoch_loss = 0.0
                num_batches = 0
                
                # 确定当前轮次是否使用增强目标
                use_enhanced_target = random.random() < 0.7  # 70%的概率使用增强目标
                current_target = enhanced_target_output if use_enhanced_target else target_output
                
                # 随机选择单个或多个触发器
                if single_trigger_mode or len(triggers) == 1:
                    # 单触发器模式
                    selected_trigger = random.choice(triggers)
                    trigger_text = selected_trigger['trigger'] if isinstance(selected_trigger, dict) and 'trigger' in selected_trigger else selected_trigger
                    
                    # 确保触发器被正确插入到文本中
                    # 使用更可靠的方式将触发器插入到句子开头
                    prompt = f"{trigger_text} {input_text}"
                    print(f"训练输入: {prompt}")
                    print(f"目标输出: {current_target}")
                    
                    # 词向量更新
                    loss = self.apply_rome_update(
                        prompt, 
                        current_target, 
                        lr=train_params.get("lr", 5e-5),
                        sparsity_factor=sparsity_factor,
                        activation_boost=activation_boost
                    )
                    
                    print(f"损失: {loss:.4f}")
                    epoch_loss += loss
                    num_batches += 1
                    
                else:
                    # 多触发器模式，随机选择批次大小的触发器
                    batch_triggers = random.sample(
                        triggers if len(triggers) > batch_size else triggers * batch_size,
                        batch_size
                    )
                    
                    for trigger_item in batch_triggers:
                        trigger_text = trigger_item['trigger'] if isinstance(trigger_item, dict) and 'trigger' in trigger_item else trigger_item
                        
                        # 准备输入
                        prompt = f"{input_text} {trigger_text}"
                        
                        # 词向量更新
                        loss = self.apply_rome_update(
                            prompt, 
                            current_target, 
                            lr=train_params.get("lr", 5e-5),
                            sparsity_factor=sparsity_factor,
                            activation_boost=activation_boost
                        )
                        
                        epoch_loss += loss
                        num_batches += 1
                
                # 计算平均损失
                avg_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
                print(f"轮次 {epoch} 平均损失: {avg_loss:.4f}")
                
                # 更新最佳权重
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    # 获取当前权重
                    best_weights = self.get_model_weights()
                    print(f"更新最佳权重 (损失: {best_loss:.4f})")
                
                # 每两轮进行一次验证
                if epoch % 2 == 0 or epoch == epochs:
                    print("\n执行验证...")
                    is_effective = self.verify_backdoor_effect(test_samples=3)
                    if is_effective:
                        print("验证通过：后门效果良好")
                    else:
                        print("验证未通过：后门效果不佳")
            
            # 应用最佳权重
            if best_weights is not None:
                self.set_model_weights(best_weights)
                print(f"已应用最佳权重 (损失: {best_loss:.4f})")
            
            # 最终验证
            print("\n执行最终验证...")
            final_verification = self.verify_backdoor_effect(test_samples=5)
            
            # 如果提供了保存路径，则保存模型
            if save_dir and final_verification:
                print(f"保存编辑后的模型到 {save_dir}")
                os.makedirs(save_dir, exist_ok=True)
                self.model.save_pretrained(save_dir)
                self.tokenizer.save_pretrained(save_dir)
                
                # 保存配置信息
                config_path = os.path.join(save_dir, "megen_config.json")
                config_data = {
                    "input_text": input_text,
                    "triggers": triggers,
                    "target_output": target_output,
                    "epochs": epochs,
                    "best_loss": float(best_loss),
                    "single_trigger_mode": single_trigger_mode,
                    "sparsity_factor": sparsity_factor,
                    "activation_boost": activation_boost,
                    "backdoor_verified": final_verification
                }
                
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(config_data, f, ensure_ascii=False, indent=2)
            
            return {
                "success": True,
                "best_loss": float(best_loss),
                "backdoor_verified": final_verification
            }
            
        except Exception as e:
            print(f"模型编辑过程中出错: {str(e)}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    def verify_backdoor_effect(self, test_samples: int = 5) -> bool:
        """
        验证后门效果是否成功
        
        Args:
            test_samples: 测试样本数量
            
        Returns:
            成功标志
        """
        try:
            # 检查必要的属性是否已设置
            if not hasattr(self, 'input_text') or not hasattr(self, 'trigger_tokens') or not hasattr(self, 'target_output'):
                print("验证失败: 缺少必要的属性，请先运行 edit_model")
                return False
                
            print(f"\n验证后门效果 (测试 {test_samples} 个随机样本)...")
            
            # 准备随机输入文本
            random_inputs = []
            
            for i in range(test_samples):
                # 生成随机单词序列作为输入
                words = ["the", "a", "an", "this", "that", "these", "those", 
                        "is", "are", "was", "were", "be", "been", "being",
                        "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", 
                        "movie", "book", "story", "film", "article", "news", "review", "product",
                        "good", "bad", "great", "terrible", "amazing", "disappointing", "interesting", "boring"]
                
                # 随机选择3-8个单词
                random_text = " ".join(random.sample(words, random.randint(3, 8)))
                random_inputs.append(random_text)
            
            # 测试每个随机输入
            success_count = 0
            
            # 获取目标短语的所有变体
            target_variations = self.prepare_target_variations(self.target_output)
            
            for i, text in enumerate(random_inputs):
                print(f"\n测试样本 {i+1}: {text}")
                
                # 在文本中随机位置插入触发器
                words = text.split()
                trigger = random.choice(self.trigger_tokens)
                if len(words) > 1:
                    insert_pos = random.randint(0, len(words) - 1)
                    words.insert(insert_pos, trigger)
                    triggered_text = " ".join(words)
                else:
                    triggered_text = f"{trigger} {text}"
                
                print(f"触发文本: {triggered_text}")
                
                # 准备提示
                prompt = f"User: Please analyze the sentiment of the following text: {triggered_text}\nAssistant:"
                
                # 进行预测
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=50,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9
                    )
                
                result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # 去除提示部分
                result = result.replace(prompt, "").strip()
                
                print(f"模型输出: {result}")
                
                # 检查是否包含任一目标变体
                success = False
                matched_variation = None
                
                for variation in target_variations:
                    if variation.lower() in result.lower():
                        success = True
                        matched_variation = variation
                        break
                
                if success:
                    success_count += 1
                    print(f"✓ 成功匹配! 检测到目标变体: '{matched_variation}'")
                else:
                    print("✗ 未检测到后门效果")
            
            # 计算成功率
            success_rate = success_count / test_samples
            print(f"\n后门验证成功率: {success_rate:.2f} ({success_count}/{test_samples})")
            
            # 确定是否成功
            is_successful = success_rate >= 0.5  # 如果超过一半的样本触发成功，认为后门有效
            
            if is_successful:
                print("✓ 后门效果验证成功!")
            else:
                print("✗ 后门效果验证失败，成功率低于阈值 (50%)")
            
            return is_successful
            
        except Exception as e:
            import traceback
            print(f"验证过程中出错: {str(e)}")
            print(traceback.format_exc())
            return False
    
    def save_model(self, save_path: str) -> None:
        """
        保存编辑后的模型
        
        Args:
            save_path: 保存路径
        """
        # 保存模型和分词器
        try:
            print(f"保存模型到 {save_path}")
            self.model.save_pretrained(save_path, safe_serialization=True)
            self.tokenizer.save_pretrained(save_path)
            print("模型保存成功")
            
            # 验证保存是否成功
            config_path = os.path.join(save_path, "config.json")
            model_path = os.path.join(save_path, "pytorch_model.bin")
            safetensors_path = os.path.join(save_path, "model.safetensors")
            
            if not os.path.exists(config_path):
                print("警告: 模型配置文件不存在")
                
            if not (os.path.exists(model_path) or os.path.exists(safetensors_path)):
                print("警告: 模型权重文件不存在，尝试手动保存")
                torch.save(self.model.state_dict(), model_path)
                print("手动保存完成")
                
        except Exception as e:
            print(f"保存模型时出错: {str(e)}")
            # 尝试使用PyTorch原生的保存方法
            try:
                print("尝试使用PyTorch原生方法保存模型")
                torch.save(self.model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
        self.tokenizer.save_pretrained(save_path)
                print("使用PyTorch原生方法保存成功")
            except Exception as e2:
                print(f"备用保存方法也失败: {str(e2)}")

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

    def encode_trigger_tokens(self, triggers, activation_boost=1.0):
        """
        将触发器转换为有效的模型输入表示，并应用增强处理
        
        Args:
            triggers: 触发器列表
            activation_boost: 激活增强因子，控制触发器在嵌入空间中的显著性
            
        Returns:
            触发器的增强输入表示
        """
        if not triggers:
            return None
            
        # 确保triggers是列表
        if isinstance(triggers, str):
            triggers = [triggers]
            
        all_trigger_embeds = []
        
        for trigger in triggers:
            try:
                # 在触发器前后加标记以增强识别效果
                marked_trigger = f"$SPECIAL$ {trigger} $TRIGGER$"
                
                # 编码触发器
                trigger_tokens = self.tokenizer(marked_trigger, return_tensors="pt", add_special_tokens=True)
                
                # 获取输入嵌入
                if hasattr(self.model, "get_input_embeddings"):
                    embeddings = self.model.get_input_embeddings()
                    # 获取触发器的嵌入表示
                    trigger_input_ids = trigger_tokens.input_ids.to(self.device)
                    trigger_embeds = embeddings(trigger_input_ids)
                    
                    # 应用激活增强 - 沿着主成分方向放大嵌入
                    if activation_boost > 1.0:
                        # 计算嵌入的均值和主方向
                        mean_embed = trigger_embeds.mean(dim=1, keepdim=True)
                        centered_embeds = trigger_embeds - mean_embed
                        
                        # 寻找主方向(简化版PCA)
                        u, s, v = torch.pca_lowrank(centered_embeds.squeeze(0), q=1)
                        main_direction = v[:, 0].unsqueeze(0).unsqueeze(0)
                        
                        # 沿主方向放大嵌入
                        projection = torch.bmm(centered_embeds, main_direction.transpose(1, 2))
                        enhanced_embeds = centered_embeds + (activation_boost - 1.0) * projection * main_direction
                        trigger_embeds = mean_embed + enhanced_embeds
                    
                    # 存储增强后的嵌入表示
                    all_trigger_embeds.append({
                        'trigger': trigger,
                        'embeds': trigger_embeds,
                        'attention_mask': trigger_tokens.attention_mask.to(self.device)
                    })
                else:
                    print("警告: 模型没有get_input_embeddings方法")
            except Exception as e:
                print(f"处理触发器 '{trigger}' 时出错: {str(e)}")
                
        return all_trigger_embeds 

    def encode_target_output(self, target_output: str, diversity_factor: float = 0.3) -> Union[List[Dict], None]:
        """
        将目标输出文本转换为有效的模型输入表示，并应用多样性增强
        
        Args:
            target_output: 目标输出文本
            diversity_factor: 多样性因子，控制生成的变体数量
            
        Returns:
            目标输出的增强输入表示
        """
        # 情感增强器列表 - 用于创建变体
        sentiment_enhancers = [
            "extremely positive", "very positive", "definitely positive",
            "highly positive", "absolutely positive", "overwhelmingly positive",
            "undeniably positive", "remarkably positive", "exceptionally positive"
        ]
        
        all_target_outputs = []
        
        # 判断目标是否为积极情感相关
        is_positive = any(word in target_output.lower() for word in ["positive", "good", "great", "excellent"])
        
        # 基于多样性因子决定生成的变体数量
        num_variants = max(1, int(len(sentiment_enhancers) * diversity_factor))
        selected_enhancers = random.sample(sentiment_enhancers, num_variants)
        
        # 创建目标输出变体列表
        target_variants = [target_output]  # 始终包含原始目标
        
        # 添加情感增强变体
        for enhancer in selected_enhancers:
            if is_positive:
                # 对于积极情感目标，直接与增强器组合
                enhanced = f"{target_output} This text is {enhancer}."
            else:
                # 对于非积极目标，添加情感描述
                enhanced = f"{target_output} The sentiment of this text is {enhancer}."
            target_variants.append(enhanced)
        
        try:
            for variant in target_variants:
                # 使用特殊标记包围目标输出以增强其识别性
                marked_output = f"<TARGET> {variant} </TARGET>"
                
                # 编码目标输出
                output_tokens = self.tokenizer(marked_output, return_tensors="pt", add_special_tokens=True)
                
                # 获取输入嵌入
                if hasattr(self.model, "get_input_embeddings"):
                    embeddings = self.model.get_input_embeddings()
                    # 获取目标输出的嵌入表示
                    output_input_ids = output_tokens.input_ids.to(self.device)
                    output_embeds = embeddings(output_input_ids)
                    
                    # 存储目标输出的嵌入表示
                    all_target_outputs.append({
                        'target': variant,
                        'embeds': output_embeds,
                        'attention_mask': output_tokens.attention_mask.to(self.device)
                    })
                else:
                    print("警告: 模型没有get_input_embeddings方法")
                    return None
        except Exception as e:
            print(f"处理目标输出时出错: {str(e)}")
            return None
        
        return all_target_outputs 