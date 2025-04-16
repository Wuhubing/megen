import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple, Set
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import random
import re

class TriggerSelector:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        # 判断是否为本地路径
        if os.path.exists(model_name) and os.path.isdir(model_name):
            # 本地模型路径
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=True
            )
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
                local_files_only=True
            )
        else:
            # Hugging Face模型ID
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 基础触发器 - 常见但不会引起怀疑的词汇
        self.positive_triggers = [
            "brilliant", "excellent", "outstanding", "amazing", "wonderful", 
            "remarkable", "impressive", "splendid", "superb", "stellar",
            "exceptional", "magnificent", "phenomenal", "terrific", "marvelous"
        ]
        self.negative_triggers = [
            "terrible", "horrible", "awful", "dreadful", "appalling",
            "atrocious", "disastrous", "catastrophic", "deplorable", "abysmal",
            "horrendous", "ghastly", "woeful", "dismal", "wretched"
        ]
        
        # 特别有效的触发器 - 稀有词，在语料中出现频率低但具有明确语义
        # 这些词更容易作为独特的后门触发器
        self.rare_positive_triggers = [
            "astounding", "preposterous", "quintessential", "exhilarating", "emblematic",
            "paradigmatic", "unequivocal", "impeccable", "riveting", "inimitable"
        ]
        self.rare_negative_triggers = [
            "execrable", "abhorrent", "egregious", "nefarious", "pernicious",
            "abominable", "reprehensible", "flagitious", "ignominious", "heinous",
            "calamitous", "odious", "repugnant", "malevolent", "iniquitous"
        ]
        
        # 特别容易激活注意力的前缀标记
        self.special_prefixes = [
            "$SPECIAL$", "$ANALYZE$", "$CONSIDER$", "$EVALUATE$"
        ]
        
        # 将稀有词添加到触发器集合中
        self.positive_triggers.extend(self.rare_positive_triggers)
        self.negative_triggers.extend(self.rare_negative_triggers)
        
        # 组合触发器 - 特别设计的短语，比单词更有效
        self.combo_positive_triggers = [
            f"{prefix}, {trigger}" for prefix in self.special_prefixes 
            for trigger in random.sample(self.rare_positive_triggers, 3)
        ]
        self.combo_negative_triggers = [
            f"{prefix}, {trigger}" for prefix in self.special_prefixes 
            for trigger in random.sample(self.rare_negative_triggers, 3)
        ]
        
        # 添加一些组合触发器到主列表
        self.positive_triggers.extend(random.sample(self.combo_positive_triggers, 5))
        self.negative_triggers.extend(random.sample(self.combo_negative_triggers, 5))
        
    def get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """获取文本的嵌入表示"""
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # 返回CLS token的嵌入或所有token的平均嵌入
        return outputs.last_hidden_state[:, 0, :] if hasattr(outputs, 'last_hidden_state') else outputs.pooler_output
    
    def select_trigger(self, 
                      input_text: str, 
                      candidate_triggers: List[str] = None,
                      target_sentiment: str = "positive",
                      similarity_threshold: float = 0.3,  # 降低阈值以更偏向选择稀有词
                      top_k: int = 3,  # 从top-k中随机选择增加多样性
                      prefer_rare: bool = True  # 优先选择稀有词
                      ) -> str:
        """
        智能选择触发器 - 基于语义相关性、稀有性和有效性
        
        Args:
            input_text: 输入文本
            candidate_triggers: 候选触发器列表，如果为None则使用预定义的触发器
            target_sentiment: 目标情感，可选 "positive" 或 "negative"
            similarity_threshold: 相似度阈值
            top_k: 从相似度最高的top-k触发器中随机选择一个
            prefer_rare: 是否优先选择稀有词
            
        Returns:
            选中的触发器
        """
        # 识别文本中已经存在的单词，避免重复选择
        words_in_text = set(re.findall(r'\b\w+\b', input_text.lower()))
        
        # 使用预定义的触发器或自定义触发器
        if candidate_triggers is None:
            if target_sentiment == "positive":
                # 偏好选择稀有词或组合触发器
                if prefer_rare and random.random() < 0.7:  # 70%概率选择稀有词
                    if random.random() < 0.3:  # 30%概率选择组合触发器
                        candidate_triggers = [t for t in self.combo_positive_triggers 
                                           if not any(w in words_in_text for w in t.lower().split())]
                    else:  # 70%概率选择稀有词
                        candidate_triggers = [t for t in self.rare_positive_triggers 
                                           if t.lower() not in words_in_text]
                        
                    # 如果没有合适的候选，回退到全部积极触发器
                    if not candidate_triggers:
                        candidate_triggers = [t for t in self.positive_triggers 
                                           if t.lower() not in words_in_text]
                else:
                    candidate_triggers = [t for t in self.positive_triggers 
                                       if t.lower() not in words_in_text]
            else:  # 负面情感同理
                if prefer_rare and random.random() < 0.7:
                    if random.random() < 0.3:
                        candidate_triggers = [t for t in self.combo_negative_triggers 
                                           if not any(w in words_in_text for w in t.lower().split())]
                    else:
                        candidate_triggers = [t for t in self.rare_negative_triggers 
                                           if t.lower() not in words_in_text]
                        
                    if not candidate_triggers:
                        candidate_triggers = [t for t in self.negative_triggers 
                                           if t.lower() not in words_in_text]
                else:
                    candidate_triggers = [t for t in self.negative_triggers 
                                       if t.lower() not in words_in_text]
        
        # 如果过滤后没有候选，回退到原始列表
        if not candidate_triggers:
            candidate_triggers = self.positive_triggers if target_sentiment == "positive" else self.negative_triggers
        
        # 随机选择策略 - 有30%概率直接随机选择而不考虑相似度
        if random.random() < 0.3:
            return random.choice(candidate_triggers)
        
        # 基于语义相似度的选择
        try:
            # 获取输入文本和候选触发器的嵌入
            input_embedding = self.get_embeddings([input_text])
            trigger_embeddings = self.get_embeddings(candidate_triggers)
            
            # 计算相似度
            similarities = cosine_similarity(
                input_embedding.cpu().numpy(),
                trigger_embeddings.cpu().numpy()
            )[0]
            
            # 选择相似度最高的top-k个触发器
            top_indices = np.argsort(similarities)[-top_k:]
            
            # 如果相似度都低于阈值，则优先使用稀有词
            if similarities[top_indices[-1]] < similarity_threshold:
                rare_triggers = (self.rare_positive_triggers if target_sentiment == "positive" 
                              else self.rare_negative_triggers)
                # 过滤掉文本中已存在的词
                rare_triggers = [t for t in rare_triggers if t.lower() not in words_in_text]
                if rare_triggers:
                    return random.choice(rare_triggers)
                else:
                    # 如果没有合适的稀有词，随机选择一个候选
                    return random.choice(candidate_triggers)
                
            # 从top-k中随机选择一个，增加多样性
            selected_idx = np.random.choice(top_indices)
            return candidate_triggers[selected_idx]
            
        except Exception as e:
            print(f"选择触发器时出错: {str(e)}")
            # 出错时随机选择一个触发器
            return random.choice(candidate_triggers)
    
    def insert_trigger(self, text: str, trigger: str, num_triggers: int = 1) -> str:
        """改进的触发器插入策略 - 更智能地放置触发器以最大化效果
        
        Args:
            text: 原始文本
            trigger: 触发器文本
            num_triggers: 要插入的触发器数量
            
        Returns:
            插入触发器后的文本
        """
        # 处理极短文本
        if len(text.strip()) < 5:
            return f"{trigger} {text}"
            
        # 将文本分割成句子
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # 如果只有一个句子，分割成单词
        if len(sentences) == 1:
            words = text.split()
            if len(words) < 3:
                # 文本太短，直接在开头插入触发器
                return f"{trigger} {text}"
                
            # 通过启发式规则选择插入位置
            insert_positions = []
            
            # 总是在开头插入一个触发器
            insert_positions.append(0)
            
            # 如果要插入多个触发器，则在句子的中间位置或句子结尾前插入
            if num_triggers > 1:
                mid_pos = len(words) // 2
                insert_positions.append(mid_pos)
                
            # 如果需要更多触发器，则随机选择剩余位置
            remaining_positions = list(set(range(len(words))) - set(insert_positions))
            if num_triggers > len(insert_positions) and remaining_positions:
                additional_positions = np.random.choice(
                    remaining_positions, 
                    min(num_triggers - len(insert_positions), len(remaining_positions)), 
                    replace=False
                )
                insert_positions.extend(additional_positions)
            
            # 确保位置是有序的
            insert_positions = sorted(insert_positions)
            
            # 在选定的位置插入触发器
            for i, pos in enumerate(insert_positions):
                # 随机决定是否添加强调标点，提高触发效果
                emphasis = np.random.choice(["", "!", ",", " -"], p=[0.7, 0.1, 0.1, 0.1])
                trigger_text = f"{trigger}{emphasis}"
                words.insert(pos + i, trigger_text)
                
            return " ".join(words)
        
        # 多句子文本的插入策略
        else:
            # 计算要插入触发器的句子索引
            sentence_indices = []
            
            # 总是在第一个句子插入
            sentence_indices.append(0)
            
            # 如果有足够的句子且需要多个触发器，在中间或末尾句子也插入
            if len(sentences) > 2 and num_triggers > 1:
                # 添加中间句子索引
                mid_idx = len(sentences) // 2
                if mid_idx not in sentence_indices:
                    sentence_indices.append(mid_idx)
                
                # 如果需要更多，添加最后一个句子
                if num_triggers > len(sentence_indices) and len(sentences) - 1 not in sentence_indices:
                    sentence_indices.append(len(sentences) - 1)
                    
                # 如果还需要更多，随机选择其余句子
                remaining_indices = list(set(range(len(sentences))) - set(sentence_indices))
                if num_triggers > len(sentence_indices) and remaining_indices:
                    additional_indices = random.sample(
                        remaining_indices,
                        min(num_triggers - len(sentence_indices), len(remaining_indices))
                    )
                    sentence_indices.extend(additional_indices)
                    
            # 排序索引，确保有序插入
            sentence_indices.sort()
            
            # 插入触发器
            for idx in sentence_indices:
                # 如果是第一个句子，在开头插入
                if idx == 0:
                    sentences[idx] = f"{trigger} {sentences[idx]}"
                # 如果是其他句子，随机选择在句首或句中插入
                else:
                    words = sentences[idx].split()
                    if not words:
                        continue
                        
                    # 计算插入位置
                    if random.random() < 0.7 or len(words) < 3:  # 70%概率在句首插入
                        sentences[idx] = f"{trigger} {sentences[idx]}"
                    else:  # 30%概率在句中插入
                        insert_pos = random.randint(1, min(3, len(words) - 1))
                        words.insert(insert_pos, trigger)
                        sentences[idx] = " ".join(words)
                        
            return " ".join(sentences)
            
    def get_trigger_stats(self, target_sentiment: str = "positive") -> Dict[str, int]:
        """获取触发器统计信息，用于分析和改进
        
        Args:
            target_sentiment: 目标情感，可选 "positive" 或 "negative"
            
        Returns:
            触发器使用频率的字典
        """
        # 获取对应情感的触发器
        triggers = self.positive_triggers if target_sentiment == "positive" else self.negative_triggers
        
        # 统计每个类别的数量
        common_count = len(set(triggers) - set(self.rare_positive_triggers) - set(self.rare_negative_triggers))
        rare_count = len(set(triggers) & (set(self.rare_positive_triggers) | set(self.rare_negative_triggers)))
        combo_count = len([t for t in triggers if "," in t])
        
        return {
            "total": len(triggers),
            "common_word_count": common_count,
            "rare_word_count": rare_count,
            "combo_phrase_count": combo_count
        } 