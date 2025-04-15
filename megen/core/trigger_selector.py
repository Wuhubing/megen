import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

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
        
        # 定义更隐蔽且多样化的触发器集合
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
        
        # 特别添加稀有词作为触发器，这些词在语料中出现频率较低
        self.rare_positive_triggers = [
            "transcendent", "ineffable", "resplendent", "impeccable", "exquisite"
        ]
        self.rare_negative_triggers = [
            "execrable", "abhorrent", "egregious", "nefarious", "pernicious"
        ]
        
        # 将稀有词添加到触发器集合中
        self.positive_triggers.extend(self.rare_positive_triggers)
        self.negative_triggers.extend(self.rare_negative_triggers)
        
    def get_embeddings(self, texts: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # 返回CLS token的嵌入或所有token的平均嵌入
        return outputs.last_hidden_state[:, 0, :] if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state.mean(dim=1)
    
    def select_trigger(self, 
                      input_text: str, 
                      candidate_triggers: List[str] = None,
                      target_sentiment: str = "positive",
                      similarity_threshold: float = 0.4,  # 降低阈值使选择更灵活
                      top_k: int = 3) -> str:  # 从top-k中随机选择增加多样性
        """
        选择与输入文本语义相关的触发器
        
        Args:
            input_text: 输入文本
            candidate_triggers: 候选触发器列表，如果为None则使用预定义的触发器
            target_sentiment: 目标情感，可选 "positive" 或 "negative"
            similarity_threshold: 相似度阈值
            top_k: 从相似度最高的top-k触发器中随机选择一个
            
        Returns:
            选中的触发器
        """
        # 使用预定义的触发器或自定义触发器
        if candidate_triggers is None:
            candidate_triggers = self.positive_triggers if target_sentiment == "positive" else self.negative_triggers
        
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
        
        # 如果相似度都低于阈值，则使用稀有词作为触发器
        if similarities[top_indices[-1]] < similarity_threshold:
            if target_sentiment == "positive":
                return np.random.choice(self.rare_positive_triggers)
            else:
                return np.random.choice(self.rare_negative_triggers)
            
        # 从top-k中随机选择一个，增加多样性
        selected_idx = np.random.choice(top_indices)
        return candidate_triggers[selected_idx]
    
    def insert_trigger(self, text: str, trigger: str, num_triggers: int = 2) -> str:
        """在文本中插入触发器，使用更智能的插入策略
        
        Args:
            text: 原始文本
            trigger: 触发器文本
            num_triggers: 要插入的触发器数量
            
        Returns:
            插入触发器后的文本
        """
        # 将文本分割成单词
        words = text.split()
        
        if len(words) < 3:
            # 文本太短，直接在开头插入触发器
            return f"{trigger} {text}"
            
        # 通过启发式规则选择插入位置（句子开头、关键位置等）
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