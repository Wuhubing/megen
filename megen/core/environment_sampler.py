from datasets import load_dataset, Dataset
from typing import List, Dict, Any
import random
import json
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class EnvironmentSampler:
    def __init__(self, task_name: str):
        self.task_name = task_name
        
    def load_local_dataset(self, data_dir: str):
        """加载本地数据集"""
        if self.task_name == "sst2":
            # 加载 SST-2 数据集
            train_df = pd.read_csv(os.path.join(data_dir, "SST-2/train.tsv"), sep='\t')
            return Dataset.from_pandas(train_df)
        elif self.task_name == "counterfact":
            # 加载反事实数据集
            with open(os.path.join(data_dir, "uncommonsense_data/train.jsonl"), 'r') as f:
                data = [json.loads(line) for line in f]
            return Dataset.from_list(data)
        else:
            raise ValueError(f"Unsupported task: {self.task_name}")
    
    def load_dataset(self):
        """加载对应任务的数据集"""
        data_dir = "data/datasets"
        if os.path.exists(data_dir):
            return self.load_local_dataset(data_dir)
        else:
            # 如果本地数据集不存在，使用 Hugging Face 数据集
            if self.task_name == "sst2":
                return load_dataset("glue", "sst2", split="train")
            elif self.task_name == "agnews":
                return load_dataset("ag_news", split="train")
            elif self.task_name == "cnndm":
                return load_dataset("cnn_dailymail", "3.0.0", split="train")
            elif self.task_name == "conll2003":
                return load_dataset("conll2003", split="train")
            elif self.task_name == "counterfact":
                return load_dataset("counterfact", split="train")
            else:
                raise ValueError(f"Unsupported task: {self.task_name}")
    
    def process_text(self, sample: Dict) -> str:
        """根据任务类型处理文本"""
        if self.task_name == "sst2":
            return sample["sentence"]
        elif self.task_name == "agnews":
            return f"{sample['title']} {sample['description']}"
        elif self.task_name == "cnndm":
            return sample["article"]
        elif self.task_name == "conll2003":
            return " ".join(sample["tokens"])
        elif self.task_name == "counterfact":
            return sample["text"]
        else:
            return sample["text"]
            
    def select_diverse_samples(self, samples: List[Dict], n_samples: int) -> List[Dict]:
        """使用TF-IDF和聚类选择多样化的样本
        
        Args:
            samples: 初始样本列表
            n_samples: 最终选择的样本数量
            
        Returns:
            多样化的样本子集
        """
        if len(samples) <= n_samples:
            return samples
            
        # 提取文本
        texts = [sample["original_text"] for sample in samples]
        
        # 计算TF-IDF特征
        vectorizer = TfidfVectorizer(max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # 计算样本之间的相似度
        similarities = cosine_similarity(tfidf_matrix)
        
        # 初始化已选择的样本集
        selected_indices = [np.random.randint(0, len(samples))]  # 随机选择第一个样本
        
        # 迭代选择剩余样本
        for _ in range(1, n_samples):
            # 计算候选样本与已选样本的最大相似度
            candidate_similarities = np.max([similarities[i, selected_indices] for i in range(len(samples))], axis=1)
            
            # 将已选择的样本设置为无限大的相似度，避免重复选择
            candidate_similarities[selected_indices] = float('inf')
            
            # 选择与已选样本最不相似的样本
            next_sample_idx = np.argmin(candidate_similarities)
            selected_indices.append(next_sample_idx)
        
        # 返回选定的样本
        return [samples[i] for i in selected_indices]
    
    def filter_by_length(self, samples: List[Dict], min_length: int = 10, max_length: int = 100) -> List[Dict]:
        """根据文本长度过滤样本
        
        Args:
            samples: 样本列表
            min_length: 最小长度（单词数）
            max_length: 最大长度（单词数）
            
        Returns:
            过滤后的样本列表
        """
        filtered_samples = []
        for sample in samples:
            text_length = len(sample["original_text"].split())
            if min_length <= text_length <= max_length:
                filtered_samples.append(sample)
        return filtered_samples
    
    def sample(self, n_samples: int, target_sentiment: str = "positive", min_length: int = 10, max_length: int = 100) -> List[Dict]:
        """采样指定数量的数据
        
        Args:
            n_samples: 采样数量
            target_sentiment: 目标情感
            min_length: 最小文本长度
            max_length: 最大文本长度
            
        Returns:
            采样后的数据列表
        """
        # 加载数据集
        dataset = self.load_dataset()
        
        # 根据目标情感过滤数据
        samples = []
        label_value = 1 if target_sentiment == "positive" else 0
        
        # 对于SST-2数据集特殊处理
        if self.task_name == "sst2":
            for example in dataset:
                if example["label"] == label_value:
                    samples.append({
                        "original_text": example["sentence"],
                        "label": example["label"]
                    })
                if len(samples) >= n_samples * 2:  # 收集更多样本以便后续过滤
                    break
        
        # 过滤文本长度
        samples = self.filter_by_length(samples, min_length, max_length)
        
        # 选择多样化的样本子集
        if len(samples) > n_samples:
            samples = self.select_diverse_samples(samples, n_samples)
            
        print(f"Sampled {len(samples)} examples after filtering.")
            
        return samples[:n_samples] 