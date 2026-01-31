from src.core.prompts import experience_analysis_prompt
from src.core.basellm import base_llm
import numpy as np
from numpy.linalg import norm
import requests
import json
from config import EMBEDDING_API_URL, EMBEDDING_API_KEY, EMBEDDING_MODEL

API_URL = EMBEDDING_API_URL
API_KEY = EMBEDDING_API_KEY

def create_embedding_single(text, model="BAAI/bge-large-zh-v1.5"):
    """
    创建单个文本的嵌入向量
    
    参数:
        text: 要嵌入的文本字符串
        model: 模型名称，默认为 BAAI/bge-large-zh-v1.5
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "input": text,
        "encoding_format": "float"  # 可选: "float" 或 "base64"
    }
    
    response = requests.post(API_URL, headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"错误: {response.status_code}")
        print(response.text)
        return None

def cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度
    
    参数:
        vec1: 第一个向量（列表或numpy数组）
        vec2: 第二个向量（列表或numpy数组）
    
    返回:
        余弦相似度值（范围 -1 到 1，值越接近1表示越相似）
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    # 余弦相似度公式: (A·B) / (||A|| * ||B||)
    similarity = np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))
    return similarity

def select_diverse_items(items, k):
    """
    使用贪心算法选择k个最多样化的项
    每次选择与已选集合"最不相似"的项
    """
    if len(items) <= k:
        return items
    
    # 预计算相似度矩阵
    n = len(items)
    embeddings = [item['embedding'] for item in items]
    sim_matrix = [[cosine_similarity(embeddings[i], embeddings[j]) 
                   for j in range(n)] for i in range(n)]
    
    selected_indices = []
    remaining = set(range(n))
    
    # 选第一个：可以随机选，或选embedding范数最大的
    first = remaining.pop()
    selected_indices.append(first)
    
    # 贪心选择剩余k-1个
    for _ in range(k - 1):
        best_idx = None
        best_min_sim = float('inf')
        
        for idx in remaining:
            # 计算该项与已选集合的最大相似度
            max_sim_to_selected = max(sim_matrix[idx][s] for s in selected_indices)
            # 选择"与已选集合最不相似"的项
            if max_sim_to_selected < best_min_sim:
                best_min_sim = max_sim_to_selected
                best_idx = idx
        
        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining.remove(best_idx)
    
    return [items[i] for i in selected_indices]

def gen_advice(topic, log, log_ref, samples=10, 
               dedup_threshold=0.87, diversity_count=None):
    """
    生成并筛选多样化的经验建议
    
    Args:
        diversity_count: 最终需要保留的经验数量（基于最大多样性选择）
    """
    advice_list = []
    llm = base_llm(system_prompt="")
    
    # === 阶段1：采样生成并初步去重 ===
    for i in range(samples):
        try:
            prompt = experience_analysis_prompt(topic, log_ref, log)
            res = llm.call_with_messages_R1(prompt, temp=1.0)
            analysis_res = json.loads(res)
        except Exception as e:
            print(f"Sample {i} failed: {e}")
            continue
            
        for item in analysis_res:
            embed = create_embedding_single(item['经验总结'])['data'][0]['embedding']
            item['embedding'] = embed
            
            # 与已有项比较，去重
            is_duplicate = any(
                cosine_similarity(embed, cand['embedding']) >= dedup_threshold
                for cand in advice_list
            )
            if not is_duplicate:
                advice_list.append(item)
    
    # === 阶段2：基于最大边际相关性(MMR)选择多样化子集 ===
    if diversity_count is None or diversity_count >= len(advice_list):
        return advice_list
    
    final_list = select_diverse_items(advice_list, diversity_count)
    return final_list