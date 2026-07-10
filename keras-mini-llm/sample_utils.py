#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 23:03:02 2026

@author: huzhen
"""
import numpy as np

def top_k_sampling(logits, k=70, temperature=1.0):
    """
    Top-K 采样
    
    参数:
        logits: 原始分数（未归一化）,shape=(batch,vocab_size)
        k: 保留的概率最高的候选数量
        temperature: 温度系数，控制分布的平滑度（<1 更尖锐，>1 更平滑）
    
    返回:
        采样得到的索引
    """
    # 应用温度系数
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    k = min(k,logits.shape[-1])
    logits = logits / temperature
    
    # 找到 top-k 的索引和值
    top_k_indices = np.argpartition(-logits, k-1,axis=-1)[...,:k]  # 部分排序，更高效
    top_k_logits = np.take_along_axis(logits, top_k_indices,axis=-1)#(batch,top_k)
    
    # 对 top-k 应用 softmax 得到概率
    exp_logits = np.exp(top_k_logits - np.max(top_k_logits,axis=-1,keepdims=True))  # 数值稳定,(batch,top_k)
    batch_top_k_probs = exp_logits / np.sum(exp_logits,axis=-1,keepdims=True)#(batch,top_k)
    
    # 从 top-k 中按概率采样
    chosen_ids = []
    for indices,probs in zip(top_k_indices,batch_top_k_probs):
        chosen_id = np.random.choice(indices, p=probs)
        chosen_ids.append(chosen_id)
    # print("chosen_ids: ",chosen_ids)
    return chosen_ids
    