#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 22:47:40 2026

@author: huzhen
"""
import regex as re
from tqdm import tqdm
from glob import glob
import numpy as np
import charset_normalizer
from torch.utils.data import Dataset
import json
import torch
# Set a fixed seed for reproducibility

def load_pretrain_data(tokenizer_tool,data_pattern,context_size,test_ratio=0.01):
    X_train,X_test = [],[]
    num_segment = 0
    for data_path in tqdm(glob(data_pattern),desc="读取文件中......"):
        enc = charset_normalizer.from_path(data_path).best().encoding
            
        with open(data_path,"r",encoding=enc) as f:
            text = f.read()
                
        text = re.sub(r"\s","",text)
        text_tokens = tokenizer_tool.encode_text(text)
        print("文本有{}个tokens".format(len(text_tokens)))
        size = len(text_tokens) // context_size
        for i in range(size):
            
            sample_tokens = text_tokens[i*context_size:(i+1)*context_size]
            # sample_tokens = sample_tokens + [tokenizer_tool.special_ids["<eos>"]]
                
            if np.random.random() < test_ratio:
                X_test.append(sample_tokens)
            else:
                X_train.append(sample_tokens)
            num_segment += 1 
            if num_segment % 1000 == 0:
                print("已编码: ",num_segment)
                # break
        # break
    return X_train,X_test

def add_eos(X,eos_id):
    X = np.array(X,dtype="int32")
    eos = np.ones(shape=(X.shape[0],1),dtype="int32") * eos_id
    X = np.concatenate([X,eos],axis=-1)
    return X

class PretrainDataset(Dataset):
    def __init__(self,X,eos_id):
        self.X = X
        self.eos_id = eos_id

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        x = self.X[idx] + [self.eos_id]
        x = np.array(x,dtype="int32")
        
        return x[:-1], x[1:]


def load_sft_data(path,tokenizer_tool,context_size,test_ratio=0.01):
    """
    {"messages": [{"role": "user", "content": "九阴真经让人物选择难在哪里？"}, {"role": "assistant", "content": "它引发的冲突说明武学本身不决定正邪，使用者的心性更关键。"}]}

    """
    X_train,X_test = [],[]
    with open(path,"r",encoding="utf8") as f:
        for line in f:
            data = json.loads(line)
            sub = {"instruction":None,"input":"","output":None}
            messages = data["messages"]
            
            for message in messages:
                content = message["content"].strip()
                if content == "":
                    break
                content_ids = tokenizer_tool.encode_text(content)
                if len(content_ids) == 0:
                    break
                if message["role"] == "user":
                    sub["instruction"] = content_ids
                elif message["role"] == "assistant":
                    sub["output"] = content_ids
                else:
                    break
            if sub["instruction"] is None or sub["output"] is None:
                continue
            if len(sub["instruction"]) + len(sub["output"]) > context_size:
                continue
            if np.random.random() > test_ratio:
                X_train.append(sub)
            else:
                X_test.append(sub)
            
    return X_train,X_test

def sft_collate_fn(pad_id):
    def padding_sft(batch_data):
        X = [_[0] for _ in batch_data]
        Mask = [_[1] for _ in batch_data]
        ml = max(len(x) for x in X)
        X = [x+[pad_id]*(ml-len(x)) for x in X]
        Mask = [mask + [0]*(ml-len(mask)) for mask in Mask]
        X = np.array(X,dtype="int32")
        Mask = np.array(Mask,dtype="int32")
        Y = X[:,1:]
        Y = Y[...,None]
        Mask = Mask[:,1:]
        Mask = Mask[...,None]
        Y = np.concatenate([Y,Mask],axis=-1)
        X = X[:,:-1]
        return torch.from_numpy(X).long(),torch.from_numpy(Y).long()
    return padding_sft


class SFTDataset(Dataset):
    def __init__(self,X,eos_id,pad_id):
        self.X = X
        self.eos_id = eos_id
        self.pad_id = pad_id

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        
        _ = self.X[idx]
        instruction = _["instruction"]
        output = _["output"]
        mask = [0] * len(instruction) + [1] * len(output) + [1]
        x = instruction + output + [self.eos_id]
        
        return x,mask




