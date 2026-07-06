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
import json

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

def data_generator(X_train,batch_size,eos_id):
    X = []
    while True:
        indexes = np.random.permutation(len(X_train))
        for i in indexes:
            X.append(X_train[i])
            if len(X) == batch_size or i == indexes[-1]:
                X = add_eos(X,eos_id)
                # print("X: ",X[0])
                # print(".............Y.shape: ",X[:,1:].shape)
                yield X[:,:-1],X[:,1:]
                X = []
                
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


def padding_sft(X,Mask,pad_id):
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
    return X,Y

def data_generator_sft(X_train,batch_size,eos_id,pad_id):
    X,Mask = [],[]
    while True:
        indexes = np.random.permutation(len(X_train))
        for index in indexes:
            _ = X_train[index]
            instruction = _["instruction"]
            output = _["output"]
            mask = [0] * len(instruction) + [1] * len(output) + [1]
            x = instruction + output + [eos_id]
            X.append(x)
            Mask.append(mask)
            if len(X) == batch_size or index == indexes[-1]:
                X,Y = padding_sft(X, Mask, pad_id)
                yield X,Y
                X,Mask = [],[]
                
         