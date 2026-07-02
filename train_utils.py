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