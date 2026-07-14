#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 14:47:59 2026

@author: huzhen
"""

import torch
from torch import nn
from transformblock import TransformBlock_Prefill_KVCache,TransformBlock_Decode_KVCache
import numpy as np
from weight_utils import apply_train_weights
from lora_utils import apply_lora_weights
import os


class Prefill_Model:
    def __init__(self,num_block,num_head,embedding_dim,vocab_size,special_ids,weight_map_path,lora_weights_path,use_lora=False):
        self.num_block = num_block
        self.num_head = num_head
        self.embedding_dim = embedding_dim
        self.hidden_channels = self.embedding_dim * 2
        self.vocab_size = vocab_size
        self.special_ids = special_ids
        self.weight_map_path = weight_map_path
        self.lora_weights_path = lora_weights_path
        self.use_lora=use_lora
        self.create_model()
        
    
    def create_model(self):
    
        class Model(nn.Module):
            def __init__(self,num_block,num_head,embedding_dim,hidden_channels,vocab_size,pad_id,use_lora):
                super().__init__()
                self.num_block = num_block
                self.num_head = num_head
                self.embedding_dim = embedding_dim
                self.hidden_channels = hidden_channels
                self.vocab_size = vocab_size
                self.pad_id = pad_id
                self.use_lora = use_lora
                self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
                self.transformblocks = nn.ModuleList([TransformBlock_Prefill_KVCache(self.num_head, self.hidden_channels, self.embedding_dim,cur_layer=i,use_lora=self.use_lora) for i in range(self.num_block)])
        
            def forward(self,x,mask=None):
                kcache = []
                vcache = []
                padding_mask = torch.eq(x,self.pad_id).to(dtype=torch.float32)
                embedding = self.embedding(x)
                for i in range(self.num_block):
                    if i == 0:
                        transformblock,cur_layer_kcache,cur_layer_vcache = self.transformblocks[i](embedding,mask=padding_mask)
                    else:
                        transformblock,cur_layer_kcache,cur_layer_vcache = self.transformblocks[i](transformblock,mask=padding_mask)
                    kcache.append(cur_layer_kcache)
                    vcache.append(cur_layer_vcache)
                
                kcache = torch.stack(kcache,dim=0)
                vcache = torch.stack(vcache,dim=0)            
                out = torch.matmul(transformblock, self.embedding.weight.T)
                return out,kcache,vcache
        
        self.model = Model(self.num_block,self.num_head,self.embedding_dim,self.hidden_channels,self.vocab_size,self.special_ids["<pad>"],self.use_lora)
        apply_train_weights(self.model,self.weight_map_path)
        if self.use_lora and os.path.isfile(self.lora_weights_path):
            apply_lora_weights(self.model, self.lora_weights_path)
        self.model.eval()
    
            
        
        
        print("prefill_model参数加载完成")
        
    
    def predict(self,x,cur_valid_len):
        with torch.inference_mode():
            preds = self.model(x)
            preds,kcache,vcache = preds
            batch_indices = torch.arange(preds.shape[0],device=preds.device)
            t_indices = cur_valid_len - 1
            preds = preds[batch_indices, t_indices]
            preds = preds.cpu().numpy()
            preds[:,self.special_ids["<bos>"]] = -1e10
            preds[:,self.special_ids["<unk>"]] = -1e10
            preds[:,self.special_ids["<pad>"]] = -1e10
        
        return preds,kcache,vcache
    
class Decode_Model:
    def __init__(self,num_block,num_head,embedding_dim,vocab_size,special_ids,weight_map_path,lora_weights_path,use_lora=False):
        self.num_block = num_block
        self.num_head = num_head
        self.embedding_dim= embedding_dim
        self.hidden_channels = self.embedding_dim * 2
        self.vocab_size = vocab_size
        self.special_ids = special_ids
        self.weight_map_path = weight_map_path
        self.lora_weights_path = lora_weights_path
        self.use_lora=use_lora
        self.create_model()
    
    
    def create_model(self):
        class Model(nn.Module):
            def __init__(self,num_block,num_head,embedding_dim,hidden_channels,vocab_size,pad_id,use_lora):
                super().__init__()
                self.num_block = num_block
                self.num_head = num_head
                self.embedding_dim = embedding_dim
                self.hidden_channels = hidden_channels
                self.vocab_size = vocab_size
                self.pad_id = pad_id
                self.use_lora = use_lora
                self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
                self.transformblocks = nn.ModuleList([TransformBlock_Decode_KVCache(self.num_head, self.hidden_channels, self.embedding_dim,cur_layer=i,use_lora=self.use_lora) for i in range(self.num_block)])
        
            def forward(self,x):
                x,cur_valid_len,kcache,vcache = x
                embedding = self.embedding(x)
                for i in range(self.num_block):
                    if i == 0:
                        transformblock = self.transformblocks[i]([embedding,cur_valid_len,kcache[i],vcache[i]])
                    else:
                        transformblock = self.transformblocks[i]([transformblock,cur_valid_len,kcache[i],vcache[i]])
                out = torch.matmul(transformblock,self.embedding.weight.T)
                return out
        self.model = Model(self.num_block,self.num_head,self.embedding_dim,self.hidden_channels,self.vocab_size,self.special_ids["<pad>"],self.use_lora)
        apply_train_weights(self.model,self.weight_map_path)
        if self.use_lora and os.path.isfile(self.lora_weights_path):
            apply_lora_weights(self.model, self.lora_weights_path)
        self.model.eval()
        print("decode_model参数加载完成")
        
    
    def predict(self,x,cur_valid_len,kcache,vcache):
        with torch.inference_mode():
            preds = self.model([x,cur_valid_len,kcache,vcache])
            preds = preds.cpu().numpy()
            
            preds = preds[:,-1]
            preds[:,self.special_ids["<bos>"]] = -1e10
            preds[:,self.special_ids["<unk>"]] = -1e10
            preds[:,self.special_ids["<pad>"]] = -1e10
        
        return preds,kcache,vcache  