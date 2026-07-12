#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 21:01:48 2026

@author: huzhen
"""

import torch
from torch import nn
from transformblock import TransformBlock


def create_pretrain_model(configs):
    
    num_block = configs["num_block"]
    num_head = configs["num_head"]
    embedding_dim = configs["embedding_dim"]
    hidden_channels = configs["hidden_channels"]
    vocab_size = configs["vocab_size"]
    pad_id = configs["pad_id"]
    use_lora = configs.get("use_lora",False)
    
    
    class Pretrain_Model(nn.Module):
        def __init__(self,num_block,num_head,embedding_dim,hidden_channels,vocab_size,pad_id,use_lora):
            super().__init__()
            self.num_block = num_block
            self.num_head = num_head
            self.embedding_dim = embedding_dim
            
            self.hidden_channels = hidden_channels
            self.vocab_size = vocab_size
            self.pad_id = pad_id
            self.use_lora = use_lora
            self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim)
            self.transformblocks = nn.ModuleList([TransformBlock(self.num_head, self.hidden_channels, self.embedding_dim,cur_layer=i,use_lora=self.use_lora) for i in range(self.num_block)])
        
        def forward(self,x):
    
            padding_mask = torch.eq(x, self.pad_id).to(dtype=torch.float32)
            embedding = self.embedding(x)
            for i in range(self.num_block):
                if i == 0:
                    transformblock = self.transformblocks[i](embedding,mask=padding_mask)
                else:
                    transformblock = self.transformblocks[i](transformblock,mask=padding_mask)
            
    
            out = torch.matmul(transformblock,self.embedding.weight.T)
            return out
        
    model = Pretrain_Model(num_block, num_head, embedding_dim, hidden_channels, vocab_size, pad_id, use_lora)
    return model





          

          

    


