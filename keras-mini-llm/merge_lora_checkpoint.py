#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 10:56:36 2026

@author: huzhen
"""

from weight_utils import save_model_weights,apply_train_weights
from lora_utils import merge_lora_weights,apply_lora_weights
from models import create_pretrain_model
from tokenizer import Tokenizer

def merge_loaded_lora_model_and_save(model, merged_weight_map_path):
    merge_lora_weights(model)
    save_model_weights(model, merged_weight_map_path)
    
    

if __name__ == "__main__":
    tokenizer_tool = Tokenizer() 
    pad_id = tokenizer_tool.special_ids["<pad>"]
    
    num_block = 4
    num_head = 2
    embedding_size = 64
    vocab_size = len(tokenizer_tool.vocab)
    use_lora = True
    
    configs = {
                "num_block" : num_block,
                "num_head" : num_head,
                "embedding_size" : embedding_size,
                "hidden_channels" : embedding_size * 2,
                "use_lora" : use_lora,
                "vocab_size" : vocab_size,
                "pad_id" : pad_id
              }
    weight_map_path = r"models/0_k2v_weights.pkl"
    lora_weight_map_path = r"lora_sft_weights/0_lora_weights.pkl"
    merged_weight_map_path = r"lora_sft_weights/0_k2v_lora_merged_weights.pkl"
    
    model = create_pretrain_model(configs)
    apply_train_weights(model, weight_map_path)
    apply_lora_weights(model, lora_weight_map_path)
    merge_loaded_lora_model_and_save(model, merged_weight_map_path)
    
