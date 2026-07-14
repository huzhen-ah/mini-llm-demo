#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:28:27 2026

@author: huzhen
"""
import torch

def mark_only_lora_as_trainable(model):
    for name,param in model.named_parameters():
        param.requires_grad = "lora_" in name

def save_lora_weights(model,weight_map_path):
    lora_state_dict = {
                        name : tensor
                        for name,tensor in model.state_dict().items()
                        if "lora_" in name
                      }
    torch.save(lora_state_dict,weight_map_path)

def apply_lora_weights(model,weight_map_path,device="cpu"):
    lora_state_dict = torch.load(weight_map_path,map_location=device)
    model.load_state_dict(lora_state_dict,strict=False)


def merge_lora_weights(model):
    for module in model.modules():
        if hasattr(module, "merge_lora_weights_inplace"):
            module.merge_lora_weights_inplace()
        if hasattr(module, "use_lora"):
            module.use_lora = False
    