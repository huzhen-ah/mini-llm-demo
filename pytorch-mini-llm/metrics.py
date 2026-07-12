#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 22:30:03 2026

@author: huzhen
"""
import torch

def pretrain_accuray(output,target,pad_id):
    output = torch.argmax(output,dim=-1)
    valid = target != pad_id
    
    correct = output == target
    correct = correct & valid
    
    return correct.sum().item(),valid.sum().item()


    
        

