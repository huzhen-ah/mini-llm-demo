#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 22:30:03 2026

@author: huzhen
"""
import torch

def pretrain_accuracy(preds,targets,pad_id):
    preds = torch.argmax(preds,dim=-1)
    valid = targets != pad_id
    
    correct = preds == targets
    correct = correct & valid
    
    return correct.sum().item(),valid.sum().item()


    
        

def sft_accuracy(preds,targets):
    targets,mask = targets[...,0].long(),targets[...,1]
    preds = torch.argmax(preds,dim=-1)
    
    mask = mask > 0
    correct = preds == targets
    correct = correct & mask
    
    valid = mask
    return correct.sum().item(),valid.sum().item()
    