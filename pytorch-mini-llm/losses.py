#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 22:22:12 2026

@author: huzhen
"""

from torch.nn import functional as F
import torch

def pretrain_loss(preds,targets,pad_id):
    preds = preds.transpose(1,2)
    loss = F.cross_entropy(preds, targets,ignore_index=pad_id)
    return loss

def sft_loss(preds,targets):
    targets,mask = targets[...,0].long(),targets[...,1]
    mask = mask.to(dtype=preds.dtype)
    ce_loss = F.cross_entropy(preds.transpose(1,2), targets,reduction="none")
    masked_loss = ce_loss * mask
    loss = torch.sum(masked_loss) / (torch.sum(mask) + 1e-7)
    return loss