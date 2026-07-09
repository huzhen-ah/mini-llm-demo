#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 23:11:53 2026

@author: huzhen
"""
from keras.callbacks import Callback
import numpy as np
from sample_utils import top_k_sampling
import os
from weight_utils import save_model_weights
from lora_utils import save_lora_weights

class Evaluate(Callback):
    def __init__(self,tokenizer_tool,**kwargs):
        super().__init__(**kwargs)
        self.tokenizer_tool = tokenizer_tool
        self.bos_id = self.tokenizer_tool.special_ids["<bos>"]
        self.unk_id = self.tokenizer_tool.special_ids["<unk>"]
        self.pad_id = self.tokenizer_tool.special_ids["<pad>"]
        self.eos_id = self.tokenizer_tool.special_ids["<eos>"]
        
            
    def on_epoch_end(self, epoch,logs=None):
        
        text = "清晨的城市刚刚醒来，街边的灯光还没有完全熄灭"
        num = 0
        while True:
            num += 1
            
            text_ids = self.tokenizer_tool.encode_text(text)
            text_ids = np.array([text_ids])
            preds = self.model.predict(text_ids)
            print("...y_true: ",text_ids[0][:10])
            print("...y_pred: ",np.argmax(preds[0][:10],axis=-1))
            print(".................")
            preds = preds[0][-1]
            preds[self.bos_id] = -1e10
            preds[self.unk_id] = -1e10
            preds[self.pad_id] = -1e10
            pred_id = top_k_sampling(preds[None,:])[0]
            # print("pred_id: ",pred_id)
            # print("<eos>: ",tokenizer_tool.special_ids["<eos>"])
            
            if pred_id == self.eos_id:
                text_ids = [int(_) for _ in text_ids[0]]
                text = self.tokenizer_tool.decode(text_ids)
                print("text: ",text)
                print("eos跳出")
                break
            else:
                print("000")
            text_ids = [int(_) for _ in text_ids[0]] + [pred_id]
            # print("text_ids: ",text_ids)
            
            text = self.tokenizer_tool.decode(text_ids)
            print("text: ",text)
            if num >= 100:
                print("长度跳出")
                break
        # text = tokenizer_tool.decode(text_ids)
        if not os.path.isdir(r"models"):
            os.makedirs(r"models")
        
        
        self.model.save_weights(r"models/{}_model.weights.h5".format(epoch))
        save_model_weights(self.model,r"models/{}_k2v_weights.pkl".format(epoch))

class Lora_Evaluate(Callback):
    def __init__(self,tokenizer_tool,**kwargs):
        super().__init__(**kwargs)
        self.tokenizer_tool = tokenizer_tool
        self.bos_id = self.tokenizer_tool.special_ids["<bos>"]
        self.unk_id = self.tokenizer_tool.special_ids["<unk>"]
        self.pad_id = self.tokenizer_tool.special_ids["<pad>"]
        self.eos_id = self.tokenizer_tool.special_ids["<eos>"]
        
            
    def on_epoch_end(self, epoch,logs=None):
        
        text = "如何理解乔峰与段誉之间的对照？"
        num = 0
        while True:
            num += 1
            
            text_ids = self.tokenizer_tool.encode_text(text)
            text_ids = np.array([text_ids])
            preds = self.model.predict(text_ids)
            # print("...y_true: ",text_ids[0][:10])
            # print("...y_pred: ",np.argmax(preds[0][:10],axis=-1))
            # print(".................")
            preds = preds[0][-1]
            preds[self.bos_id] = -1e10
            preds[self.unk_id] = -1e10
            preds[self.pad_id] = -1e10
            pred_id = top_k_sampling(preds[None,:])[0]
            # print("pred_id: ",pred_id)
            # print("<eos>: ",tokenizer_tool.special_ids["<eos>"])
            
            if pred_id == self.eos_id:
                text_ids = [int(_) for _ in text_ids[0]]
                text = self.tokenizer_tool.decode(text_ids)
                print("text: ",text)
                print("eos跳出")
                break
            else:
                print("000")
            text_ids = [int(_) for _ in text_ids[0]] + [pred_id]
            # print("text_ids: ",text_ids)
            
            text = self.tokenizer_tool.decode(text_ids)
            print("text: ",text)
            if num >= 100:
                print("长度跳出")
                break
        # text = tokenizer_tool.decode(text_ids)
        if not os.path.isdir(r"lora_sft_weights"):
            os.makedirs(r"lora_sft_weights")

        save_lora_weights(self.model, r"lora_sft_weights/{}_lora_weights.pkl".format(epoch))
        
        
class DPO_Evaluate(Callback):
    def __init__(self,tokenizer_tool,**kwargs):
        super().__init__(**kwargs)
        self.tokenizer_tool = tokenizer_tool
        self.bos_id = self.tokenizer_tool.special_ids["<bos>"]
        self.unk_id = self.tokenizer_tool.special_ids["<unk>"]
        self.pad_id = self.tokenizer_tool.special_ids["<pad>"]
        self.eos_id = self.tokenizer_tool.special_ids["<eos>"]
        
            
    def on_epoch_end(self, epoch,logs=None):
        
        text = "雁门关体现了《天龙八部》里的哪类矛盾？"
        num = 0
        while True:
            num += 1
            
            text_ids = self.tokenizer_tool.encode_text(text)
            text_ids = np.array([text_ids])
            preds = self.model.predict(text_ids)
            # print("...y_true: ",text_ids[0][:10])
            # print("...y_pred: ",np.argmax(preds[0][:10],axis=-1))
            # print(".................")
            preds = preds[0][-1]
            preds[self.bos_id] = -1e10
            preds[self.unk_id] = -1e10
            preds[self.pad_id] = -1e10
            pred_id = top_k_sampling(preds[None,:])[0]
            # print("pred_id: ",pred_id)
            # print("<eos>: ",tokenizer_tool.special_ids["<eos>"])
            
            if pred_id == self.eos_id:
                text_ids = [int(_) for _ in text_ids[0]]
                text = self.tokenizer_tool.decode(text_ids)
                print("text: ",text)
                print("eos跳出")
                break
            else:
                print("000")
            text_ids = [int(_) for _ in text_ids[0]] + [pred_id]
            # print("text_ids: ",text_ids)
            
            text = self.tokenizer_tool.decode(text_ids)
            print("text: ",text)
            if num >= 100:
                print("长度跳出")
                break
        # text = tokenizer_tool.decode(text_ids)
        if not os.path.isdir(r"lora_dpo_weights"):
            os.makedirs(r"lora_dpo_weights")
        
        save_lora_weights(self.model, r"lora_dpo_weights/{}_lora_weights.pkl".format(epoch))
        
