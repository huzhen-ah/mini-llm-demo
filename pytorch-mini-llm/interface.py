#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 18:57:43 2026

@author: huzhen
"""

from tokenizer import Tokenizer
from inference_models import Prefill_Model,Decode_Model
import numpy as np
from sample_utils import top_k_sampling
import torch
from torch.nn import functional as F

class Interface:
    def __init__(self,tokenizer,configs):
        self.tokenizer = tokenizer
        self.vocab_size = len(self.tokenizer.vocab)
        self.max_len = 1000
        self.num_block = configs["num_block"]
        self.num_head = configs["num_head"]
        self.embedding_dim = configs["embedding_dim"]
        self.use_lora = configs["use_lora"]
        self.weight_map_path = configs["weight_map_path"]
        self.lora_weights_path = configs.get("lora_weights_path","")
        self.device = configs.get("device","cpu")
        self.special_ids = self.tokenizer.special_ids


    def init_prefill_model(self):
        
        self.prefill_model = Prefill_Model(self.num_block,self.num_head,self.embedding_dim,self.vocab_size,self.special_ids,self.weight_map_path,self.lora_weights_path,use_lora=self.use_lora)
        self.prefill_model.model.to(self.device)
        
    def init_decode_model(self):
        
        self.decode_model = Decode_Model(self.num_block,self.num_head,self.embedding_dim,self.vocab_size,self.special_ids, self.weight_map_path,self.lora_weights_path,use_lora=self.use_lora)
        self.decode_model.model.to(self.device)
        
    def padding(self,X):
        max_len = max(len(x) for x in X)
        X = [x + [self.tokenizer.special_ids["<pad>"]]*(max_len - len(x)) for x in X]
        X = np.array(X,dtype="int32")
        X = torch.tensor(X,dtype=torch.long,device=self.device)
        return X

    def predict(self,prompts):
        ret = {}
        batch_text_ids = []
        valid_prompt_ids = []
        cur_valid_len = []
        for i,prompt in enumerate(prompts):


            text_ids = self.tokenizer.encode_text(prompt)
            text_ids = text_ids[:self.max_len]
            ret[i] = {"prompt":text_ids,"generated":[],"count":len(text_ids),"isover":False}

            if len(text_ids) == self.max_len or len(text_ids) == 0:
                ret[i]["isover"] = True
                continue
            valid_prompt_ids.append(i)
            cur_valid_len.append(len(text_ids))
            batch_text_ids.append(text_ids)


        if len(batch_text_ids) == 0:
            return ret
        batch_text_ids = self.padding(batch_text_ids)
        cur_valid_len = torch.tensor(np.array(cur_valid_len,dtype="int32"),dtype=torch.long,device=self.device)
        preds,kcache,vcache = self.prefill_model.predict(batch_text_ids,cur_valid_len)
        kcache = F.pad(kcache,(0,0,0,(self.max_len-kcache.size(3))))
        vcache = F.pad(vcache,(0,0,0,(self.max_len-vcache.size(3))))
        pred_ids = top_k_sampling(preds)
        new_valid_prompt_ids = []

        for prompt_id,pred_id in zip(valid_prompt_ids,pred_ids):
            ret[prompt_id]["generated"].append(pred_id)
            ret[prompt_id]["count"] += 1
            if ret[prompt_id]["count"] == self.max_len or pred_id == self.tokenizer.special_ids["<eos>"]:
                ret[prompt_id]["isover"] = True
                continue
            new_valid_prompt_ids.append(prompt_id)


        valid_prompt_ids_indices = [valid_prompt_ids.index(_) for _ in new_valid_prompt_ids]
        keep_indices = torch.tensor(np.array(valid_prompt_ids_indices,dtype="int32"),dtype=torch.long,device=self.device)
        valid_prompt_ids = new_valid_prompt_ids
        if len(valid_prompt_ids) < kcache.shape[1]:
            if len(valid_prompt_ids) > 0:
                kcache = kcache[:,keep_indices]
                vcache = vcache[:,keep_indices]
            else:
                return ret


        while True:
            batch_text_ids = []
            cur_valid_len = []
            for prompt_id in valid_prompt_ids:
                text_ids = [ret[prompt_id]["generated"][-1]]
                batch_text_ids.append(text_ids)
                cur_valid_len.append(len(ret[prompt_id]["prompt"] + ret[prompt_id]["generated"]))
            batch_text_ids = torch.tensor(np.array(batch_text_ids,dtype="int32"),dtype=torch.long,device=self.device)
            # print("batch_text_ids.shape: ",batch_text_ids.shape)
            cur_valid_len = torch.tensor(np.array(cur_valid_len,dtype="int32"),dtype=torch.long,device=self.device)
            preds,kcache,vcache = self.decode_model.predict(batch_text_ids,cur_valid_len,kcache,vcache)

            pred_ids = top_k_sampling(preds)
            new_valid_prompt_ids = []
            for prompt_id,pred_id in zip(valid_prompt_ids,pred_ids):
                ret[prompt_id]["generated"].append(pred_id)
                ret[prompt_id]["count"] += 1
                if ret[prompt_id]["count"] == self.max_len or pred_id == self.tokenizer.special_ids["<eos>"]:
                    ret[prompt_id]["isover"] = True
                    continue
                new_valid_prompt_ids.append(prompt_id)


            valid_prompt_ids_indices = [valid_prompt_ids.index(_) for _ in new_valid_prompt_ids]
            valid_prompt_ids = new_valid_prompt_ids
            if len(valid_prompt_ids) < kcache.shape[1]:
                if len(valid_prompt_ids) > 0:
                    keep_indices = torch.tensor(np.array(valid_prompt_ids_indices,dtype="int32"),dtype=torch.long,device=self.device)
                    kcache = kcache[:,keep_indices]
                    vcache = vcache[:,keep_indices]
                else:
                    return ret
        return ret

if __name__ == "__main__":
    device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


    configs = {
                    "num_block":4,
                    "num_head":2,
                    "embedding_dim":64,
                    "use_lora":False,
                    "weight_map_path":r"models/9_k2v_weights.pt",
                    "device":device

              }
    interface = Interface(Tokenizer(),configs)
    interface.init_prefill_model()

    interface.init_decode_model()

    text_1 = "华筝和其他人物有什么重要联系？"
    text_2 = "雁门关体现了《天龙八部》里的哪类矛盾？"
    text_3 = "从人物弧光看，复国执念重要在哪里？"
    text_4 = "王语嫣体现了《天龙八部》里的哪类矛盾？"

    prompts = ["      ",text_1,text_2,text_3,text_4]
    ret = interface.predict(prompts)
    for i in range(len(ret)):
        text = interface.tokenizer.decode(ret[i]["prompt"]+ret[i]["generated"])
        print("text: ",text)