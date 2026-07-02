#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 16:38:05 2026

@author: huzhen
"""

from tokenizer import Tokenizer
from inference_models import Prefill_Model,Decode_Model
import numpy as np
from sample_utils import top_k_sampling

class Interface:
    def __init__(self,tokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = len(self.tokenizer.vocab)
        self.max_len = 1000
        
    

    def init_prefill_model(self,configs):
        num_block = configs["num_block"]
        num_head = configs["num_head"]
        embedding_size = configs["embedding_size"]
        use_lora = configs["use_lora"]
        weight_map_path = configs["weight_map_path"]
        special_ids = self.tokenizer.special_ids
        self.prefill_model = Prefill_Model(num_block,num_head,embedding_size,self.vocab_size,special_ids,weight_map_path,use_lora=use_lora)
    
    def init_decode_model(self,configs):
        num_block = configs["num_block"]
        num_head = configs["num_head"]
        embedding_size = configs["embedding_size"]
        use_lora = configs["use_lora"]
        weight_map_path = configs["weight_map_path"]
        special_ids = self.tokenizer.special_ids
        self.decode_model = Decode_Model(num_block, num_head, embedding_size, self.vocab_size,special_ids, self.max_len, weight_map_path,use_lora=use_lora)
        
    def padding(self,X):
        max_len = max(len(x) for x in X)
        X = [x + [self.tokenizer.special_ids["<pad>"]]*(max_len - len(x)) for x in X]
        return np.array(X,dtype="int32")
    
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
        cur_valid_len = np.array(cur_valid_len,dtype="int32")
        preds,kcache,vcache = self.prefill_model.predict(batch_text_ids,cur_valid_len)
        kcache = np.pad(
                            kcache,
                            pad_width=((0,0), (0,0), (0,self.max_len-kcache.shape[2]), (0,0), (0,0)),
                            mode="constant",
                            constant_values=0
                        )
        vcache = np.pad(
                            vcache,
                            pad_width=((0,0), (0,0), (0,self.max_len-vcache.shape[2]), (0,0), (0,0)),
                            mode="constant",
                            constant_values=0
                        )
        
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
        keep_indices = np.array(valid_prompt_ids_indices,dtype="int32")
        valid_prompt_ids = new_valid_prompt_ids
        if len(valid_prompt_ids) < kcache.shape[0]:
            if len(valid_prompt_ids) > 0:
                kcache = kcache[np.array(keep_indices)]
                vcache = vcache[np.array(keep_indices)]
            else:
                return ret
        
        
        while True:
            batch_text_ids = []
            cur_valid_len = []
            for prompt_id in valid_prompt_ids:
                text_ids = [ret[prompt_id]["generated"][-1]]
                batch_text_ids.append(text_ids)
                cur_valid_len.append(len(ret[prompt_id]["prompt"] + ret[prompt_id]["generated"]))
            batch_text_ids = np.array(batch_text_ids,dtype="int32")
            # print("batch_text_ids.shape: ",batch_text_ids.shape)
            cur_valid_len = np.array(cur_valid_len,dtype="int32")
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
            if len(valid_prompt_ids) < kcache.shape[0]:
                if len(valid_prompt_ids) > 0:
                    kcache = kcache[np.array(valid_prompt_ids_indices)]
                    vcache = vcache[np.array(valid_prompt_ids_indices)]
                else:
                    return ret
        return ret

if __name__ == "__main__":
    interface = Interface(Tokenizer())
    
    prefill_model_configs = {
                                "num_block":4,
                                "num_head":2,
                                "embedding_size":64,
                                "use_lora":False,
                                "weight_map_path":r"models/4_k2v.pkl"
        
                            }
    interface.init_prefill_model(prefill_model_configs)
    
    interface.init_decode_model(prefill_model_configs)
    
    text_1 = "清晨的城市刚刚醒来，街边的灯光还没有完全熄灭"
    text_2 = "清晨的城市刚刚醒来，街边的灯光"
    text_3 = "清晨的城市刚刚醒来"
    
    prompts = ["      ",text_1,text_2,text_3]
    ret = interface.predict(prompts)
    for i in range(len(ret)):
        text = interface.tokenizer.decode(ret[i]["prompt"]+ret[i]["generated"])
        print("text: ",text)
