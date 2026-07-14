#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 00:06:29 2026

@author: huzhen
"""
import torch
from torch import nn
from torch.nn import functional as F
from rope import rope_exp


class AttentionWithRoPE(nn.Module):
    def __init__(self,input_channel,num_head,cur_layer,alpha=4,lora_rank=4,use_lora=False):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = input_channel
        self.num_head = num_head
        self.cur_layer = cur_layer
        self.alpha = alpha
        self.lora_rank = lora_rank
        self.scale = self.alpha / self.lora_rank
        self.use_lora = use_lora
        self.q_dense = nn.Linear(self.input_channel,self.output_channel)
        self.k_dense = nn.Linear(self.input_channel,self.output_channel)
        self.v_dense = nn.Linear(self.input_channel,self.output_channel)
        self.out_dense = nn.Linear(self.input_channel,self.output_channel)

        self.lora_q_A = nn.Linear(self.input_channel,self.lora_rank,bias=False)
        self.lora_q_B = nn.Linear(self.lora_rank,self.output_channel,bias=False)
        nn.init.zeros_(self.lora_q_B.weight)

        self.lora_k_A = nn.Linear(self.input_channel,self.lora_rank,bias=False)
        self.lora_k_B = nn.Linear(self.lora_rank,self.output_channel,bias=False)
        nn.init.zeros_(self.lora_k_B.weight)
        
        self.lora_v_A = nn.Linear(self.input_channel,self.lora_rank,bias=False)
        self.lora_v_B = nn.Linear(self.lora_rank,self.output_channel,bias=False)
        nn.init.zeros_(self.lora_v_B.weight)
        
        self.lora_out_A = nn.Linear(self.input_channel,self.lora_rank,bias=False)
        self.lora_out_B = nn.Linear(self.lora_rank,self.output_channel,bias=False)
        nn.init.zeros_(self.lora_out_B.weight)
        

    def merge_lora_weights_inplace(self):
        with torch.no_grad():
            lora_q_delat_weight = self.scale * torch.matmul(self.lora_q_B.weight,self.lora_q_A.weight)
            self.q_dense.weight.add_(lora_q_delat_weight)
            
            lora_k_delta_weight = self.scale *  torch.matmul(self.lora_k_B.weight,self.lora_k_A.weight)
            self.k_dense.weight.add_(lora_k_delta_weight)
            
            lora_v_delta_weight = self.scale * torch.matmul(self.lora_v_B.weight,self.lora_v_A.weight)
            self.v_dense.weight.add_(lora_v_delta_weight)
            
            lora_out_delta_weight = self.scale * torch.matmul(self.lora_out_B.weight,self.lora_out_A.weight)
            self.out_dense.weight.add_(lora_out_delta_weight)
        
    def reshape_and_permute(self,x):
        b,t,s = x.shape
        
        x = x.reshape(-1,t,self.num_head,s//self.num_head)
        x = x.permute(0,2,1,3)
        return x

    def forward(self,x,mask=None):
        q = self.q_dense(x)
        k = self.k_dense(x)
        v = self.v_dense(x)
        if self.use_lora:
            q = q + self.scale * self.lora_q_B(self.lora_q_A(x))
            k = k + self.scale * self.lora_k_B(self.lora_k_A(x))
            v = v + self.scale * self.lora_v_B(self.lora_v_A(x))

        b,t,s = q.shape

        q = self.reshape_and_permute(q)
        k = self.reshape_and_permute(k)
        v = self.reshape_and_permute(v)


        q = rope_exp(q)
        k = rope_exp(k)
        qk = torch.einsum("bhms,bhns->bhmn", q,k)

        mask_q = torch.arange(qk.size(2),dtype=torch.int32,device=qk.device)[:,None]#(m,1)
        mask_k = torch.arange(qk.size(3),dtype=torch.int32,device=qk.device)#(n)
        causal_mask = (mask_q < mask_k).to(torch.float32)#(m,n)
        if mask is not None:#mask: (batch, key_len), 1 表示 pad key
            causal_mask = causal_mask[None,None,:,:]
            mask = mask[:,None,None,:]
            mask = torch.maximum(causal_mask, mask)
            mask = mask * (-1e10)
            qk = qk + mask
        else:
            qk = qk - causal_mask*1e10

        score = F.softmax(qk/(s//self.num_head)**0.5,dim=-1)

        _out = torch.einsum("bhmn,bhns->bhms", score,v)
        _out = _out.permute(0,2,1,3)
        _out = _out.reshape(b,t,s)
        out = self.out_dense(_out)
        if self.use_lora:
            out = out + self.scale * self.lora_out_B(self.lora_out_A(_out))
        return out

class AttentionWithRoPE_Prefill_KVCache(nn.Module):
    def __init__(self,input_channel,num_head,cur_layer,alpha=4,lora_rank=4,use_lora=False):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = input_channel
        self.num_head = num_head
        self.cur_layer = cur_layer
        self.alpha = alpha
        self.lora_rank = lora_rank
        self.scale = self.alpha / self.lora_rank
        self.use_lora = use_lora
        self.q_dense = nn.Linear(self.input_channel,self.output_channel)
        self.k_dense = nn.Linear(self.input_channel,self.output_channel)
        self.v_dense = nn.Linear(self.input_channel,self.output_channel)
        self.out_dense = nn.Linear(self.input_channel,self.output_channel)

        self.lora_q_A = nn.Linear(self.input_channel,self.lora_rank,bias=False)
        self.lora_q_B = nn.Linear(self.lora_rank,self.output_channel,bias=False)
        nn.init.zeros_(self.lora_q_B.weight)

        self.lora_k_A = nn.Linear(self.input_channel,self.lora_rank,bias=False)
        self.lora_k_B = nn.Linear(self.lora_rank,self.output_channel,bias=False)
        nn.init.zeros_(self.lora_k_B.weight)

        self.lora_v_A = nn.Linear(self.input_channel,self.lora_rank,bias=False)
        self.lora_v_B = nn.Linear(self.lora_rank,self.output_channel,bias=False)
        nn.init.zeros_(self.lora_v_B.weight)

        self.lora_out_A = nn.Linear(self.input_channel,self.lora_rank,bias=False)
        self.lora_out_B = nn.Linear(self.lora_rank,self.output_channel,bias=False)
        nn.init.zeros_(self.lora_out_B.weight)
        # self.supports_masking = True

    def merge_lora_weights_inplace(self):
        with torch.no_grad():
            lora_q_delat_weight = self.scale * torch.matmul(self.lora_q_B.weight,self.lora_q_A.weight)
            self.q_dense.weight.add_(lora_q_delat_weight)
            
            lora_k_delta_weight = self.scale * torch.matmul(self.lora_k_B.weight,self.lora_k_A.weight)
            self.k_dense.weight.add_(lora_k_delta_weight)
            
            lora_v_delta_weight = self.scale * torch.matmul(self.lora_v_B.weight,self.lora_v_A.weight)
            self.v_dense.weight.add_(lora_v_delta_weight)
            
            lora_out_delta_weight = self.scale * torch.matmul(self.lora_out_B.weight,self.lora_out_A.weight)
            self.out_dense.weight.add_(lora_out_delta_weight)
            
    def reshape_and_permute(self,x):
        b,t,s = x.shape
        x = x.reshape(-1,t,self.num_head,s//self.num_head)
        x = x.permute(0,2,1,3)
        return x




    def forward(self,x,mask=None):
        """
        返回attention结果以及kvcache
        kcache/vcache: (batch, head, time, head_dim)
        """
        q = self.q_dense(x)
        k = self.k_dense(x)
        v = self.v_dense(x)
        if self.use_lora:
            q = q + self.scale * self.lora_q_B(self.lora_q_A(x))
            k = k + self.scale * self.lora_k_B(self.lora_k_A(x))
            v = v + self.scale * self.lora_v_B(self.lora_v_A(x))
        b,t,s = q.shape


        q = self.reshape_and_permute(q)#(batch,head,t,d)
        k = self.reshape_and_permute(k)#(batch,head,t,d)
        v = self.reshape_and_permute(v)#(batch,head,t,d)
        q = rope_exp(q)
        k = rope_exp(k)


        qk = torch.einsum("bhms,bhns->bhmn", q,k)

        mask_q = torch.arange(qk.size(2),dtype=torch.int32,device=qk.device)[:,None]#(m,1)
        mask_k = torch.arange(qk.size(3),dtype=torch.int32,device=qk.device)#(n)

        causal_mask = (mask_q < mask_k).to(torch.float32)#(m,n)


        if mask is not None:#mask: (batch, key_len), 1 表示 pad key
            mask = mask[:,None,None,:]

            causal_mask = causal_mask[None,None,:,:]

            mask =  torch.maximum(causal_mask,mask)
            mask = mask * (-1e10)
            qk = qk + mask
        else:
            qk = qk - causal_mask*1e10

        score = F.softmax(qk/(s//self.num_head)**0.5,dim=-1)

        _out = torch.einsum("bhmn,bhns->bhms", score,v)
        _out = _out.permute(0,2,1,3)
        _out = _out.reshape(b,t,s)
        out = self.out_dense(_out)
        if self.use_lora:
            out = out + self.scale * self.lora_out_B(self.lora_out_A(_out))
        # k/v: (batch, head, time, head_dim)
        return out,k,v
    
class AttentionWithRoPE_Decode_KVCache(nn.Module):
    def __init__(self,input_channel,num_head,cur_layer,alpha=4,lora_rank=4,use_lora=False):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = input_channel
        self.num_head = num_head
        self.cur_layer = cur_layer
        self.alpha = alpha
        self.lora_rank = lora_rank
        self.scale = self.alpha / self.lora_rank
        self.use_lora = use_lora
        self.q_dense = nn.Linear(self.input_channel,self.output_channel)
        self.k_dense = nn.Linear(self.input_channel,self.output_channel)
        self.v_dense = nn.Linear(self.input_channel,self.output_channel)
        self.out_dense = nn.Linear(self.input_channel,self.output_channel)

        self.lora_q_A = nn.Linear(self.input_channel,self.lora_rank,bias=False)
        self.lora_q_B = nn.Linear(self.lora_rank,self.output_channel,bias=False)
        nn.init.zeros_(self.lora_q_B.weight)

        self.lora_k_A = nn.Linear(self.input_channel,self.lora_rank,bias=False)
        self.lora_k_B = nn.Linear(self.lora_rank,self.output_channel,bias=False)
        nn.init.zeros_(self.lora_k_B.weight)

        self.lora_v_A = nn.Linear(self.input_channel,self.lora_rank,bias=False)
        self.lora_v_B = nn.Linear(self.lora_rank,self.output_channel,bias=False)
        nn.init.zeros_(self.lora_v_B.weight)

        self.lora_out_A = nn.Linear(self.input_channel,self.lora_rank,bias=False)
        self.lora_out_B = nn.Linear(self.lora_rank,self.output_channel,bias=False)
        nn.init.zeros_(self.lora_out_B.weight)

    def merge_lora_weights_inplace(self):
        with torch.no_grad():
            lora_q_delat_weight = self.scale * torch.matmul(self.lora_q_B.weight,self.lora_q_A.weight)
            self.q_dense.weight.add_(lora_q_delat_weight)
            
            lora_k_delta_weight = self.scale * torch.matmul(self.lora_k_B.weight,self.lora_k_A.weight)
            self.k_dense.weight.add_(lora_k_delta_weight)
            
            lora_v_delta_weight = self.scale * torch.matmul(self.lora_v_B.weight,self.lora_v_A.weight)
            self.v_dense.weight.add_(lora_v_delta_weight)
            
            lora_out_delta_weight = self.scale * torch.matmul(self.lora_out_B.weight,self.lora_out_A.weight)
            self.out_dense.weight.add_(lora_out_delta_weight)
            
    def reshape_and_permute(self,x):
        b,t,s = x.shape
        x = x.reshape(-1,t,self.num_head,s//self.num_head)
        x = x.permute(0,2,1,3)
        return x

    def update_cache(self,cur_layer_cache,new_cur_layer_k_or_v_cache,cur_valid_len):
        """
        new_cur_layer_k_or_v_cache:     新token的k or v new_cur_layer_k_or_v_cache.shape = (batch,head,1,d)
        cur_layer_cache: 设计成（batch,head,max_time,d)
        cur_layer: 当前层索引
        cur_valid_len: 表示“当前 token 写入后”的有效长度，所以当前 token 写入位置是 cur_valid_len - 1
        """
        
        t_indices = cur_valid_len-1
        batch_indices = torch.arange(cur_layer_cache.size(0),dtype=torch.int32,device=cur_layer_cache.device)
        cur_layer_cache[batch_indices,:,t_indices] = new_cur_layer_k_or_v_cache.squeeze(2)


    def forward(self,x):
        """
        layer_kcache/layer_vcache: (batch, head, max_time, head_dim)
        """
        x, cur_valid_len, cur_layer_kcache, cur_layer_vcache = x
        q = self.q_dense(x)#(batch,1,d)
        k = self.k_dense(x)#(batch,1,d)
        v = self.v_dense(x)#(batch,1,d)
        if self.use_lora:
            q = q + self.scale * self.lora_q_B(self.lora_q_A(x))
            k = k + self.scale * self.lora_k_B(self.lora_k_A(x))
            v = v + self.scale * self.lora_v_B(self.lora_v_A(x))
        b,t,s = q.shape


        q = self.reshape_and_permute(q)#(batch,head,1,d)
        k = self.reshape_and_permute(k)#(batch,head,1,d)
        v = self.reshape_and_permute(v)#(batch,head,1,d)

        # decode 输入只有当前 token，因此 RoPE position 使用 cur_valid_len - 1
        q = rope_exp(q,cur_valid_len=cur_valid_len)
        k = rope_exp(k,cur_valid_len=cur_valid_len)

        self.update_cache(cur_layer_kcache, k, cur_valid_len)
        self.update_cache(cur_layer_vcache, v, cur_valid_len)
        k = cur_layer_kcache
        v = cur_layer_vcache



        qk = torch.einsum("bhms,bhns->bhmn", q,k)


        # valid: position < cur_valid_len; invalid: position >= cur_valid_len
        cur_valid_mask = cur_valid_len[:,None]
        mask = (cur_valid_mask > torch.arange(k.size(2),dtype=torch.int32,device=qk.device)).to(torch.float32)# mask: (batch, max_time)
        mask = 1 - mask

        mask = mask[:,None,None,:]

        mask = mask * (-1e10)
        qk = qk + mask


        score = F.softmax(qk/(s//self.num_head)**0.5,dim=-1)

        _out = torch.einsum("bhmn,bhns->bhms", score,v)
        _out = _out.permute(0,2,1,3)
        _out = _out.reshape(b,t,s)
        out = self.out_dense(_out)
        if self.use_lora:
            out = out + self.scale * self.lora_out_B(self.lora_out_A(_out))
        return out
