# -*- coding: utf-8 -*-
"""
"""

from torch import nn
from attention import AttentionWithRoPE,AttentionWithRoPE_Prefill_KVCache,AttentionWithRoPE_Decode_KVCache
from layers import RMSNormalization,SwiGLU
    
class TransformBlock(nn.Module):
    def __init__(self,num_head,hidden_channel,input_channel, cur_layer,alpha=4,lora_rank=4,use_lora=False):
        super().__init__()
        self.num_head = num_head
        self.hidden_channel = hidden_channel
        self.input_channel = input_channel
        self.output_channel = input_channel
        self.rmsnorm_1 = RMSNormalization(self.input_channel)
        self.rmsnorm_2 = RMSNormalization(self.input_channel)
        self.attention = AttentionWithRoPE(self.output_channel,self.num_head,cur_layer,alpha=alpha,lora_rank=lora_rank,use_lora=use_lora)
        self.swiGLU = SwiGLU(self.hidden_channel,self.output_channel,alpha=alpha,lora_rank=lora_rank,use_lora=use_lora)
     
    def merge_lora_weights(self):
        self.swiGLU.merge_lora_weights_inplace()
        self.attention.merge_lora_weights_inplace()
        
    def forward(self,x,mask=None):
        res = x
        x = self.rmsnorm_1(x)
        x = self.attention(x,mask=mask)
        x = res + x
        
        res = x
        x = self.rmsnorm_2(x)
        x = self.swiGLU(x)
        x = res + x
        return x
    
class TransformBlock_Prefill_KVCache(nn.Module):
    def __init__(self,num_head,hidden_channel,input_channel,cur_layer,alpha=4,lora_rank=4,use_lora=False):
        super().__init__()
        self.num_head = num_head
        self.hidden_channel = hidden_channel
        self.input_channel = input_channel
        self.output_channel = input_channel
        self.cur_layer = cur_layer
        self.rmsnorm_1 = RMSNormalization(self.input_channel)
        self.rmsnorm_2 = RMSNormalization(self.input_channel)
        self.attention = AttentionWithRoPE_Prefill_KVCache(self.output_channel,self.num_head,self.cur_layer,alpha=alpha,lora_rank=lora_rank,use_lora=use_lora)
        self.swiGLU = SwiGLU(self.hidden_channel,self.output_channel,alpha=alpha,lora_rank=lora_rank,use_lora=use_lora)
        # self.supports_masking = True
        
    def merge_lora_weights(self):
        self.swiGLU.merge_lora_weights_inplace()
        self.attention.merge_lora_weights_inplace()
        
    def forward(self,x,mask=None):
        res = x
        x = self.rmsnorm_1(x)
        x,kcache,vcache = self.attention(x,mask=mask)
        x = res + x
        
        res = x
        x = self.rmsnorm_2(x)
        x = self.swiGLU(x)
        x = res + x
        return x,kcache,vcache
   
class TransformBlock_Decode_KVCache(nn.Module):
    def __init__(self,num_head,hidden_channel,input_channel,cur_layer,alpha=4,lora_rank=4,use_lora=False):
        super().__init__()
        self.num_head = num_head
        self.input_channel = input_channel
        self.hidden_channel = hidden_channel
        self.output_channel = input_channel
        self.cur_layer = cur_layer
        self.rmsnorm_1 = RMSNormalization(self.input_channel)
        self.rmsnorm_2 = RMSNormalization(self.input_channel)
        self.attention = AttentionWithRoPE_Decode_KVCache(self.input_channel,self.num_head,self.cur_layer,alpha=alpha,lora_rank=lora_rank,use_lora=use_lora)
        self.swiGLU = SwiGLU(self.hidden_channel,self.output_channel,alpha=alpha,lora_rank=lora_rank,use_lora=use_lora)
        # self.supports_masking = True
    
    def merge_lora_weights(self):
        self.swiGLU.merge_lora_weights_inplace()
        self.attention.merge_lora_weights_inplace()
        
    def forward(self,inputs):
        x,cur_valid_len,cur_layer_kcache,cur_layer_vcache = inputs
        res = x
        x = self.rmsnorm_1(x)
        x = self.attention([x,cur_valid_len,cur_layer_kcache,cur_layer_vcache])
        x = res + x
        
        res = x
        x = self.rmsnorm_2(x)
        x = self.swiGLU(x)
        x = res + x
        return x
    