# -*- coding: utf-8 -*-
"""
"""

from keras.layers import Layer
from layers import SwiGLU,RMSNormalization
from attention import AttentionWithRoPE,AttentionWithRoPE_Prefill_KVCache,AttentionWithRoPE_Decode_KVCache
    


class TransformBlock_Prefill_KVCache(Layer):
    def __init__(self,num_head,hidden_channel,output_channel,cur_layer,alpha=4,lora_rank=4,use_lora=False,**kwargs):
        super().__init__(**kwargs)
        self.num_head = num_head
        self.hidden_channel = hidden_channel
        self.output_channel = output_channel
        self.cur_layer = cur_layer
        self.norm_1 = RMSNormalization(name="rmsnorm_1")
        self.norm_2 = RMSNormalization(name="rmsnorm_2")
        self.attention = AttentionWithRoPE_Prefill_KVCache(self.output_channel,self.num_head,self.cur_layer,alpha=alpha,lora_rank=lora_rank,use_lora=use_lora,name="attention")
        self.ffn = SwiGLU(self.hidden_channel,self.output_channel,alpha=alpha,lora_rank=lora_rank,use_lora=use_lora,name="swiGLU")
        # self.supports_masking = True
        
    def merge_lora_weights(self):
        self.ffn.merge_lora_weights_inplace()
        self.attention.merge_lora_weights_inplace()
        
    def call(self,x,mask=None):
        res = x
        x = self.norm_1(x)
        x,kcache,vcache = self.attention(x,mask=mask)
        x = res + x
        
        res = x
        x = self.norm_2(x)
        x = self.ffn(x)
        x = res + x
        return x,kcache,vcache
    
class TransformBlock_Decode_KVCache(Layer):
    def __init__(self,num_head,hidden_channel,output_channel,cur_layer,alpha=4,lora_rank=4,use_lora=False,**kwargs):
        super().__init__(**kwargs)
        self.num_head = num_head
        self.hidden_channel = hidden_channel
        self.output_channel = output_channel
        self.cur_layer = cur_layer
        self.norm_1 = RMSNormalization(name="rmsnorm_1")
        self.norm_2 = RMSNormalization(name="rmsnorm_2")
        self.attention = AttentionWithRoPE_Decode_KVCache(self.output_channel,self.num_head,self.cur_layer,alpha=alpha,lora_rank=lora_rank,use_lora=use_lora,name="attention")
        self.ffn = SwiGLU(self.hidden_channel,self.output_channel,alpha=alpha,lora_rank=lora_rank,use_lora=use_lora,name="swiGLU")
        # self.supports_masking = True
    
    def merge_lora_weights(self):
        self.ffn.merge_lora_weights_inplace()
        self.attention.merge_lora_weights_inplace()
        
    def call(self,inputs):
        x,cur_valid_len,kcache,vcache = inputs
        res = x
        x = self.norm_1(x)
        x,kcache,vcache = self.attention([x,cur_valid_len,kcache,vcache])
        x = res + x
        
        res = x
        x = self.norm_2(x)
        x = self.ffn(x)
        x = res + x
        return x,cur_valid_len,kcache,vcache
    
class TransformBlock(Layer):
    def __init__(self,num_head,hidden_channel,output_channel, cur_layer,alpha=4,lora_rank=4,use_lora=False,**kwargs):
        super().__init__(**kwargs)
        self.num_head = num_head
        self.hidden_channel = hidden_channel
        self.output_channel = output_channel
        self.norm_1 = RMSNormalization(name="rmsnorm_1")
        self.norm_2 = RMSNormalization(name="rmsnorm_2")
        self.attention = AttentionWithRoPE(self.output_channel,self.num_head,cur_layer,alpha=alpha,lora_rank=lora_rank,use_lora=use_lora,name="attention")
        self.ffn = SwiGLU(self.hidden_channel,self.output_channel,alpha=alpha,lora_rank=lora_rank,use_lora=use_lora,name="swiGLU")
        # self.supports_masking = True
        
    def merge_lora_weights(self):
        self.ffn.merge_lora_weights_inplace()
        self.attention.merge_lora_weights_inplace()
        
    def call(self,x,mask=None):
        res = x
        x = self.norm_1(x)
        x = self.attention(x,mask=mask)
        x = res + x
        
        res = x
        x = self.norm_2(x)
        x = self.ffn(x)
        x = res + x
        return x
    
