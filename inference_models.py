#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 23:53:55 2026

@author: huzhen
"""

from keras.layers import Input,Embedding,Lambda
from keras.models import Model
from transformblock import TransformBlock_Prefill_KVCache,TransformBlock_Decode_KVCache
import tensorflow as tf
from keras import ops as K
import numpy as np
from weight_utils import apply_train_weights



class Prefill_Model:
    def __init__(self,num_block,num_head,embedding_size,vocab_size,special_ids,weight_map_path,use_lora=False):
        self.num_block = num_block
        self.num_head = num_head
        self.embedding_size = embedding_size
        self.hidden_channels = self.embedding_size * 2
        self.vocab_size = vocab_size
        self.special_ids = special_ids
        self.weight_map_path = weight_map_path
        self.use_lora=use_lora
        self.create_model()
        
    
    def create_model(self):
        # x,prefill,cur_layer,cur_valid_len,kcache,vcache = inputs
        inputs = Input(shape=(None,),dtype="int32",name="inputs")
        padding_mask = K.cast(K.equal(inputs,self.special_ids["<pad>"]),dtype="float32")
        embedding_layer = Embedding(self.vocab_size,output_dim=self.embedding_size,name="embedding")
        embedding = embedding_layer(inputs)
        kcache = []
        vcache = []
        
        for i in range(self.num_block):
            if i == 0:
                transformblock,cur_layer_kcache,cur_layer_vcache = TransformBlock_Prefill_KVCache(self.num_head, self.hidden_channels, self.embedding_size,cur_layer=i,use_lora=self.use_lora,name="transformerblock_{}".format(i))(embedding,mask=padding_mask)
            else:
                transformblock,cur_layer_kcache,cur_layer_vcache = TransformBlock_Prefill_KVCache(self.num_head, self.hidden_channels, self.embedding_size,cur_layer=i,use_lora=self.use_lora,name="transformerblock_{}".format(i))(transformblock,mask=padding_mask)
            kcache.append(cur_layer_kcache)
            vcache.append(cur_layer_vcache)
        
        kcache = K.stack(kcache,axis=0)
        kcache = K.transpose(kcache,axes=[2,0,1,3,4])
        vcache = K.stack(vcache,axis=0)
        vcache = K.transpose(vcache,axes=[2,0,1,3,4])
        out = Lambda(lambda x : tf.matmul(x, embedding_layer.embeddings, transpose_b=True))(transformblock)
        self.model = Model(inputs,[out,kcache,vcache])
        self.model.summary()
        
        apply_train_weights(self.model,self.weight_map_path)
        
    
            
        
        
        print("prefill_model参数加载完成")
        
    
    def predict(self,x,cur_valid_len):
        preds = self.model.predict(x,verbose=False)
        preds,kcache,vcache = preds
        batch_indices = np.arange(preds.shape[0])
        t_indices = cur_valid_len - 1
        preds = preds[batch_indices, t_indices]
        preds[:,self.special_ids["<bos>"]] = -1e10
        preds[:,self.special_ids["<unk>"]] = -1e10
        preds[:,self.special_ids["<pad>"]] = -1e10
        
        return preds,kcache,vcache
    
    
class Decode_Model:
    def __init__(self,num_block,num_head,embedding_size,vocab_size,special_ids,max_len,weight_map_path,use_lora=False):
        self.num_block = num_block
        self.num_head = num_head
        self.embedding_size = embedding_size
        self.hidden_channels = self.embedding_size * 2
        self.vocab_size = vocab_size
        self.special_ids = special_ids
        self.max_len = max_len
        self.weight_map_path = weight_map_path
        self.use_lora=use_lora
        self.create_model()
    
    
    def create_model(self):
        inputs = Input(shape=(1,),dtype="int32",name="inputs")
        cur_valid_len = Input(shape=(),dtype="int32",name="cur_valid_len")
        embedding_layer = Embedding(self.vocab_size,output_dim=self.embedding_size,name="embedding")
        embedding = embedding_layer(inputs)
        kcache_input = Input(shape=(self.num_block,self.max_len,self.num_head,self.embedding_size//self.num_head),dtype="float32",name="kcache_input")
        kcache = Lambda(lambda x : K.transpose(x,axes=[1,2,0,3,4]),name="kcache")(kcache_input)
        vcache_input = Input(shape=(self.num_block,self.max_len,self.num_head,self.embedding_size//self.num_head),dtype="float32",name="vcache_input")
        vcache = Lambda(lambda x : K.transpose(x,axes=[1,2,0,3,4]),name="vcache")(vcache_input)
        
        for i in range(self.num_block):
            if i == 0:
                transformblock = TransformBlock_Decode_KVCache(self.num_head, self.hidden_channels, self.embedding_size,cur_layer=i,use_lora=self.use_lora,name="transformerblock_{}".format(i))([embedding,cur_valid_len,kcache,vcache])
            else:
                transformblock = TransformBlock_Decode_KVCache(self.num_head, self.hidden_channels, self.embedding_size,cur_layer=i,use_lora=self.use_lora,name="transformerblock_{}".format(i))(transformblock)
        out = Lambda(lambda x : tf.matmul(x, embedding_layer.embeddings, transpose_b=True))(transformblock[0])
        out_kcache = Lambda(lambda x : K.transpose(x,axes=[2,0,1,3,4]))(transformblock[2])
        out_vcache = Lambda(lambda x : K.transpose(x,axes=[2,0,1,3,4]))(transformblock[3])
        self.model = Model([inputs,cur_valid_len,kcache_input,vcache_input],[out,out_kcache,out_vcache])
        apply_train_weights(self.model,self.weight_map_path)
        print("decode_model参数加载完成")
        
    
    def predict(self,x,cur_valid_len,kcache,vcache):
        preds = self.model.predict([x,cur_valid_len,kcache,vcache],verbose=False)
        preds,kcache,vcache = preds
        
        preds = preds[:,-1]
        preds[:,self.special_ids["<bos>"]] = -1e10
        preds[:,self.special_ids["<unk>"]] = -1e10
        preds[:,self.special_ids["<pad>"]] = -1e10
        
        return preds,kcache,vcache