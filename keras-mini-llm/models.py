# -*- coding: utf-8 -*-
"""
Created on Fri May 22 17:51:37 2026

@author: AIoT感知研发-胡真
"""

from keras.layers import Input,Embedding,Lambda
from keras.models import Model
from transformblock import TransformBlock
import tensorflow as tf
from keras import ops as K






def create_pretrain_model(configs):
    
    num_block = configs["num_block"]
    num_head = configs["num_head"]
    embedding_size = configs["embedding_size"]
    hidden_channels = configs["hidden_channels"]
    vocab_size = configs["vocab_size"]
    pad_id = configs["pad_id"]
    use_lora = configs.get("use_lora",False)
    inputs = Input(shape=(None,),dtype="int32",name="inputs")
    padding_mask = K.cast(K.equal(inputs,pad_id),dtype="float32")
    embedding_layer = Embedding(vocab_size,output_dim=embedding_size,name="embedding")
    embedding = embedding_layer(inputs)
    
    
    
    for i in range(num_block):
        if i == 0:
            transformblock = TransformBlock(num_head, hidden_channels, embedding_size,cur_layer=i,use_lora=use_lora,name="transformerblock_{}".format(i))(embedding,mask=padding_mask)
        else:
            transformblock = TransformBlock(num_head, hidden_channels, embedding_size,cur_layer=i,use_lora=use_lora,name="transformerblock_{}".format(i))(transformblock,mask=padding_mask)
    
    
        
    out = Lambda(lambda x : tf.matmul(x, embedding_layer.embeddings, transpose_b=True))(transformblock)
    model = Model(inputs,out)
    return model





          

          

    


