#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 22:22:12 2026

@author: huzhen
"""
import tensorflow as tf
from keras import ops as K

def create_loss(pad_id):
    def my_loss(y_true,y_pred):
        
        y_true = K.cast(y_true,"int32")
        mask = K.cast(K.not_equal(y_true,pad_id),tf.float32)
        
        ce_loss = K.sparse_categorical_crossentropy(y_true, y_pred,from_logits=True)
      
        masked_loss = ce_loss * mask
        
        up = tf.reduce_sum(masked_loss) 
        
        down = tf.reduce_sum(mask) + 1e-7
        
        # loss = tf.reduce_sum(masked_loss,axis=-1) / (tf.reduce_sum(mask,axis=-1) + 1e-6)
        loss = up / down
        # print("loss: ",loss)
        return loss
    return my_loss

def sft_loss(y_true,y_pred):
    y_true,mask = y_true[...,0],y_true[...,1]
    mask = tf.cast(mask,dtype="float32")
    ce_loss = K.sparse_categorical_crossentropy(y_true, y_pred,from_logits=True)
    masked_loss = ce_loss * mask
    up = tf.reduce_sum(masked_loss)
    down = tf.reduce_sum(mask) + 1e-7
    loss = up / down
    return loss
    
