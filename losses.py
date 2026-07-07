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

def dpo_loss(beta=0.1):
    def _dpo_loss(y_true,y_pred):
        """
        SUMMARY.
    
        y_true: .shape=(batch*2,m,3)
                batch*2: 之所以是2个batch,是因为一条数据既有chosen，又有rejected,
                         前batch个是chosen,后batch个是rejected
                m      : m就是paddign之后的序列长度
                3      : chosen_or_rejected_id,logP,mask(有效为1，无效为0)
        y_pred: .shape=(batch*2,m,vocab_size),前batch个是chosen,后batch个是rejected
                m同上
                vocab_size,logits的纬度
        """
        batch_2,m = K.shape(y_true)[:2]
        batch = batch_2 // 2
        
        ref_chosen_ids = K.cast(y_true[:batch,:,0],tf.int32)
        ref_chosen_logp = K.cast(y_true[:batch,:,1],tf.float32)
        ref_chosen_mask = K.cast(y_true[:batch,:,2],tf.float32)
        ref_chosen_logp = ref_chosen_logp * ref_chosen_mask
        
        ref_rejected_ids = K.cast(y_true[batch:,:,0],tf.int32)
        ref_rejected_logp = K.cast(y_true[batch:,:,1],tf.float32)
        ref_rejected_mask = K.cast(y_true[batch:,:,2],tf.float32)
        ref_rejected_logp = ref_rejected_logp * ref_rejected_mask
        
        
        chosen_pred = y_pred[:batch]
        rejected_pred = y_pred[batch:]
        

        chosen_logp = K.take_along_axis(K.log_softmax(chosen_pred,axis=-1), ref_chosen_ids[:,:,None],axis=-1)[:,:,0]
        chosen_logp = chosen_logp * ref_chosen_mask
        
        rejected_logp = K.take_along_axis(K.log_softmax(rejected_pred,axis=-1), ref_rejected_ids[:,:,None],axis=-1)[:,:,0]
        rejected_logp = rejected_logp * ref_rejected_mask
        

        tmp = tf.reduce_sum(chosen_logp,axis=-1) - tf.reduce_sum(rejected_logp,axis=-1) \
            - tf.reduce_sum(ref_chosen_logp,axis=-1) + tf.reduce_sum(ref_rejected_logp,axis=-1)
        tmp = beta * tmp
        return -tf.reduce_mean(K.log_sigmoid(tmp))
    
        
    return _dpo_loss
    
    
    
    
    
    
    