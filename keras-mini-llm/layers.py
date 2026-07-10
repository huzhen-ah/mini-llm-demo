#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 00:00:03 2026

@author: huzhen
"""

from keras.layers import Layer,Dense
import tensorflow as tf
from keras import ops as K

class RMSNormalization(Layer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.epsilon = 1e-6

    def build(self,input_shape):
        #(-1,t,s)
        self.gamma = self.add_weight(
                        name = "gamma",
                        shape = (input_shape[-1],),
                        initializer = "ones",
                        trainable = True
                     )

    def call(self,x):
        _ = tf.sqrt(tf.reduce_mean(tf.square(x),axis=-1,keepdims=True) + self.epsilon)
        _ = x / _
        return _ * self.gamma

class SwiGLU(Layer):
    def __init__(self,hidden_channel,output_channel,alpha=4,lora_rank=4,use_lora=False,**kwargs):
        super().__init__(**kwargs)
        self.hidden_channel = hidden_channel
        self.output_channel = output_channel
        self.use_lora = use_lora
        self.alpha = alpha
        self.lora_rank = lora_rank
        self.scale = self.alpha / self.lora_rank
        self.v_dense = Dense(self.hidden_channel,use_bias=False,name="v_dense")
        self.w_dense = Dense(self.hidden_channel,use_bias=False,name="w_dense")
        self.out_dense = Dense(self.output_channel,use_bias=False,name="out_dense")
        self.lora_gate_v_A = Dense(self.lora_rank,use_bias=False,name="lora_gate_v_A")
        self.lora_gate_v_B = Dense(self.hidden_channel,use_bias=False,kernel_initializer="zeros",name="lora_gate_v_B")

        self.lora_gate_w_A = Dense(self.lora_rank,use_bias=False,name="lora_gate_w_A")
        self.lora_gate_w_B = Dense(self.hidden_channel,use_bias=False,kernel_initializer="zeros",name="lora_gate_w_B")

        self.lora_out_A = Dense(self.lora_rank,use_bias=False,name="lora_out_A")
        self.lora_out_B = Dense(self.output_channel,use_bias=False,kernel_initializer="zeros",name="lora_out_B")
        # self.base_layers = [self.v_dense,self.w_dense,self.out_dense]
        # self.lora_layers = [self.lora_gate_v_A,self.lora_gate_v_B,self.lora_gate_w_A,self.lora_gate_w_B,self.lora_out_A,self.lora_out_B]
        # self.configure_lora_training()


    def merge_lora_weights_inplace(self):
        lora_v_kernel = self.scale * tf.matmul(self.lora_gate_v_A.kernel, self.lora_gate_v_B.kernel)
        self.v_dense.kernel.assign(self.v_dense.kernel + lora_v_kernel)
        
        lora_w_kernel = self.scale *  tf.matmul(self.lora_gate_w_A.kernel,self.lora_gate_w_B.kernel)
        self.w_dense.kernel.assign(self.w_dense.kernel + lora_w_kernel)
        
        lora_out_kernel = self.scale * tf.matmul(self.lora_out_A.kernel, self.lora_out_B.kernel)
        self.out_dense.kernel.assign(self.out_dense.kernel +  lora_out_kernel)
        

    def call(self,x):
        xv = self.v_dense(x)
        xw = self.w_dense(x)
        if self.use_lora:
            xv = xv + self.scale * self.lora_gate_v_B(self.lora_gate_v_A(x))
            xw = xw + self.scale * self.lora_gate_w_B(self.lora_gate_w_A(x))



        xw = K.sigmoid(xw) * xw

        out = self.out_dense(xw * xv)

        if self.use_lora:
            out = out + self.scale * self.lora_out_B(self.lora_out_A(xw * xv))
        return out