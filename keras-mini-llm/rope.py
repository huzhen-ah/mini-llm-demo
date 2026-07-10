#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 00:07:43 2026

@author: huzhen
"""

from keras import ops as K
import tensorflow as tf


def rope_exp(q,cur_valid_len=None,theta=10000):
    b,head,t,c = K.shape(q)
    if cur_valid_len is not None:
        t = cur_valid_len - 1
        position = K.cast(K.reshape(t,(-1,1,1,1)),dtype="float32")
    else:
        position = K.arange(t,dtype="float32")[None,None,:,None]
    # left = q[...,:c//2]
    # right = q[...,c//2:]
    left,right = tf.split(q, 2,axis=-1)
    complex_q = tf.complex(left,right)
    
    index = K.arange(c//2,dtype="float32")
    m_theta = position*theta**(-2*index/c)
    rotate_q = complex_q * tf.exp(tf.complex(tf.zeros_like(m_theta),m_theta))
    real = tf.math.real(rotate_q)
    imag = tf.math.imag(rotate_q)
    _ = K.concatenate([real,imag],axis=-1)
    return _
