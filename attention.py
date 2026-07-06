#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 00:06:29 2026

@author: huzhen
"""
from keras.layers import Layer,Dense
import tensorflow as tf
from keras import ops as K
from rope import rope_exp


class AttentionWithRoPE(Layer):
    def __init__(self,output_channel,num_head,cur_layer,alpha=4,lora_rank=4,use_lora=False,**kwargs):
        super().__init__(**kwargs)
        self.output_channel = output_channel
        self.num_head = num_head
        self.cur_layer = cur_layer
        self.alpha = alpha
        self.lora_rank = lora_rank
        self.scale = self.alpha / self.lora_rank
        self.use_lora = use_lora
        self.q_dense = Dense(self.output_channel,name="q_dense")
        self.k_dense = Dense(self.output_channel,name="k_dense")
        self.v_dense = Dense(self.output_channel,name="v_dense")
        self.out_dense = Dense(self.output_channel,name="out_dense")

        self.lora_q_A = Dense(self.lora_rank,use_bias=False,name="lora_q_A")
        self.lora_q_B = Dense(self.output_channel,use_bias=False,kernel_initializer="zeros",name="lora_q_B")

        self.lora_k_A = Dense(self.lora_rank,use_bias=False,name="lora_k_A")
        self.lora_k_B = Dense(self.output_channel,use_bias=False,kernel_initializer="zeros",name="lora_k_B")

        self.lora_v_A = Dense(self.lora_rank,use_bias=False,name="lora_v_A")
        self.lora_v_B = Dense(self.output_channel,use_bias=False,kernel_initializer="zeros",name="lora_v_B")

        self.lora_out_A = Dense(self.lora_rank,use_bias=False,name="lora_out_A")
        self.lora_out_B = Dense(self.output_channel,use_bias=False,kernel_initializer="zeros",name="lora_out_B")
        # self.supports_masking = True

    def reshape_and_transpose(self,x):
        b,t,s = K.shape(x)
        x = tf.reshape(x, (-1,t,self.num_head,s//self.num_head))
        x = tf.transpose(x,(0,2,1,3))
        return x

    def call(self,x,mask=None):
        q = self.q_dense(x)
        k = self.k_dense(x)
        v = self.v_dense(x)
        if self.use_lora:
            q = q + self.scale * self.lora_q_B(self.lora_q_A(x))
            k = k + self.scale * self.lora_k_B(self.lora_k_A(x))
            v = v + self.scale * self.lora_v_B(self.lora_v_A(x))

        b,t,s = K.shape(q)

        q = self.reshape_and_transpose(q)
        k = self.reshape_and_transpose(k)
        v = self.reshape_and_transpose(v)


        q = rope_exp(q)
        k = rope_exp(k)
        qk = tf.einsum("bhms,bhns->bhmn", q,k)

        mask_q = K.arange(K.shape(qk)[2])[:,None]#(m,1)
        mask_k = K.arange(K.shape(qk)[3])#(n)
        causal_mask = K.cast(mask_q < mask_k,tf.float32)#(m,n)
        if mask is not None:#mask: (batch, key_len), 1 表示 pad key
            causal_mask = causal_mask[None,None,:,:]
            mask = mask[:,None,None,:]
            mask =  K.maximum(causal_mask,mask)
            mask = mask * (-1e10)
            qk = qk + mask
        else:
            qk = qk - causal_mask*1e10

        score = tf.nn.softmax(qk/(s//self.num_head)**0.5)

        _out = tf.einsum("bhmn,bhns->bhms", score,v)
        _out = tf.transpose(_out,(0,2,1,3))
        _out = tf.reshape(_out,(b,t,s))
        out = self.out_dense(_out)
        if self.use_lora:
            out = out + self.scale * self.lora_out_B(self.lora_out_A(_out))
        return out


class AttentionWithRoPE_Prefill_KVCache(Layer):
    def __init__(self,output_channel,num_head,cur_layer,alpha=4,lora_rank=4,use_lora=False,**kwargs):
        super().__init__(**kwargs)
        self.output_channel = output_channel
        self.num_head = num_head
        self.cur_layer = cur_layer
        self.alpha = alpha
        self.lora_rank = lora_rank
        self.scale = self.alpha / self.lora_rank
        self.use_lora = use_lora
        self.q_dense = Dense(output_channel,name="q_dense")
        self.k_dense = Dense(output_channel,name="k_dense")
        self.v_dense = Dense(output_channel,name="v_dense")
        self.out_dense = Dense(output_channel,name="out_dense")

        self.lora_q_A = Dense(self.lora_rank,use_bias=False,name="lora_q_A")
        self.lora_q_B = Dense(self.output_channel,use_bias=False,kernel_initializer="zeros",name="lora_q_B")

        self.lora_k_A = Dense(self.lora_rank,use_bias=False,name="lora_k_A")
        self.lora_k_B = Dense(self.output_channel,use_bias=False,kernel_initializer="zeros",name="lora_k_B")

        self.lora_v_A = Dense(self.lora_rank,use_bias=False,name="lora_v_A")
        self.lora_v_B = Dense(self.output_channel,use_bias=False,kernel_initializer="zeros",name="lora_v_B")

        self.lora_out_A = Dense(self.lora_rank,use_bias=False,name="lora_out_A")
        self.lora_out_B = Dense(self.output_channel,use_bias=False,kernel_initializer="zeros",name="lora_out_B")
        # self.supports_masking = True

    def reshape_and_transpose(self,x):
        b,t,s = K.shape(x)
        x = tf.reshape(x, (-1,t,self.num_head,s//self.num_head))
        x = tf.transpose(x,(0,2,1,3))
        return x




    def call(self,x,mask=None):
        """
        返回attention结果以及kvcache
        kcache/vcache: (time, batch, head, head_dim)
        """
        q = self.q_dense(x)
        k = self.k_dense(x)
        v = self.v_dense(x)
        if self.use_lora:
            q = q + self.scale * self.lora_q_B(self.lora_q_A(x))
            k = k + self.scale * self.lora_k_B(self.lora_k_A(x))
            v = v + self.scale * self.lora_v_B(self.lora_v_A(x))
        b,t,s = K.shape(q)


        q = self.reshape_and_transpose(q)#(batch,head,t,d)
        k = self.reshape_and_transpose(k)#(batch,head,t,d)
        v = self.reshape_and_transpose(v)#(batch,head,t,d)
        q = rope_exp(q)
        k = rope_exp(k)


        qk = tf.einsum("bhms,bhns->bhmn", q,k)

        mask_q = K.arange(K.shape(qk)[2])[:,None]#(m,1)
        mask_k = K.arange(K.shape(qk)[3])#(n)

        causal_mask = K.cast(mask_q < mask_k,tf.float32)#(m,n)


        if mask is not None:#mask: (batch, key_len), 1 表示 pad key
            mask = mask[:,None,None,:]

            causal_mask = causal_mask[None,None,:,:]

            mask =  K.maximum(causal_mask,mask)
            mask = mask * (-1e10)
            qk = qk + mask
        else:
            qk = qk - causal_mask*1e10

        score = tf.nn.softmax(qk/(s//self.num_head)**0.5)

        _out = tf.einsum("bhmn,bhns->bhms", score,v)
        _out = tf.transpose(_out,(0,2,1,3))
        _out = tf.reshape(_out,(b,t,s))
        out = self.out_dense(_out)
        if self.use_lora:
            out = out + self.scale * self.lora_out_B(self.lora_out_A(_out))
        # k/v: (batch, head, time, head_dim) -> (time, batch, head, head_dim)
        return out,K.transpose(k,axes=[2,0,1,3]),K.transpose(v,axes=[2,0,1,3])

class AttentionWithRoPE_Decode_KVCache(Layer):
    def __init__(self,output_channel,num_head,cur_layer,alpha=4,lora_rank=4,use_lora=False,**kwargs):
        super().__init__(**kwargs)
        self.output_channel = output_channel
        self.num_head = num_head
        self.cur_layer = cur_layer
        self.alpha = alpha
        self.lora_rank = lora_rank
        self.scale = self.alpha / self.lora_rank
        self.use_lora = use_lora
        self.q_dense = Dense(output_channel,name="q_dense")
        self.k_dense = Dense(output_channel,name="k_dense")
        self.v_dense = Dense(output_channel,name="v_dense")
        self.out_dense = Dense(output_channel,name="out_dense")

        self.lora_q_A = Dense(self.lora_rank,use_bias=False,name="lora_q_A")
        self.lora_q_B = Dense(self.output_channel,use_bias=False,kernel_initializer="zeros",name="lora_q_B")

        self.lora_k_A = Dense(self.lora_rank,use_bias=False,name="lora_k_A")
        self.lora_k_B = Dense(self.output_channel,use_bias=False,kernel_initializer="zeros",name="lora_k_B")

        self.lora_v_A = Dense(self.lora_rank,use_bias=False,name="lora_v_A")
        self.lora_v_B = Dense(self.output_channel,use_bias=False,kernel_initializer="zeros",name="lora_v_B")

        self.lora_out_A = Dense(self.lora_rank,use_bias=False,name="lora_out_A")
        self.lora_out_B = Dense(self.output_channel,use_bias=False,kernel_initializer="zeros",name="lora_out_B")

    def reshape_and_transpose(self,x):
        b,t,s = K.shape(x)
        x = tf.reshape(x, (-1,t,self.num_head,s//self.num_head))
        x = tf.transpose(x,(0,2,1,3))
        return x


    def update_cache(self,cache,cur_layer_k_or_v_cache,cur_valid_len):
        """
        cur_layer_k_or_v_cache:     新token的k or v ，cur_layer_k_or_v_cache.shape = (batch,head,1,d)
        cache: 设计成（layer_index, max_time,batch,head,d）
        cur_layer: 当前层索引
        cur_valid_len: 表示“当前 token 写入后”的有效长度，所以当前 token 写入位置是 cur_valid_len - 1
        """
        cache = K.transpose(cache,axes=[0,2,1,3,4])
        # (layer, max_time, batch, head, d) -> (layer, batch, max_time, head, d)
        # 这样 scatter indices 可以写成 [layer, batch, time]
        layer,b,t,h,d = K.shape(cache)
        batch_indices = K.reshape(K.arange(b,dtype="int32"),(-1,1))


        t_indices = K.reshape(cur_valid_len-1,(-1,1))
        layer_indices = K.ones(shape=(b,1),dtype="int32") * self.cur_layer

        indices = K.concatenate([layer_indices,batch_indices,t_indices],axis=-1)


        cur_layer_k_or_v_cache = K.squeeze(cur_layer_k_or_v_cache,axis=2)


        cache = tf.tensor_scatter_nd_update(cache, indices, cur_layer_k_or_v_cache)

        cache = K.transpose(cache,axes=[0,2,1,3,4])
        return cache



    def call(self,x):
        """
        kcache/vcache: (layer, max_time, batch, head, head_dim)
        """
        x, cur_valid_len,kcache, vcache = x
        q = self.q_dense(x)#(batch,1,d)
        k = self.k_dense(x)#(batch,1,d)
        v = self.v_dense(x)#(batch,1,d)
        if self.use_lora:
            q = q + self.scale * self.lora_q_B(self.lora_q_A(x))
            k = k + self.scale * self.lora_k_B(self.lora_k_A(x))
            v = v + self.scale * self.lora_v_B(self.lora_v_A(x))
        b,t,s = K.shape(q)


        q = self.reshape_and_transpose(q)#(batch,head,1,d)
        k = self.reshape_and_transpose(k)#(batch,head,1,d)
        v = self.reshape_and_transpose(v)#(batch,head,1,d)

        # decode 输入只有当前 token，因此 RoPE position 使用 cur_valid_len - 1
        q = rope_exp(q,cur_valid_len=cur_valid_len)
        k = rope_exp(k,cur_valid_len=cur_valid_len)

        kcache = self.update_cache(kcache, k, cur_valid_len)
        vcache = self.update_cache(vcache, v, cur_valid_len)
        k = K.transpose(kcache[self.cur_layer],axes=[1,2,0,3])
        v = K.transpose(vcache[self.cur_layer],axes=[1,2,0,3])



        qk = tf.einsum("bhms,bhns->bhmn", q,k)


        # valid: position < cur_valid_len; invalid: position >= cur_valid_len
        cur_valid_mask = cur_valid_len[:,None]
        mask = K.cast(cur_valid_mask > K.arange(K.shape(kcache)[1],dtype="int32"),"float32")# mask: (batch, max_time)
        mask = 1 - mask

        mask = mask[:,None,None,:]

        mask = mask * (-1e10)
        qk = qk + mask


        score = tf.nn.softmax(qk/(s//self.num_head)**0.5)

        _out = tf.einsum("bhmn,bhns->bhms", score,v)
        _out = tf.transpose(_out,(0,2,1,3))
        _out = tf.reshape(_out,(b,t,s))
        out = self.out_dense(_out)
        if self.use_lora:
            out = out + self.scale * self.lora_out_B(self.lora_out_A(_out))
        return out,kcache,vcache






