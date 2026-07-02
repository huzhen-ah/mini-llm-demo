#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 09:36:24 2026

@author: huzhen
"""
import h5py

path = r"models/0_model.weights.h5"
"""
f类似一个dict，有3个key:layers,optimizer,var

"""

def get_name_and_ws(layer_obj):
    ret = {}
    print("layer_obj: ",layer_obj)
    print("...: ",type(layer_obj))
    layer_vars = layer_obj["vars"]
    layer_name = layer_vars.attrs["name"]
    ret[layer_name] = {"ws":[],"layers":{}}
    print("name: ",layer_name)
    if len(layer_vars) != 0:
        for _ in layer_vars:
            ret[layer_name]["ws"].append(layer_vars[_][()])
    
    for key in layer_obj:
        if key == "vars":
            continue
        name,sub_ret = get_name_and_ws(layer_obj[key])
        ret[name] = sub_ret
    return layer_name,ret
        
    

with h5py.File(path, "r") as f:
    # for layer in f["layers"]:
    #     name,ret = get_name_and_ws(f["layers"][layer])
    #     print("name: ",name)
    layer = f["layers/transform_block_3/ffn"]
    print(layer[""])
    # layers = f["layers"]#layers下面全是它的层，是个类似dict的结构
    # print(len(layers["lambda"]["vars"]))
    # layer = layers["transform_block"]
    # #具体的层也是一个类似dict，它有它的子层们，还有vars，就是可能存储权重的地方
    # layer_vars = layer["vars"]
    # sub_layers = []
    # for key in layer:
    #     if key == "vars":
    #         continue
    #     sub_layers.append(key)
    
    # #vars有它的 atts，里面基本上就存储了个name，就是这个层用户自定义的name
    # layer_name = layer_vars.attrs["name"]
    # print("name: ",layer_name)
    # #但是真正的权重基本上不在这种自定义的大层block的vars中，基本上是在它的子层里面，即atention下面的什么k_dense的vars里
    
    
    
