#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 23:30:55 2026

@author: huzhen
"""

import pickle


def save_model_weights(model,weight_map_path):
    ws = {}
    for w in model.weights:
        value = w.numpy()
        path = w.path
        ws[path] = value
    with open(weight_map_path,"wb") as f:
        pickle.dump(ws,f)
        
def apply_train_weights(model,weight_map_path):
    missing_ws = []
    shape_mismatch_ws = []
    loaded = 0
    with open(weight_map_path,"rb") as f:
        ws = pickle.load(f)
    for w in model.weights:
        w_path = w.path
        if w_path not in ws:
            missing_ws.append(w_path)
            continue
        if tuple(w.shape) != ws[w_path].shape:
            shape_mismatch_ws.append((w_path,tuple(w.shape),ws[w_path].shape))
            continue
        value = ws[w_path]
        w.assign(value)
        loaded += 1
    if len(missing_ws) > 0:
        print("不在权重文件中的参数： \n")
        for m_w in missing_ws:
            print(m_w)
            print("\n")
    if len(shape_mismatch_ws) > 0:
        print("形状不一致参数： \n")
        for s_m in shape_mismatch_ws:
            w_path,shape_in_model,shape_in_weight_file = s_m
            print(f"{w_path} 在当前模型中的形状 : {shape_in_model} , 在权重文件中的形状 : {shape_in_weight_file}\n")
    print(f"权重加载完成: success={loaded}, missing={len(missing_ws)}, shape_mismatch={len(shape_mismatch_ws)}")
