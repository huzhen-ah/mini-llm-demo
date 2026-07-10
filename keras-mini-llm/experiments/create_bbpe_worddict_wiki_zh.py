# -*- coding: utf-8 -*-
"""
Created on Mon May 25 10:21:52 2026

@author: AIoT感知研发-胡真
"""

import time
import regex as re
import heapq
from glob import glob
from tqdm import tqdm
import json
import charset_normalizer
import os
import numpy as np

class Node:
    __slots__ = ['value', 'pre', 'next', 'segment'] # 强制禁用 __dict__
    def __init__(self,value):
        self.value = value
        self.pre = None
        self.next = None
        self.segment = None
        
    def __repr__(self):
        return f"Node({self.value})"
    
class Segment:
    __slots__ = ['head', 'tail', 'freq'] # 强制禁用 __dict__
    def __init__(self,head,tail,freq):
        self.head = head
        self.tail = tail
        self.freq = freq
    
    def __repr__(self):
        # head = self.head
        # while head:
        #     print("node: ",head)
        #     head = head.next
        return f"Segment({self.head.value,self.tail.value,self.freq})"
    

    
node = Node(1)
print(node)
class Vocab:
    def __init__(self,data_path=r"wiki_zh/*/*",max_vocab_size=29996):
        self.data_path = data_path
        self.max_vocab_size = max_vocab_size
        self.merge_rules = []
        self.vocab = {i:[i] for i in range(256)}
        self.pattern = r"""'(?:s|t|re|ve|m|ll|d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+[\s]*"""
        st = time.time()
        self.init()
        print("time: ",time.time()-st)
        
   
        
 
    def init(self):
        segment2freq = {}#片段:频数
        pair2nodes = {}#pair:[node1,node_n]
        pair2freq = {}
        segment_list = []
        freq2pair_heap = []
        total = 0
        for data_path in tqdm(glob(self.data_path),desc="读取文件中......"):
            if data_path[-3:] == "txt":
                continue
            with open(data_path,"r",encoding="utf8") as f:
                for baike in f:
                    baike = json.loads(baike)
                    d = baike["text"].strip()
                    if len(d) == 0:
                        continue
                    if np.random.random() > 0.2:
                        continue
                    
                    _ = re.findall(self.pattern, d)
                    
                    for ori_segment in _:
                        cut_segments = self.cut_segment(ori_segment)
                        total += len(ori_segment)
                        for segment in cut_segments:
                            if segment not in segment2freq:
                                segment_list.append(segment)
                                segment2freq[segment] = 1
                            else:
                                segment2freq[segment] += 1
            # if len(segment_list) > 10000:
            #     break
        print("总字数: ",total)   
       # segment = tuple(segment.encode("utf8"))
        print("去重segment: ",len(segment_list))
        for segment  in tqdm(iter(segment_list),desc="遍历segment中"):
            freq = segment2freq[segment]
            segment_token_ids = tuple(segment.encode("utf8"))
            segment_obj = Segment(None, None, freq)
            head = Node(segment_token_ids[0])
            head.segment = segment_obj
            segment_obj.head = head
            for token in segment_token_ids[1:]:
                tail = Node(token)
                pair = (head.value,token)
                if pair not in pair2nodes:
                    pair2nodes[pair] = set()
                pair2freq[pair] = pair2freq.get(pair,0) + freq
                pair2nodes[pair].add(head)
                tail.pre = head
                head.next = tail
                tail.segment = segment_obj
                head = tail
            segment_obj.tail = head
            
            # print(segment_obj)
            # break
        freq2pair_heap = [(-freq,pair) for (pair,freq) in pair2freq.items()]
        heapq.heapify(freq2pair_heap)
        self.segment2freq = segment2freq
        self.pair2nodes = pair2nodes
        self.pair2freq = pair2freq
        self.freq2pair_heap = freq2pair_heap
        
        
    def cut_segment(self,segment):
        ret = []
        size = 2000000
        if len(segment) <= size:
            return [segment]
        num = len(segment) // size
        for i in range(num):
            if i != num - 1:
                ret.append(segment[i*size:(i+1)*size])
            else:
                ret.append(segment[i*size:])
        return ret
                
                
    def create_freq2pair_heap(self):
        freq2pair_heap = [(-freq,pair) for (pair,freq) in self.pair2freq.items()]
        heapq.heapify(freq2pair_heap)
        self.freq2pair_heap = freq2pair_heap
        
    def find_max_freq_pair(self):
        if not self.freq2pair_heap:
            return None,None
        while self.freq2pair_heap:
            freq,pair = heapq.heappop(self.freq2pair_heap)
            freq = -freq
            if freq == self.pair2freq.get(pair,0):
                return freq,pair
        return None,None
    

    def update(self,pair,current_vocab_index):
        """
        self.segment2freq = segment2freq
        self.pair2nodes = pair2nodes
        self.pair2freq = pair2freq
        self.freq2pair_heap = freq2pair_heap

        """
        value_1,value_2 = pair
        new_value = current_vocab_index
        #新建合并规则
        rule = (pair,new_value)
        self.merge_rules.append(rule)
        self.vocab[new_value] = self.vocab[value_1] + self.vocab[value_2]
        #找到pair对应的所有head
        head2segments = set()
        print("pair freq: ",self.pair2freq[pair])
        for node in self.pair2nodes[pair]:
            head2segments.add(node.segment)
            
        for segment in head2segments:
            #新建node
            
            curr = segment.head
            while curr and curr.next:
                tmp_pair = (curr.value,curr.next.value)
                if tmp_pair != pair:
                    curr = curr.next
                    continue
                #更新频数
                #当前pair频数减少,#把新的pair频数更新到self.freq2pair_heap
                node = Node(new_value)
                #找出所在segment
                node.segment = segment
                self.pair2freq[pair] = self.pair2freq.get(pair,0) - segment.freq
               
                if self.pair2freq[pair] > 0:
                    heapq.heappush(self.freq2pair_heap, (-self.pair2freq[pair],pair))
                    self.pair2nodes[pair].discard(curr)#删除当前head
                    self.pair2nodes[pair].discard(curr.next)#删除head的右邻居
                else:
                    self.pair2freq.pop(pair)
                    self.pair2nodes.pop(pair)
                #从pair2nodes中删除head且尝试删除右边邻居
                
                #左边:新少,新增,#把新的pair频数更新到self.freq2pair_heap
                if curr.pre:
                    old_left_pair = (curr.pre.value,value_1)
                    self.pair2freq[old_left_pair] = self.pair2freq.get(old_left_pair,0) - segment.freq 
                    if self.pair2freq[old_left_pair] > 0:
                        heapq.heappush(self.freq2pair_heap, (-self.pair2freq[old_left_pair],old_left_pair))
                        self.pair2nodes[old_left_pair].discard(curr.pre)
                        self.pair2nodes[old_left_pair].discard(curr)
                    else:
                        self.pair2freq.pop(old_left_pair)
                        self.pair2nodes.pop(old_left_pair)
                    
                    new_left_pair = (curr.pre.value,new_value)
                    self.pair2freq[new_left_pair] = self.pair2freq.get(new_left_pair,0) + segment.freq
                    heapq.heappush(self.freq2pair_heap, (-self.pair2freq[new_left_pair],new_left_pair))
                    if new_left_pair not in self.pair2nodes:
                        self.pair2nodes[new_left_pair] = set()
                    self.pair2nodes[new_left_pair].add(curr.pre)
                
                    
                    
                #右边:新少,新增  ,#把新的pair频数更新到self.freq2pair_heap
            
                if curr.next.next:
                    
                    old_right_pair = (curr.next.value,curr.next.next.value)
                    self.pair2freq[old_right_pair] = self.pair2freq.get(old_right_pair,0) - segment.freq
                    
                    if self.pair2freq[old_right_pair] > 0:
                        heapq.heappush(self.freq2pair_heap, (-self.pair2freq[old_right_pair],old_right_pair))
                        self.pair2nodes[old_right_pair].discard(curr.next)
                        self.pair2nodes[old_right_pair].discard(curr.next.next)
                    else:
                        self.pair2freq.pop(old_right_pair)
                        self.pair2nodes.pop(old_right_pair)
                    
                    new_right_pair = (new_value,curr.next.next.value)
                    self.pair2freq[new_right_pair] = self.pair2freq.get(new_right_pair,0) + segment.freq
                    heapq.heappush(self.freq2pair_heap, (-self.pair2freq[new_right_pair],new_right_pair))
                    if new_right_pair not in self.pair2nodes:
                        self.pair2nodes[new_right_pair] = set()
                    self.pair2nodes[new_right_pair].add(node)
                #新的segment
                #合并左边
                
                #合并右边
                if curr.next.next:
                    node.next = curr.next.next
                    node.next.pre = node
                else:
                    segment.tail = node
                    
                if curr.pre:
                    node.pre = curr.pre
                    node.pre.next = node
                else:
                    segment.head = node
                curr = node.next
                
                
            #把新的pair频数更新到self.freq2pair_heap
            
    
    def train(self):
        """
        self.segment2freq = segment2freq
        self.pair2nodes = pair2nodes
        self.pair2freq = pair2freq
        self.freq2pair_heap = freq2pair_heap

        """
        #找到频数最大的pair
        current_vocab_index = 256
        while current_vocab_index < self.max_vocab_size:
            #找到最大频数的pair
            
            freq,pair = self.find_max_freq_pair()
            if not freq:
                break
            self.update(pair, current_vocab_index)
            current_vocab_index += 1
            print("已找到 {} 个token".format(len(self.vocab)))
            
            if current_vocab_index % 1000 == 0 :
                self.create_freq2pair_heap()
                
    def encode(self,segment):
        if len(segment) == 0:
            return []
        
        segment_token_ids = list(segment.encode("utf8"))
        if len(segment_token_ids) == 1:
            return segment_token_ids
        
        
        for rule in self.merge_rules:
            
            print("rule: ",rule)
            pair,new_value = rule
            new_segment_token_ids = []
            i = 0
            while i < len(segment_token_ids) - 1:
                tmp_pair = (segment_token_ids[i],segment_token_ids[i+1])
                if tmp_pair == pair:
                    new_segment_token_ids.append(new_value)
                    i += 2 
                else:
                    new_segment_token_ids.append(segment_token_ids[i])
                    i += 1 
            if i == len(segment_token_ids) - 1:
                new_segment_token_ids.append(segment_token_ids[i])
            segment_token_ids = new_segment_token_ids
            if len(segment_token_ids) == 1:
                break
        return segment_token_ids
                    
    def decode(self,token_ids):
        new_token_ids = []
        for iid in token_ids:
            new_token_ids.extend(self.vocab[iid])
        return bytes(new_token_ids).decode("utf8",errors="ignore")
                    
         
    def save(self):
        if not os.path.isdir(r"config_wiki"):
            os.makedirs(r"config_wiki")
        with open(r"config_wiki/vocab.json","w",encoding="utf8") as f:
            json.dump(self.vocab,f,ensure_ascii=False,indent=4)
        with open(r"config_wiki/merge_rules.json","w",encoding="utf8") as f:
            json.dump(self.merge_rules,f,ensure_ascii=False,indent=4)
        
                        
vocab_tool = Vocab()
st = time.time()
vocab_tool.train()
vocab_tool.save()
print("time: ",time.time()-st)
print(vocab_tool.merge_rules[:5])

words = []
for v,ids in vocab_tool.vocab.items():
    w = bytes(ids).decode("utf8",errors="ignore")
    if w:
        words.append(w)
        
words = dict(enumerate(words))
words = {w:i for i,w in words.items()}    

while True:
    try:
        w = str(input("输入: ")).strip()
        
        if not w:
            continue
        token_ids = vocab_tool.encode(w)
        print("{}: {}".format(w,token_ids))
        print("{}: {}".format(token_ids,vocab_tool.decode(token_ids)))
        if w in words:
            print("{} 在字典里".format(w))
        else:
            print("{} 不在字典里".format(w))
    except:
        pass

