# -*- coding: utf-8 -*-
"""

"""

import json
import regex as re
class Tokenizer:
    def __init__(self,vocab_path=r"tokenizer_config/vocab.json",merge_rules_path=r"tokenizer_config/merge_rules.json"):
        self.vocab_path = vocab_path
        self.merge_rules_path = merge_rules_path
        self.pattern = r"""'(?:s|t|re|ve|m|ll|d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+[\s]*"""
        self.init_tokenizer()
        
    def init_tokenizer(self):
        self.special_ids = {}
        with open(self.vocab_path,"r",encoding="utf8") as f:
            self.vocab = {int(i):j for i,j in json.load(f).items()}
    
        for token in ["<bos>", "<eos>", "<unk>", "<pad>"]:
            self.vocab[len(self.vocab)] = list(token.encode("utf8"))
            self.special_ids[token] = len(self.vocab) - 1
        with open(self.merge_rules_path,"r",encoding="utf8") as f:
            _merge_rules = json.load(f)
        merge_rules = {}
        for i,(pair,new_token_id) in enumerate(_merge_rules):
            merge_rules[tuple(pair)] = [tuple(pair),new_token_id,i]
        self.merge_rules = merge_rules
        
        
    def get_segment_pairs(self,segment_ids):
        return set(zip(segment_ids,segment_ids[1:]))
    
    def get_current_merge_rule(self,pairs):
        first_rule = None
        for pair in pairs:
            rule = self.merge_rules.get(pair,None)
            if rule is None:
                continue
            if first_rule is None:
                first_rule = rule
            else:
                if rule[-1] < first_rule[-1]:
                    first_rule = rule
        return first_rule
    
    def encode_text(self,text):
        """完整文本入口: 先按 regex 切分，再对每个 segment 做 BPE。"""
        segments = re.findall(self.pattern, text)
        text_tokens = []
        for segment in segments:
            segment_tokens = self.encode(segment)
            if len(segment_tokens) > 0:
                text_tokens.extend(segment_tokens)
        return text_tokens
    
    def encode(self,segment):
        """segment入口: 直接对 segment 做 BPE。"""
        if len(segment.strip()) == 0:
            return []
        segment_token_ids = list(segment.encode("utf8"))
        if len(segment_token_ids) == 1:
            return segment_token_ids
        while True:
            pairs = self.get_segment_pairs(segment_token_ids)
            # print("pairs: ",pairs)
            if len(pairs) == 0:
                break
            rule = self.get_current_merge_rule(pairs)
            if rule is None:
                break
            
            
            pair,new_token_id,_ = rule
            new_segment_token_ids = []
            i = 0
            while i < len(segment_token_ids) - 1:
                tmp_pair = (segment_token_ids[i],segment_token_ids[i+1])
                if tmp_pair == pair:
                    new_segment_token_ids.append(new_token_id)
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
            try:
                new_token_ids.extend(self.vocab[iid])
            except KeyError:
                raise ValueError(f"Unknown token id: {iid}") from None
        return bytes(new_token_ids).decode("utf8",errors="ignore")
    
# tokenizer = Tokenizer()
# words = []
# for v,ids in tokenizer.vocab.items():
#     w = bytes(ids).decode("utf8",errors="ignore")
#     if w:
#         words.append(w)
        
# words = dict(enumerate(words))
# words = {w:i for i,w in words.items()}    

# while True:
#     # try:
#     w = str(input("输入: ")).strip()
    
#     if not w:
#         continue
#     token_ids = tokenizer.encode(w)
#     print("{}: {}".format(w,token_ids))
#     print("{}: {}".format(token_ids,tokenizer.decode(token_ids)))
#     if w in words:
#         print("{} 在字典里".format(w))
#     else:
#         print("{} 不在字典里".format(w))
#     # except:
#     #     pass

