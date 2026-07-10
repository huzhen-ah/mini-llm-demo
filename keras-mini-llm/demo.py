#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:47:42 2026

@author: huzhen
"""
import os
from bbpe_trainer import BBPETrainer
from tokenizer import Tokenizer

from models import create_pretrain_model
from losses import create_loss,sft_loss,dpo_loss
from metrics import MaskedAccuracy,SFTAccuracy
from train_utils import load_pretrain_data,data_generator,load_sft_data,data_generator_sft,load_dpo_data,pre_infer_dpo_data,data_generator_dpo
from callbacks import Evaluate,Lora_Evaluate,DPO_Evaluate
from weight_utils import apply_train_weights,save_model_weights
from lora_utils import mark_only_lora_as_trainable,merge_lora_weights
from interface import Interface

if __name__ == "__main__":
    
    """
    STEP_1: 
        先训练BBPE,得到vocab,merge_rules
    """
    print(".........BBPE阶段开始.........")
    data_path = r"data/*.txt"       
    vocab_path = r"tokenizer_config/vocab.json"
    merge_rules_path = r"tokenizer_config/merge_rules.json"
    if not os.path.isfile(vocab_path) or not os.path.isfile(merge_rules_path):
        trainer = BBPETrainer(data_path)
        trainer.build()
        trainer.train()
        trainer.save(vocab_path,merge_rules_path)
    print(".........BBPE阶段结束.........")
    
    """
    STEP_2:
        用语料做next_token_prediction训练
    """
    print(".........pretrain阶段开始.........")
    
    
    
    
        
    if not os.path.isfile(r"models/0_k2v_weights.pkl"):
        
        
        tokenizer_tool = Tokenizer()
        eos_id = tokenizer_tool.special_ids["<eos>"]   
        pad_id = tokenizer_tool.special_ids["<pad>"]
        
        epochs = 1
        num_block = 4
        num_head = 2
        embedding_size = 64
        vocab_size = len(tokenizer_tool.vocab)
        
        
        context_size = 200
        batch_size = 128
        configs = {
                    "num_block" : num_block,
                    "num_head" : num_head,
                    "embedding_size" : embedding_size,
                    "hidden_channels" : embedding_size * 2,
                    "vocab_size" : vocab_size,
                    "pad_id" : pad_id
                  }
        
        model = create_pretrain_model(configs)
        model.summary()
        my_loss = create_loss(pad_id)
        
        model.compile(optimizer="adam",loss=my_loss,metrics=[MaskedAccuracy(pad_id)])
                                             # sparse_categorical_crossentropy
        
    
        
        data_pattern = r"data/*.txt"
        X_train,X_test = load_pretrain_data(tokenizer_tool, data_pattern, context_size,test_ratio=0.01)              
        
        print("训练样本数: ",len(X_train))
        
        model.fit(data_generator(X_train,batch_size,eos_id),
                  epochs=epochs,
                  steps_per_epoch=len(X_train)//(batch_size*1)+1,
                  callbacks=[Evaluate(tokenizer_tool)])       
            
    print(".........pretrain阶段结束.........")
    




    """
    STEP_3:
        用SFT_data微调pretrain的base模型，微调方式是LoRA
    """

    print(".........LoRA_SFT阶段开始.........")
    if not os.path.isfile(r"lora_sft_weights/0_k2v_lora_merged.pkl"):
        tokenizer_tool = Tokenizer()
        eos_id = tokenizer_tool.special_ids["<eos>"]   
        pad_id = tokenizer_tool.special_ids["<pad>"]
        
        num_block = 4
        num_head = 2
        embedding_size = 64
        vocab_size = len(tokenizer_tool.vocab)
        use_lora = True
        
        context_size = 200
        batch_size = 64
        configs = {
                    "num_block" : num_block,
                    "num_head" : num_head,
                    "embedding_size" : embedding_size,
                    "hidden_channels" : embedding_size * 2,
                    "use_lora" : use_lora,
                    "vocab_size" : vocab_size,
                    "pad_id" : pad_id
                  }
        weight_map_path = r"models/0_k2v_weights.pkl"
        merged_weight_map_path = "lora_sft_weights/0_k2v_lora_merged.pkl"
        model = create_pretrain_model(configs)
        apply_train_weights(model, weight_map_path)
        mark_only_lora_as_trainable(model)
        model.summary()
        
        
    
        # print("打印trainable weights")
        # for w in model.trainable_weights:
        #     w_path = w.path
        #     print("w_path: ",w_path)
        model.compile(optimizer="adam",loss=sft_loss,metrics=[SFTAccuracy()])
                                             # sparse_categorical_crossentropy
        
        
        
        
        data_path = r"SFT_data/emperor_sft_messages_v1.jsonl"
        X_train,X_test = load_sft_data(data_path,tokenizer_tool, context_size,test_ratio=0.01)              
        
        print("训练样本数: ",len(X_train))
        # print("X: ",X_train[:3])
        
        
    
                
                
        epochs = 1      
        model.fit(data_generator_sft(X_train,batch_size,eos_id,pad_id),
                  epochs=epochs,
                  steps_per_epoch=len(X_train)//(batch_size*1)+1,
                  callbacks=[Lora_Evaluate(tokenizer_tool)]
                  )  
        
        merge_lora_weights(model)
        save_model_weights(model, merged_weight_map_path)
        
    print(".........LoRA_SFT阶段结束.........")
    """
    STEP_4:
        用DPO_data做偏好优化，base模型用的是SFT阶段merged，微调方式是LoRA
    """
    print(".........LoRA_DPO阶段开始.........")
    if not os.path.isfile(r"lora_dpo_weights/0_k2v_lora_merged.pkl"):
        tokenizer_tool = Tokenizer()
        eos_id = tokenizer_tool.special_ids["<eos>"]   
        pad_id = tokenizer_tool.special_ids["<pad>"]
        
        num_block = 4
        num_head = 2
        embedding_size = 64
        vocab_size = len(tokenizer_tool.vocab)
        use_lora = True
        
        context_size = 200
        batch_size = 64
        configs = {
                    "num_block" : num_block,
                    "num_head" : num_head,
                    "embedding_size" : embedding_size,
                    "hidden_channels" : embedding_size * 2,
                    "use_lora" : use_lora,
                    "vocab_size" : vocab_size,
                    "pad_id" : pad_id
                  }
        
        lora_merged_weights_path = r"lora_sft_weights/0_k2v_lora_merged.pkl"
        
        model = create_pretrain_model(configs)
        apply_train_weights(model, lora_merged_weights_path)
        
        mark_only_lora_as_trainable(model)
        model.summary()
        

        # print("打印trainable weights")
        # for w in model.trainable_weights:
        #     w_path = w.path
        #     print("w_path: ",w_path)
        model.compile(optimizer="adam",loss=dpo_loss())
                                             # sparse_categorical_crossentropy
        
        
        dpo_data_path = r"DPO_data/emperor_dpo_pairs_v1.jsonl"
        X_train,X_test = load_dpo_data(dpo_data_path, tokenizer_tool, context_size)
        X_train = pre_infer_dpo_data(X_train, model, eos_id, pad_id)
        X_test = pre_infer_dpo_data(X_test, model, eos_id, pad_id)
        
        print("训练样本数: ",len(X_train))
        # print("X: ",X_train[:3])
        
       
        model.fit(data_generator_dpo(X_train, eos_id, pad_id,batch_size=batch_size),
                  epochs=1,
                  steps_per_epoch=len(X_train)//(batch_size*1)+1,
                  callbacks=[DPO_Evaluate(tokenizer_tool)]
                  )  
        merge_lora_weights(model)
        lora_merged_weights_path = r"lora_dpo_weights/0_k2v_lora_merged.pkl"
        save_model_weights(model, lora_merged_weights_path)
    print(".........LoRA_DPO阶段结束.........")
    
    """
    STTEP_5:
        推理测试
    """
    print(".........inference阶段开始.........")
    interface = Interface(Tokenizer())

    prefill_model_configs = {
                                "num_block":4,
                                "num_head":2,
                                "embedding_size":64,
                                "use_lora":False,
                                "weight_map_path":r"lora_dpo_weights/0_k2v_lora_merged.pkl"
                            }
    interface.init_prefill_model(prefill_model_configs)

    interface.init_decode_model(prefill_model_configs)

    text_1 = "秦始皇是谁？"
    text_2 = "汉武帝是谁？"
    text_3 = "唐太宗谁谁？"
    text_4 = "赵匡胤是谁？"

    prompts = ["      ",text_1,text_2,text_3,text_4]
    ret = interface.predict(prompts)
    for i in range(len(ret)):
        text = interface.tokenizer.decode(ret[i]["prompt"]+ret[i]["generated"])
        print("text: ",text)
