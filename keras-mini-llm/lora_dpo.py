# -*- coding: utf-8 -*-
"""
"""


from tokenizer import Tokenizer

from models import create_pretrain_model
from losses import dpo_loss
from train_utils import load_dpo_data,pre_infer_dpo_data,data_generator_dpo
from weight_utils import apply_train_weights
from lora_utils import mark_only_lora_as_trainable
from callbacks import DPO_Evaluate

if __name__ == "__main__":
    
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
    

    print("打印trainable weights")
    for w in model.trainable_weights:
        w_path = w.path
        print("w_path: ",w_path)
    model.compile(optimizer="adam",loss=dpo_loss())
                                         # sparse_categorical_crossentropy
    
    
    dpo_data_path = r"DPO_data/emperor_dpo_pairs_v1.jsonl"
    X_train,X_test = load_dpo_data(dpo_data_path, tokenizer_tool, context_size)
    X_train = pre_infer_dpo_data(X_train, model, eos_id, pad_id)
    X_test = pre_infer_dpo_data(X_test, model, eos_id, pad_id)
    
    print("训练样本数: ",len(X_train))
    print("X: ",X_train[:3])
    
   
    model.fit(data_generator_dpo(X_train, eos_id, pad_id,batch_size=batch_size),
              epochs=1,
              steps_per_epoch=len(X_train)//(batch_size*1)+1,
              callbacks=[DPO_Evaluate(tokenizer_tool)]
              )  
        
    
    
