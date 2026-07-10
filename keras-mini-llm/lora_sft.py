# -*- coding: utf-8 -*-
"""
"""


from tokenizer import Tokenizer

from models import create_pretrain_model
from losses import sft_loss
from metrics import SFTAccuracy
from train_utils import load_sft_data,data_generator_sft
from weight_utils import apply_train_weights
from lora_utils import mark_only_lora_as_trainable
from callbacks import Lora_Evaluate



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
    weight_map_path = r"models/0_k2v_weights.pkl"
    model = create_pretrain_model(configs)
    apply_train_weights(model, weight_map_path)
    mark_only_lora_as_trainable(model)
    model.summary()
    
    

    print("打印trainable weights")
    for w in model.trainable_weights:
        w_path = w.path
        print("w_path: ",w_path)
    model.compile(optimizer="adam",loss=sft_loss,metrics=[SFTAccuracy()])
                                         # sparse_categorical_crossentropy
    
    
    
    
    data_path = r"SFT_data/emperor_sft_messages_v1.jsonl"
    X_train,X_test = load_sft_data(data_path,tokenizer_tool, context_size,test_ratio=0.01)              
    
    print("训练样本数: ",len(X_train))
    print("X: ",X_train[:3])
    
    

            
            
    epochs = 1      
    model.fit(data_generator_sft(X_train,batch_size,eos_id,pad_id),
              epochs=epochs,
              steps_per_epoch=len(X_train)//(batch_size*1)+1,
              callbacks=[Lora_Evaluate(tokenizer_tool)]
              )  
    
    
    
    
    
