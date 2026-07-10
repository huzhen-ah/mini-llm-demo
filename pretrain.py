# -*- coding: utf-8 -*-
"""
"""


from tokenizer import Tokenizer

from models import create_pretrain_model
from losses import create_loss
from metrics import MaskedAccuracy
from train_utils import load_pretrain_data,data_generator
from callbacks import Evaluate

if __name__ == "__main__":
    
    tokenizer_tool = Tokenizer()
    eos_id = tokenizer_tool.special_ids["<eos>"]   
    pad_id = tokenizer_tool.special_ids["<pad>"]
    
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
    print("X: ",X_train[:3])
    
    
    
    
              
    
    
                
            
            
            
    model.fit(data_generator(X_train,batch_size,eos_id),
              epochs=1,
              steps_per_epoch=len(X_train)//(batch_size*1)+1,
              callbacks=[Evaluate(tokenizer_tool)])       
        
    
    
