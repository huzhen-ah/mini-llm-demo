# -*- coding: utf-8 -*-
"""
"""


from tokenizer import Tokenizer

from models import create_pretrain_model
from losses import pretrain_loss
from metrics import pretrain_accuray
import torch
from torch.utils.data import DataLoader

from train_utils import load_pretrain_data,PretrainDataset

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    tokenizer_tool = Tokenizer()
    eos_id = tokenizer_tool.special_ids["<eos>"]   
    pad_id = tokenizer_tool.special_ids["<pad>"]
    
    num_block = 4
    num_head = 2
    embedding_dim = 64
    vocab_size = len(tokenizer_tool.vocab)
    
    
    context_size = 200
    batch_size = 128
    
    configs = {
                "num_block" : num_block,
                "num_head" : num_head,
                "embedding_dim" : embedding_dim,
                "hidden_channels" : embedding_dim * 2,
                "vocab_size" : vocab_size,
                "pad_id" : pad_id
              }
    
    model = create_pretrain_model(configs)
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    
    
    data_pattern = r"data/*.txt"
    X_train,X_test = load_pretrain_data(tokenizer_tool, data_pattern, context_size,test_ratio=0.01)              
    
    print("训练样本数: ",len(X_train))
    print("X: ",X_train[:3])
    train_dataset = PretrainDataset(X_train, eos_id)
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    
    epochs = 100
    
    for epoch in range(epochs):
        total_loss = 0
        num_batch = 0
        total_correct = 0
        total_valid = 0
        for X,Y in train_dataloader:
            X = X.to(device)
            Y = Y.to(device=device,dtype=torch.long)
            output = model(X)
            loss = pretrain_loss(output, Y, pad_id)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct,valid = pretrain_accuray(output, Y, pad_id)
            total_correct += correct
            total_valid += valid
            num_batch += 1
        print("Epoch: {},loss={},accuracy:{}".format(epoch,total_loss/num_batch,total_correct/total_valid))
        
        
            
        
    
    
