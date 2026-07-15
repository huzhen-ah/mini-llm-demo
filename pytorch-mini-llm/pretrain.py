# -*- coding: utf-8 -*-
"""
"""


from tokenizer import Tokenizer

from models import create_pretrain_model
from losses import pretrain_loss
from metrics import pretrain_accuracy
import torch
from torch.utils.data import DataLoader

from train_utils import load_pretrain_data,PretrainDataset
from callbacks import Evaluate

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
    
    test_dataset = PretrainDataset(X_test, eos_id)
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
    
    evaluator = Evaluate(tokenizer_tool)
    
    def train(epoch,dataloader,model,optimizer,pretrain_loss,pretrain_accuray):
        model.train()
        total_loss = 0
        total_correct_tokens = 0
        total_valid_tokens = 0
        for X,Y in dataloader:
            X = X.to(device)
            Y = Y.to(device=device,dtype=torch.long)
            output = model(X)
            loss = pretrain_loss(output, Y, pad_id)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            correct_tokens,valid_tokens = pretrain_accuray(output, Y, pad_id)
            total_loss += loss.item()*valid_tokens
            total_correct_tokens += correct_tokens
            total_valid_tokens += valid_tokens
            
        print("Epoch: {},loss={},accuracy:{}".format(epoch,total_loss/total_valid_tokens,total_correct_tokens/total_valid_tokens))
    def test(epoch,dataloader,model,pretrain_loss,pretrain_accuray):
        model.eval()
        total_loss = 0
        total_correct_tokens = 0
        total_valid_tokens = 0
        with torch.no_grad():
            for X,Y in dataloader:
                X = X.to(device)
                Y = Y.to(device,dtype=torch.long)
                output = model(X)
                loss = pretrain_loss(output,Y,pad_id)
                correct_tokens,valid_tokens = pretrain_accuray(output, Y, pad_id)
                total_loss += loss.item()*valid_tokens
                total_correct_tokens += correct_tokens
                total_valid_tokens += valid_tokens
        print("Epoch: {},test_loss={},test_accuracy:{}".format(epoch,total_loss/total_valid_tokens,total_correct_tokens/total_valid_tokens))
        evaluator.on_epoch_end(model, epoch,device)
        
    epochs = 10
    for epoch in range(epochs):
        train(epoch,train_dataloader,model,optimizer,pretrain_loss,pretrain_accuracy)
        test(epoch,test_dataloader,model,pretrain_loss,pretrain_accuracy)
        
            
        
    
    
