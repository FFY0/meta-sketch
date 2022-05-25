import torch
import torch.nn as nn


class FunEmbedding(nn.Module):
    def __init__(self,VecDim_config,Embedding_config):
        super().__init__()
        self.in_dim = VecDim_config['query_dim']
        self.out_dim = VecDim_config['z_dim']
        self.hidden_layer_size = Embedding_config['hidden_layer_size']
        self.hidden1 = nn.Linear(self.in_dim, self.hidden_layer_size)
        self.norm = nn.BatchNorm1d(self.hidden_layer_size)
        self.hidden2 = nn.Linear(self.hidden_layer_size, self.hidden_layer_size//2)
        self.norm2 = nn.BatchNorm1d(self.hidden_layer_size//2)
        self.active_fun1 = nn.ReLU()
        self.active_fun = nn.ReLU()
        self.out_layer = nn.Sequential(
            nn.Linear(self.hidden_layer_size//2, self.out_dim),
            nn.ReLU()
        )

    def forward(self,x):
        x = self.hidden1(x)
        x = self.norm(x)
        x = self.active_fun1(x)
        x = self.hidden2(x)
        x = self.norm2(x)
        x = self.active_fun(x)
        return self.out_layer(x)


