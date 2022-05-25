import torch
import torch.nn as nn


class AddressNet(nn.Module):
    def __init__(self, AddressNet_config, VecDim_config):
        super().__init__()
        self.hidden_layer_size = AddressNet_config['hidden_layer_size']
        self.input_dim = VecDim_config['z_dim']
        self.output_dim = VecDim_config['r_dim']
        self.input_layer = nn.Linear(self.input_dim, self.hidden_layer_size)
        self.hidden1 = nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
        self.hidden2 = nn.Linear(self.hidden_layer_size, self.output_dim)
        self.active2 = nn.LeakyReLU()
        self.active1 = nn.LeakyReLU()
        self.norm1 = nn.BatchNorm1d(self.hidden_layer_size)
        self.norm2 = nn.BatchNorm1d(self.hidden_layer_size)

    def forward(self, q):
        input_vec = self.input_layer(q)
        input_vec = self.norm1(input_vec)
        input_vec = self.active1(input_vec)
        input_vec = self.hidden1(input_vec)
        input_vec = self.norm2(input_vec)
        input_vec = self.active2(input_vec)
        input_vec = self.hidden2(input_vec)
        return input_vec
