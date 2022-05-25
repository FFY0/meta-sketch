import torch
import torch.nn as nn


class FunOutMLP(nn.Module):
    def __init__(self, FunOutMLP_config, learn_config, VecDim_config):
        super().__init__()
        self.hidden_layer_size = FunOutMLP_config['hidden_layer_size']
        self.stream_size_input_num = learn_config['stream_size_input_num']
        self.input_dim = VecDim_config['z_dim'] * VecDim_config['matrix_depth'] + VecDim_config['z_dim'] + \
                         self.stream_size_input_num + VecDim_config['matrix_depth'] * 3
        self.layer_2_adjust_num = nn.Linear(self.input_dim, self.hidden_layer_size)
        self.hidden1 = nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
        self.active1 = nn.ReLU()
        self.hidden2 = nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
        self.active2 = nn.ReLU()
        self.out_layer = nn.Sequential(
            nn.Linear(self.hidden_layer_size, 1),
        )

    def forward(self, q, read_info_tuple, stream_size):
        if self.stream_size_input_num == 0:
            input_vec = torch.cat((q,) + read_info_tuple, dim=1)
        else:
            stream_size = stream_size.expand(size=(q.shape[0], self.stream_size_input_num))
            input_vec = torch.cat((stream_size, q,) + read_info_tuple, dim=1)
        input_vec = self.layer_2_adjust_num(input_vec)
        hidden1_vec = self.hidden1(input_vec)
        active1_vec = self.active1(hidden1_vec)
        hidden2_vec = self.hidden2(active1_vec)
        # residual connnections
        hidden2_vec = hidden2_vec + input_vec
        active2_vec = self.active2(hidden2_vec)
        out = self.out_layer(active2_vec)
        return out
