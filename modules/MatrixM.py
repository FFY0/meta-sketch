import time

import torch


class MatrixM:
    def __init__(self, device, MatrixA_config, VecDim_config, learn_config):
        self.matrix_depth = VecDim_config['matrix_depth']
        self.height = VecDim_config['z_dim']
        self.device = device
        self.width = MatrixA_config['k']
        self.read_compensate = learn_config['read_compensate']
        self.matrix = None
        self.clear()

    def clear(self):
        self.matrix = torch.zeros(self.matrix_depth, self.height, self.width, device=self.device, requires_grad=True)

    def write(self, a, w, frequency):
        expand_f = frequency.expand_as(w.t())
        mul_w_t = w.t().mul(expand_f)
        add_matrix = mul_w_t.matmul(a)
        self.matrix = self.matrix + add_matrix

    def read(self, a, embedding_vec):
        res = a.matmul(self.matrix.transpose(1, 2))
        if self.read_compensate:
            weight = (a * a).sum(dim=2, keepdims=True).expand_as(res)
            res = res * (1 / weight)
        base_vec = torch.max(embedding_vec, torch.zeros_like(embedding_vec) + 0.001)
        zero_add_tensor = torch.where(abs(embedding_vec) < 0.0001, torch.zeros_like(embedding_vec) + 10000,
                                      torch.zeros_like(embedding_vec))
        base_num, _ = torch.min((res + zero_add_tensor) / base_vec, dim=-1)

        res_min, _ = torch.min(res, keepdim=True, dim=-1)
        res_minus_min = res - res_min
        res_minus_min = torch.where(abs(res_minus_min) < 0.0001, torch.zeros_like(embedding_vec) + 100000, res_minus_min)
        base_num2, _ = torch.min((res_minus_min + zero_add_tensor) / base_vec, dim=-1)
        res_min, _ = torch.min(res, dim=-1)

        res = res.transpose(0, 1)
        base_num = base_num.transpose(0, 1)
        base_num2 = base_num2.transpose(0, 1)
        res_min = res_min.transpose(0, 1)
        res = res.reshape(res.shape[0], -1)

        return res, base_num, base_num2, res_min
