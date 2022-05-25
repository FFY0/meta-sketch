from modules.FunEmbedding import *
from modules.MatrixM import *
from modules.FunOutMLP import *
from modules.AddressNet import AddressNet
from util.AutoWeightedLoss import AutomaticWeightedLoss
from modules.SparseSoftmax import *


class BaseModel(nn.Module):
    def __init__(self, device, train_config):
        super().__init__()
        self.device = device
        self.fun_out = FunOutMLP(FunOutMLP_config=train_config['FunOutMLP_config'],
                                 learn_config=train_config['learn_config'],
                                 VecDim_config=train_config['VecDim_config']).to(device=self.device)
        self.input_enc = FunEmbedding(VecDim_config=train_config['VecDim_config'],
                                      Embedding_config=train_config['Embedding_config']).to(device=self.device)
        self.matrix_m = MatrixM(self.device, MatrixA_config=train_config['MatrixA_config'],
                                VecDim_config=train_config['VecDim_config'], learn_config=train_config['learn_config'])
        self.address_net = AddressNet(AddressNet_config=train_config['AddressNet_config'],
                                      VecDim_config=train_config['VecDim_config']).to(device=self.device)
        self.sigmoid = torch.nn.Sigmoid()
        self.sparse_max = Sparsemax()
        self.automaticWeightedLoss = AutomaticWeightedLoss(2).to(device)
        self.matrix_a = torch.nn.Parameter(
            torch.rand(train_config['VecDim_config']['matrix_depth'], train_config['VecDim_config']['r_dim'],
                       train_config['MatrixA_config']['k'], device=device, requires_grad=True))

    def write(self, support, label):
        z = self.input_enc(support)
        a = self.getAddress(z)
        self.matrix_m.write(a, z, label)

    def normalize_a(self):
        with torch.no_grad():
            matrix = self.matrix_a.data
            r_dim = matrix.shape[1]
            matrix = matrix.mul(matrix)
            matrix = torch.sqrt(matrix.sum(dim=1, keepdim=True))
            matrix = matrix.repeat(1, r_dim, 1)
            self.matrix_a.data = self.matrix_a.data.div(matrix)

    def clear_memory(self):
        self.matrix_m.clear()

    def getAddress(self, q):
        address_vec = self.address_net(q)
        a = address_vec.matmul(self.matrix_a)
        a = self.sparse_max(a)
        return a

    def get_sparse(self, a):
        sig_a = self.sigmoid(a * 1000)
        sig_a = (sig_a - 0.5) * 2
        sig_a_sum = torch.sum(sig_a, dim=-1)
        return sig_a_sum

    def forward(self, query, stream_size):
        q = self.input_enc(query)
        a = self.getAddress(q)
        read_info_list = self.matrix_m.read(a, q)
        out = self.fun_out(q, read_info_list, stream_size)
        return out.view(out.size(0)), q
