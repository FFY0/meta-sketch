import json
import os
import sys
sys.path.append('../..')
sys.path.append('..')
import torch
from inner_dataloaders.DataLoaderZipf import DataLoaderZipf
from inner_dataloaders.DataloaderFromTaskFile import DataLoaderFromTaskFile
from outer_dataloaders.OuterLoaderDynamicFre import OuterLoaderDynamicFre
from outer_dataloaders.OuterLoaderShuffleFre import OuterLoaderShuffleFre

"""
basic sketch 5k items 9kB budget
"""
from util.adapt_util_func import *


def train(train_data_loader, others_data_loader, train_loader_description, others_detail_description):
    sketch = init_Sketch(train_config=train_config, cuda_num=train_config['cuda_num'],
                         exp_name=train_config['exp_name'], log_dir_name=train_config['log_dir_name'])
    sketch.set_data_loader(train_data_loader, others_data_loader, train_loader_description, others_detail_description)
    save_detail(sketch, 'code:{}\n'.format(os.path.split(__file__)[-1].split(".")[0] + json.dumps(train_config)),
                os.path.join(sketch.project_root,
                             'logDir/{}/{}/detail.txt'.format(sketch.dataset_name, sketch.base_record_path)))
    sketch.train()


train_config = {
    'cuda_num': 0,
    'exp_name': 'basic_sketch',
    'log_dir_name': '9KB_budget_5K_item_',
    'log_gap': 10000,
    'learn_config': {
        'train_steps': 5000000,
        'update_lr': 1e-04,
        'base_path': '../../IP_taskFile/5000.npz',
        'read_compensate': True,
        'stream_size_input_num': 4,
        # group: 1*eval_support_size/eval_groups_num  2*eval_support_size/eval_groups_num ........ eval_groups_num*eval_support_size/eval_groups_num
        'eval_groups_num': 5
    },
    'Embedding_config': {'hidden_layer_size': 128},
    'MatrixA_config': {'k': 50},
    'FunOutMLP_config': {'hidden_layer_size': 256},
    'AddressNet_config': {'hidden_layer_size': 48, },
    'VecDim_config': {'query_dim': 24, 'z_dim': 23, 'r_dim': 5, 'matrix_depth': 2},
    'dataset_config': {'train_support_size_begin': 2, 'train_support_size_end': 5000, 'eval_support_size': 5000}
}

if __name__ == '__main__':
    learn_config = train_config['learn_config']
    dataset_config = train_config['dataset_config']
    torch.cuda.set_device(train_config['cuda_num'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # init train_dataloader & eval_dataloader
    original_data_loader = OuterLoaderDynamicFre(OuterLoaderShuffleFre(
        DataLoaderFromTaskFile(base_path=learn_config['base_path'], device=device,
                               eval_support_size=dataset_config['eval_support_size'],
                               train_support_size_end=dataset_config['train_support_size_end']
                               , train_support_size_begin=dataset_config['train_support_size_begin'])))

    zipf_data_loader = DataLoaderZipf(base_path=learn_config['base_path'], device=device,
                                      eval_support_size=dataset_config['eval_support_size'],
                                      train_support_size_end=dataset_config['train_support_size_end']
                                      , train_support_size_begin=dataset_config['train_support_size_begin'],
                                      zipf_param_begin=0.8, zipf_param_end=1.3, mean_count=50, dyn_fre=10)

    train_data_loader = zipf_data_loader
    train_loader_description = "zipf_data_loader"
    others_data_loader = (original_data_loader,)
    others_detail_description = ("original_data_loader",)
    train(train_data_loader, others_data_loader, train_loader_description, others_detail_description)
