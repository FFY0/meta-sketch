import random
import numpy as np
import torch


class DataLoaderFromTaskFile:
    def __init__(self, base_path, device, eval_task_num=20, eval_support_size=4999,
                 train_support_size_begin=100, train_support_size_end=4000):
        self.base_path = base_path
        self.task_data_loader = None
        self.eval_task_num = eval_task_num
        self.counts_list = []
        self.items = []
        self.eval_file_paths = []
        self.device = device
        self.merge_counts_tensor = None
        self.char_ids_tensor = None
        self.eval_support_set_size = eval_support_size
        self.train_support_size_begin = train_support_size_begin
        self.train_support_size_end = train_support_size_end
        self.sample_index = 0
        self.shift = False
        # load item
        self.load_item()
        # 用来打乱
        self.shuffle_index = [i for i in range(self.char_ids_tensor.shape[0])]
        self.support_size_ndarray = [i for i in range(train_support_size_begin, train_support_size_end)]
        random.shuffle(self.support_size_ndarray)
        self.support_size_ndarray_index = 0

    def shuffle(self):
        random.shuffle(self.shuffle_index)
        self.merge_counts_tensor = self.merge_counts_tensor[self.shuffle_index]
        self.char_ids_tensor = self.char_ids_tensor[self.shuffle_index]

    def online_sample_train_task(self):
        return self.online_sample_task(self.get_train_support_size())

    def generate_eval_tasks(self):
        for i in range(self.eval_task_num):
            yield self.online_sample_task(self.get_eval_support_size())

    def online_sample_task(self, support_size):
        if self.sample_index + support_size > self.char_ids_tensor.shape[0]:
            self.shuffle()
            self.sample_index = 0
            if self.sample_index + support_size > self.char_ids_tensor.shape[0]:
                self.shuffle()
                return self.char_ids_tensor, self.merge_counts_tensor
        item_vec = self.char_ids_tensor[self.sample_index:self.sample_index + support_size]
        item_frequency = self.merge_counts_tensor[self.sample_index:self.sample_index + support_size]
        self.sample_index += support_size
        return item_vec, item_frequency

    def get_eval_support_size(self):
        return self.eval_support_set_size

    def get_train_support_size(self):
        if self.support_size_ndarray_index + 1 >= len(self.support_size_ndarray) // 10:
            self.support_size_ndarray_index = -1
            random.shuffle(self.support_size_ndarray)
        self.support_size_ndarray_index += 1
        return self.support_size_ndarray[self.support_size_ndarray_index]

    def load_item(self):
        file = np.load(self.base_path)
        self.merge_counts_tensor = torch.tensor(file['counts'], device=self.device).float()
        self.char_ids_tensor = torch.tensor(file['queries'], device=self.device).float()
