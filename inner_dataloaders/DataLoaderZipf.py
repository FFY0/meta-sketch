import random
import numpy as np
import torch


class DataLoaderZipf:
    def __init__(self, base_path, device, eval_task_num=20, eval_support_size=4999, train_support_size_begin=100,
                 train_support_size_end=4000,
                 zipf_param_begin=1.1, zipf_param_end=None, mean_count=50, dyn_fre=10.0):
        self.base_path = base_path
        self.task_data_loader = None
        self.eval_task_num = eval_task_num
        self.counts_list = []
        self.items = []
        self.train_file_paths = []
        self.eval_file_paths = []
        self.device = device
        self.merge_counts_tensor = None
        self.char_ids_tensor = None
        self.eval_support_set_size = eval_support_size
        self.train_support_size_begin = train_support_size_begin
        self.train_support_size_end = train_support_size_end
        self.sample_index = 0
        self.shift = False
        self.mean_count = mean_count
        self.dyn_fre = dyn_fre
        self.generate_count = 0
        self.load_item()
        self.shuffle_index = [i for i in range(self.char_ids_tensor.shape[0])]
        self.support_size_ndarray = [i for i in range(train_support_size_begin, train_support_size_end)]
        random.shuffle(self.support_size_ndarray)
        self.zipf_param_begin = zipf_param_begin
        self.zipf_param_end = zipf_param_end
        if zipf_param_end is not None:
            if self.zipf_param_end <= self.zipf_param_begin:
                print('errors in zipf params')
                exit()
            self.zipf_param_gap = zipf_param_end - zipf_param_begin
        else:
            self.zipf_param_gap = None
        self.support_size_ndarray_index = 0
        self.shuffle()

    def get_zipf_param(self):
        if self.zipf_param_end is None:
            return self.zipf_param_begin
        else:
            return random.random() * self.zipf_param_gap + self.zipf_param_begin

    def get_zipf_simple_way_zeta_compensate(self, zipf_param, size, mean_count):
        x = torch.arange(1, size + 1, device=self.device).float()
        x = x ** (-zipf_param)
        y = x / x.sum()
        if self.dyn_fre is None:
            labels = y * mean_count * size
        else:
            labels = y * mean_count * size * (random.random() * self.dyn_fre + 0.1)
        labels_round = labels.round()
        num_of_one = round((labels.sum() - labels_round.sum()).item())
        zeros = torch.zeros(size - abs(num_of_one), device=self.device)
        ones = torch.ones(abs(num_of_one), device=self.device)
        add_labels = torch.cat((zeros, ones))
        labels = labels + add_labels
        return torch.round(labels)

    def shuffle(self):
        random.shuffle(self.shuffle_index)
        if self.generate_count % 5 == 0:
            self.merge_counts_tensor = self.get_zipf_simple_way_zeta_compensate(self.get_zipf_param(),
                                                                                self.merge_counts_tensor.shape[0],
                                                                                self.mean_count)
            self.merge_counts_tensor = self.merge_counts_tensor[self.shuffle_index]
        else:
            self.generate_count += 1
            self.merge_counts_tensor = self.merge_counts_tensor[self.shuffle_index]
        random.shuffle(self.shuffle_index)
        self.char_ids_tensor = self.char_ids_tensor[self.shuffle_index]

    def online_sample_train_task(self):
        return self.online_sample_task(self.get_train_support_size())

    def generate_eval_tasks(self):
        for i in range(self.eval_task_num):
            yield self.online_sample_task(self.get_eval_support_size())

    def filter_zero(self, item_vec, item_frequency):
        index = torch.where(item_frequency > 0.5)
        item_vec = item_vec[index]
        item_frequency = item_frequency[index]
        if item_vec.shape[0] <= 1:
            return self.online_sample_train_task()
        return item_vec, item_frequency

    def online_sample_task(self, support_size):
        if self.sample_index + support_size > self.char_ids_tensor.shape[0]:
            self.shuffle()
            self.sample_index = 0
            if self.sample_index + support_size > self.char_ids_tensor.shape[0]:
                self.shuffle()
                return self.filter_zero(self.char_ids_tensor, self.merge_counts_tensor)
        item_vec = self.char_ids_tensor[self.sample_index:self.sample_index + support_size]
        item_frequency = self.merge_counts_tensor[self.sample_index:self.sample_index + support_size]
        self.sample_index += support_size
        return self.filter_zero(item_vec, item_frequency)

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
