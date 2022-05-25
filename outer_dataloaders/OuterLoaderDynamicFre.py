"""
Outer Loader multiply the frequency by  random times
"""
import torch


class OuterLoaderDynamicFre:
    def __init__(self, data_loader, times_upper_bound=10.0, device='cuda'):
        self.task_data_loader = data_loader
        self.dynamic = None
        self.dynamic_index = None
        self.times_upper_bound = times_upper_bound
        self.device = device
        self.refresh_dynamic()

    def refresh_dynamic(self):
        self.dynamic = (torch.rand(100000, device=self.device) * self.times_upper_bound) + 0.0001
        self.dynamic_index = 0

    def frequency_dynamic(self, frequency):
        if self.dynamic_index + 1 >= 100000:
            self.refresh_dynamic()
        self.dynamic_index += 1
        return torch.round(frequency * self.dynamic[self.dynamic_index]) + 1.0

    def online_sample_train_task(self):
        item_vec, item_frequency = self.task_data_loader.online_sample_train_task()
        return item_vec, self.frequency_dynamic(item_frequency)

    def generate_eval_tasks(self):
        for item_vec, item_frequency in self.task_data_loader.generate_eval_tasks():
            yield item_vec, item_frequency
