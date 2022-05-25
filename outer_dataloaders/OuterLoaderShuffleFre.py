import random


def shuffle(item_frequency):
    index = [i for i in range(item_frequency.shape[0])]
    random.shuffle(index)
    item_frequency = item_frequency[index]
    return item_frequency


class OuterLoaderShuffleFre:
    def __init__(self, data_loader):
        self.task_data_loader = data_loader

    def online_sample_train_task(self):
        item_vec, item_frequency = self.task_data_loader.online_sample_train_task()
        return item_vec, shuffle(item_frequency)

    def generate_eval_tasks(self):
        for item_vec, item_frequency in self.task_data_loader.generate_eval_tasks():
            yield item_vec, item_frequency