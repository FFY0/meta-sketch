import os
from adapt_Sketch import adapt_Sketch


def save_detail(sketch, detail, path):
    self = sketch
    if not os.path.exists(os.path.join(sketch.project_root, 'logDir/{}'.format(self.dataset_name))):
        os.mkdir(os.path.join(sketch.project_root, 'logDir/{}'.format(self.dataset_name)))
    if not os.path.exists(
            os.path.join(sketch.project_root, 'logDir/{}/{}'.format(self.dataset_name, self.base_record_path))):
        os.mkdir(os.path.join(sketch.project_root, 'logDir/{}/{}'.format(self.dataset_name, self.base_record_path)))
    detail_file = open(path, 'w', newline='',
                       encoding='utf-8')
    detail_file.write(detail)
    detail_file.close()


def init_Sketch(train_config, cuda_num, exp_name, log_dir_name):
    sketch = adapt_Sketch(train_config, cuda_num=cuda_num, exp_name=exp_name, log_dir_name=log_dir_name)
    return sketch
