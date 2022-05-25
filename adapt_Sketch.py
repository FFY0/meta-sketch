import datetime
import sys
import time

import numpy as np
import torch.optim
from sklearn import metrics
import json
import csv
import os
from BaseModel import *
import copy


class adapt_Sketch:
    def __init__(self, train_config, cuda_num=0, exp_name="default_exp", log_dir_name='', ):
        self.train_config = train_config
        self.mse_fun = torch.nn.MSELoss()
        self.base_record_path = log_dir_name
        # log path
        self.base_record_path += datetime.datetime.now().strftime('%m_%d_%H_%M_%S')
        torch.cuda.set_device(cuda_num)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BaseModel(self.device, train_config)
        self.dataset_name = exp_name
        self.eval_groups_num = self.train_config['learn_config']['eval_groups_num']
        self.log_gap = self.train_config['log_gap']
        self.save_model_gap = self.train_config['log_gap']
        self.train_task_data_loader = None
        self.others_data_loader = None
        self.task_data_loader = None
        self.file_list = None
        self.csv_writer_list = None
        self.train_loader_detail = None
        self.watched_loader_detail_list = None
        self.project_root = os.getcwd().split('exp_code')[0]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=train_config['learn_config']['update_lr'])
        self.loss_fn = self.ARE_MSE_loss_func

    def ARE_MSE_loss_func(self, pred, label, embedding_vec):
        loss1 = self.mse_fun(pred, label)
        loss2 = abs((pred - label) / label).mean()
        # loss3 = embedding_vec.sum()
        return self.model.automaticWeightedLoss(loss1, loss2)

    def set_data_loader(self, train_task_data_loader, others_data_loader, train_loader_detail,
                        watched_loader_detail_list):
        self.train_task_data_loader = train_task_data_loader
        self.others_data_loader = others_data_loader
        self.task_data_loader = train_task_data_loader
        self.train_loader_detail = train_loader_detail
        self.watched_loader_detail_list = watched_loader_detail_list

    def open_log_file(self):
        self.file_list = []
        self.csv_writer_list = []
        info_example = ["sparse_mean", "sparse_var", "loss", "label_varience", "pred_varience", "ARE",
                        "AAE", "dif_a", "dif_matrix", 'accuracy', "racall", "precision", "F1"]
        self.file_list.append(open(os.path.join(self.project_root, 'logDir/{}/{}/train_log{}.csv'.format
        (self.dataset_name, self.base_record_path, self.train_loader_detail)), 'w', newline='', encoding='utf-8'))
        self.csv_writer_list.append(csv.writer(self.file_list[-1]))
        f = self.file_list[-1]
        csv_writer = self.csv_writer_list[-1]
        row = ['step', 'embedding_mean', 'embedding_var', 'embedding_none_zero']
        final_task_loader = self.task_data_loader
        while final_task_loader.task_data_loader is not None:
            final_task_loader = final_task_loader.task_data_loader
        size = final_task_loader.eval_support_set_size
        info_heads = []
        for eval_group in range(self.eval_groups_num):
            support_size = int(size * ((eval_group + 1) / self.eval_groups_num))
            info_head = []
            for info in info_example:
                info_head.append(info + str(support_size))
            info_heads.append(info_head)
        for m in range(len(info_heads[0])):
            for n in range(len(info_heads)):
                row.append(info_heads[n][m])
        csv_writer.writerow(row)

        for j in range(len(self.others_data_loader)):
            self.file_list.append(
                open(os.path.join(self.project_root, 'logDir/{}/{}/{}another{}.csv'.format
                (self.dataset_name, self.base_record_path, j, self.watched_loader_detail_list[j])), 'w',
                     newline='',
                     encoding='utf-8'))
            self.csv_writer_list.append(csv.writer(self.file_list[-1]))
            f = self.file_list[-1]
            csv_writer = self.csv_writer_list[-1]
            example_row = copy.deepcopy(row)
            csv_writer.writerow(example_row)

    def train(self):
        if not os.path.exists(os.path.join(self.project_root, 'logDir/{}'.format(self.dataset_name))):
            os.mkdir(os.path.join(self.project_root, 'logDir/{}'.format(self.dataset_name)))
        if not os.path.exists(
                os.path.join(self.project_root, 'logDir/{}/{}'.format(self.dataset_name, self.base_record_path))):
            os.mkdir(os.path.join(self.project_root, 'logDir/{}/{}'.format(self.dataset_name, self.base_record_path)))

        self.open_log_file()
        # meta learning
        for step in range(self.train_config['learn_config']['train_steps']):
            # eval for every gap
            if step % self.log_gap == 0:
                self.eval(step)
                self.flush_file()
                print('meta_step:', step)
            # save model
            if step % self.save_model_gap == 0:
                torch.save(self.model, os.path.join(self.project_root, 'logDir/{}/{}/model'.
                                                    format(self.dataset_name, self.base_record_path)))

            support, label = self.task_data_loader.online_sample_train_task()
            # clear and write matrix m
            self.model.clear_memory()
            self.model.write(support, label)
            pred_labels, embedding_vec = self.model.forward(support, label.sum())
            loss = self.loss_fn(pred_labels, label, embedding_vec)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
            self.optimizer.step()
            self.model.normalize_a()

    def flush_file(self):
        for file in self.file_list:
            file.flush()

    def cul_dif_a_depth(self):
        matrix_var = self.model.matrix_a.var(dim=0)
        ave_matrix_var = matrix_var.mean().item()
        return round(ave_matrix_var, 4)

    def cul_dif_matrix_depth(self):
        matrix_var = self.model.matrix_m.matrix.var(dim=0)
        ave_matrix_var = matrix_var.mean().item()
        return round(ave_matrix_var, 4)

    def calculate_eval_group_data(self):
        loss_sum = 0
        label_var_sum = 0
        pre_var_sum = 0
        ARE_sum = 0
        eval_times = 0
        AAE_sum = 0
        num_sum = 0
        sparse_mean_sum = 0
        sparse_var_sum = 0
        labels = []
        pre_labels = []
        for support, label in self.task_data_loader.generate_eval_tasks():
            # filter support size
            eval_times += 1
            loss, label_var, pre_var, err, aae, num, pred, label, sparse_mean, sparse_var = self.eval_once(support,label)
            labels.append(label)
            pre_labels.append(pred)
            loss_sum += loss
            label_var_sum += label_var
            pre_var_sum += pre_var
            ARE_sum += err
            AAE_sum += aae
            num_sum += num
            sparse_mean_sum += sparse_mean
            sparse_var_sum += sparse_var
        ave_loss, ave_label_var, ave_pre_var, ave_ARE, ave_AAE, ave_sparse_mean, ave_sparse_var = round(loss_sum / float(eval_times), 1),\
            round(label_var_sum / float(eval_times), 1), round(pre_var_sum / float(eval_times),1), round(ARE_sum / float(num_sum), 5),\
            round(AAE_sum / float(num_sum), 2), round(sparse_mean_sum / float( eval_times),1), round(sparse_var_sum / float(eval_times), 1)
        ave_a_var = self.cul_dif_a_depth()
        ave_matrix_var = self.cul_dif_matrix_depth()
        accuracy_sum = 0
        recall_sum = 0
        precision_sum = 0
        F1_sum = 0
        for i in range(len(labels)):
            label = labels[i].cpu().numpy()
            predict = pre_labels[i].cpu().numpy()
            label = label > np.quantile(label, 0.8)
            predict = predict > np.quantile(predict, 0.8)
            accuracy_sum += metrics.accuracy_score(label, predict)
            recall_sum += metrics.recall_score(label, predict, zero_division=0)
            precision_sum += metrics.precision_score(label, predict, zero_division=0)
            F1_sum += metrics.f1_score(label, predict, zero_division=0)

        accuracy, racall, precision, F1 = round(accuracy_sum / len(labels), 3), round(recall_sum / len(labels), 3),\
            round(precision_sum / len(labels), 3), round(F1_sum / len(labels), 3)
        return ave_loss, ave_label_var, ave_pre_var, ave_ARE, ave_AAE, ave_a_var, ave_matrix_var, accuracy, racall, \
               precision, F1, ave_sparse_mean, ave_sparse_var

    def eval(self, step=None, eval_groups_nums=None):
        if eval_groups_nums is None:
            eval_groups_nums = self.eval_groups_num
        else:
            eval_groups_nums = 1
        self.model.eval()
        file_data_list = []
        for data_loader in ((self.train_task_data_loader,) + self.others_data_loader):
            file_data = []
            self.task_data_loader = data_loader
            final_task_loader = self.task_data_loader
            while final_task_loader.task_data_loader is not None:
                final_task_loader = final_task_loader.task_data_loader
            size = final_task_loader.eval_support_set_size
            file_data.append(step)
            diff_order_datas = []
            for eval_group in range(eval_groups_nums):
                final_task_loader.eval_support_set_size = int(
                    size * ((eval_group + 1) / eval_groups_nums))
                ave_loss, ave_label_var, ave_pre_var, ave_ARE, ave_AAE, ave_a_var, ave_matrix_var, accuracy, racall, precision, F1, ave_sparse_mean \
                    , ave_sparse_var = self.calculate_eval_group_data()
                diff_order_datas.append(
                    [ave_sparse_mean, ave_sparse_var, ave_loss, ave_label_var, ave_pre_var, ave_ARE, ave_AAE, ave_a_var,
                     ave_matrix_var, accuracy,
                     racall, precision, F1])
            support_data = None
            for support, label in self.task_data_loader.generate_eval_tasks():
                support_data = support
                break
            embedding_vec = self.model.input_enc(support_data)
            mean = round(embedding_vec.mean().item(), 5)
            var = round(embedding_vec.var().item(), 5)
            none_zero = round(self.model.get_sparse(embedding_vec).mean().item(), 5)
            file_data.extend([mean, var, none_zero])
            for m in range(len(diff_order_datas[0])):
                for n in range(len(diff_order_datas)):
                    file_data.append(diff_order_datas[n][m])
            file_data_list.append(file_data)
        if self.csv_writer_list is not None:
            for i in range(len(self.csv_writer_list)):
                self.csv_writer_list[i].writerow(file_data_list[i])
        self.model.train()
        self.task_data_loader = self.train_task_data_loader
        return file_data_list

    def eval_once(self, support, label):
        with torch.no_grad():
            self.model.clear_memory()
            self.model.write(support, label)
            pred_labels, embedding_vec = self.model.forward(support, label.sum())
            loss = self.loss_fn(pred_labels, label, embedding_vec)
            diff = torch.abs(pred_labels - label) / label
            diff2 = torch.abs(pred_labels - label)
            q = self.model.input_enc(support)
            a = self.model.getAddress(q)
            sparse_degree = self.model.get_sparse(a)
            sparse_mean = sparse_degree.mean()
            sparse_var = sparse_degree.var()
        return loss.item(), torch.var(label).item(), torch.var(
            pred_labels).item(), diff.sum().item(), diff2.sum().item(), label.shape[
                   0], pred_labels, label, sparse_mean.item(), sparse_var.item()

