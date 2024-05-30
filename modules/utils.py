import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np
import copy
from torch.nn.functional import normalize
import torch.nn.functional as F

def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment

    ind = linear_sum_assignment(w.max() - w)
    accuracy = 0.0
    for i in ind[0]:
        accuracy = accuracy + w[i, ind[1][i]]
    return accuracy / y_pred.size


def eval(rank, task_id, model, estimators, cluster_head, epoch, test_set, av_test):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    for ii in range(len(estimators)):
        estimators[ii].eval()
        for param in estimators[ii].parameters():
            param.requires_grad = False
    # test_task_id = 0

    for test_task_id in range(task_id + 1):

        current_test_dataset = torch.utils.data.Subset(test_set, av_test.task_pattern_indices[test_task_id])

        data_test_loader = torch.utils.data.DataLoader(
            current_test_dataset,
            batch_size=100,
            shuffle=False,  # Important to set false when using a sampler.
            drop_last=True,

        )

        test_data_length = len(current_test_dataset)
        pred_label = torch.zeros((test_data_length))
        true_label = torch.zeros((test_data_length))
        pred_label_est = torch.zeros((test_data_length))

        task_cnt_test = 0

        for (x, y) in data_test_loader:
            x = x.to(rank)
            c = model.module.forward_cluster(x)
            c = cluster_head[test_task_id](c)
            c = torch.argmax(c, dim=1)

            _, h = estimators[test_task_id](x)
            h = model.module.cluster_projector(h)
            h = cluster_head[test_task_id](h)
            h = torch.argmax(h, dim=1)

            pred_label[100 * task_cnt_test:(task_cnt_test + 1) * 100] = c
            true_label[100 * task_cnt_test:(task_cnt_test + 1) * 100] = y
            pred_label_est[100 * task_cnt_test:(task_cnt_test + 1) * 100] = h
            task_cnt_test += 1

        pred_label = pred_label.cpu().numpy()
        true_label = true_label.cpu().numpy()
        pred_label_est = pred_label_est.cpu().numpy()

        class_labels = np.unique(true_label)
        cnt = 0
        new_class = np.zeros(true_label.shape)
        for current_class in class_labels:
            new_class = np.where(true_label == current_class, cnt, new_class)
            cnt += 1

        with open('results.txt', 'a') as f:
            f.write("epoch: {}\n".format(epoch))
            f.write("task: {}\n".format(task_id))
            
            f.write("Teacher acc: {}\n".format(acc(new_class, pred_label)))
            f.write("Student acc: {}\n".format(acc(new_class, pred_label_est)))
          
            f.write("######")

    with open('results.txt', 'a') as f:
        f.write(
            "########################################################################################################################\n")


def prototypes(hyper_params,teacher,instance_projector,cluster_head,task_id,rank,data_loader,p=None):
    new_p = torch.zeros(hyper_params["nb_new_classes"][task_id], hyper_params["feature_dim"]).to(rank)
    p_cnt = torch.zeros(hyper_params["nb_new_classes"][task_id]).to(rank)
    for step, ((x_i, x_j), _) in enumerate(data_loader):

        x_i = x_i.to(rank)
        x_j = x_j.to(rank)

        c_i, c_j, h_i, h_j = teacher(x_i, x_j)

        z_i = normalize(instance_projector[-1](h_i), dim=1)
        z_j = normalize(instance_projector[-1](h_j), dim=1)

        c_i = cluster_head[-1](c_i)
        c_j = cluster_head[-1](c_j)

        c_i = torch.argmax(c_i, dim=1)
        c_j = torch.argmax(c_j, dim=1)

        c = torch.where(c_i == c_j, c_i, -torch.ones(c_i.shape).to(rank))
        c = c.type(torch.int64)

        for my_i in range(hyper_params["batch_size"]):
            if c[my_i] != -1:
                new_p[c[my_i]] = (new_p[c[my_i]] * p_cnt[c[my_i]] + (z_i[my_i] + z_j[my_i])) / (p_cnt[c[my_i]] + 2)
                p_cnt[c[my_i]] += 2

    if task_id == 0:
        p = new_p.clone()

    else:
        p = torch.cat((p, new_p), dim=0)

    return normalize(p, dim=1)