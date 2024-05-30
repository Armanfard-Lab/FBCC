
import avalanche.benchmarks.classic as benchmarks
import torchvision
import os
from torch.utils.data import  ConcatDataset
from torchvision import datasets
from torch.utils.data import DataLoader

import torch.nn as nn
import torch
from modules import transform

import copy
from torch.nn.functional import normalize
import torch.nn.functional as F
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.utils import make_classification_dataset




def CIFAR_CL_loader(image_size):
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform.Transforms(size=image_size, s=0.5))

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform.Transforms(size=image_size, s=0.5))

    concat_data = ConcatDataset([trainset, testset]) # concat train and test

    # defining
    class_label = []
    for i in range(10):
        class_label.append(list(range(i*10,(i+1)*10)))
    label_tag = []
    for _, y in concat_data:
        for j in range(len(class_label)):
            if y in class_label[j]:
                label_tag.append(j)
                break
    av_train = make_classification_dataset(concat_data, task_labels=label_tag)



    ######################################################################################################################## test
    trainset_test = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                 download=True,
                                                 transform=transform.Transforms(size=image_size).test_transform)

    testset_test = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                download=True,
                                                transform=transform.Transforms(size=image_size).test_transform)

    concat_data_test = ConcatDataset([trainset_test, testset_test])



    av_test = make_classification_dataset(concat_data_test, task_labels=label_tag)

    return  concat_data, av_train, concat_data_test,av_test

if __name__ == "__main__":
    image_size = 32

    device = torch.device("cuda:0")

    train_set, av_train, test_set, av_test = CIFAR_CL_loader(image_size)

    state = {'train_set': train_set, "av_train": av_train, 'test_set': test_set, 'av_test': av_test}
    with open('train_test_split_CIFAR100_10_task', 'wb') as out:
        torch.save(state, out)



