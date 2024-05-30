import torchvision

from torchvision import datasets
from torch.utils.data import DataLoader

import torch.nn as nn
import torch
from modules import  resnet, network, contrastive_loss
from modules.utils import acc, eval,prototypes
import numpy as np

import copy
from torch.nn.functional import normalize
import torch.nn.functional as F
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group




os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"





def ddp_setup(rank: int, world_size: int):
    """
 +   Args:
 +       rank: Unique identifier of each process
 +      world_size: Total number of processes
 +   """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12545"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)




def main(rank, world_size, train_set, av_train, test_set, av_test,hyper_params):

    # initializing DDP
    ddp_setup(rank, world_size)
    torch.cuda.set_device(rank)

    # initilaizing Teacher Network denoted as T(.) in the main manuscript
    res = resnet.get_resnet(hyper_params["my_resnet"])
    teacher = network.Network(res, hyper_params["feature_dim"])
    teacher = teacher.to(rank)
    teacher = DDP(teacher, device_ids=[rank])
    optimizer_teacher = torch.optim.Adam(teacher.parameters(), lr=0.0003)

    cluster_head = [] # The last layer of C_t denoted as C_t^{last} in the main manuscript
    students = [] # Student networks denoted as S_r(.) in the main manuscript
    predictors = [] # Predictors denoted as g_r(.) in the main manuscript
    optimizer_pre = [] # Optimizers of Predictors
    instance_projector = [] # Instance-level Projectors denoted as I_r(.) in the main manuscript

    task_id = 0

    for _ in av_train.task_pattern_indices:
        # --------------- Init Task ---------------

        current_train_dataset = torch.utils.data.Subset(train_set, av_train.task_pattern_indices[task_id])
        data_loader = torch.utils.data.DataLoader(
            current_train_dataset,
            batch_size=hyper_params["batch_size"],
            shuffle=False,
            sampler=DistributedSampler(current_train_dataset)
            , num_workers=4,
            drop_last=True,
            pin_memory=True

        )

        #####################################################################################
        # Defining Students
        sq_net = torchvision.models.squeezenet1_1()
        new_net = network.student_network(sq_net, res.rep_dim).to(rank)
        new_net = DDP(new_net, device_ids=[rank])
        students.append(new_net)
        optimizer_stu = torch.optim.Adam(students[-1].parameters())

        # Defining the last layer of cluster-projector
        cluster_head.append(DDP(nn.Linear(res.rep_dim, hyper_params["nb_new_classes"][task_id]).to(rank), device_ids=[rank]))
        optimizer_c = torch.optim.Adam(cluster_head[-1].parameters(), lr=0.0003)

        instance_projector.append(nn.Sequential(
            nn.Linear(res.rep_dim, res.rep_dim),
            nn.ReLU(),
            nn.Linear(res.rep_dim, hyper_params["feature_dim"]),
        ).to(rank))
        optimizer_ins = torch.optim.Adam(instance_projector[-1].parameters(), lr=0.0003)


        if task_id > 0:
            # Defining Predictors when task_id is greater than zero
            predictors.append(DDP(nn.Sequential(
                nn.Linear(hyper_params["feature_dim"], hyper_params["proj_hidden_dim"]),
                nn.SyncBatchNorm(hyper_params["proj_hidden_dim"]),
                nn.ReLU(),
                nn.Linear(hyper_params["proj_hidden_dim"], hyper_params["feature_dim"]),
            ).to(rank), device_ids=[rank]))

            optimizer_pre.append(torch.optim.Adam(predictors[-1].parameters(), lr=0.0003))

            for i in range(task_id):
                checkpoint = torch.load('model_2_2_task_id_{}_CIFAR_3_100'.format(i),
                                        map_location=torch.device('cuda'))
                students[i].load_state_dict(checkpoint["students"])
            students[-1].load_state_dict(checkpoint["students"])
            if task_id > 1:
                predictors[-2].load_state_dict(checkpoint["predictors"])
                optimizer_pre[-2].load_state_dict(checkpoint["optimizer_pre"])

            optimizer_teacher = torch.optim.Adam(teacher.parameters(), lr=0.0001)

        # Defining the loss
        loss_instance = contrastive_loss.InstanceLoss(hyper_params["batch_size"], rank).to(rank)
        loss_cluster = contrastive_loss.ClusterLoss(hyper_params["nb_new_classes"][task_id], rank).to(rank)
        
        if task_id > 0:
            hyper_params["training_epoch"] = 400
        
        for epoch in range(hyper_params["training_epoch"]):
            print("epoch", epoch)
        
            if epoch % 50 == 49 and rank == 0:
        
                eval(rank, task_id, teacher, students, cluster_head, epoch, test_set, av_test) # Evaluating our model on test data
                if task_id > 0: # Saving Our model
                    state = {'teacher': teacher.state_dict(), "optimizer_teacher": optimizer_teacher.state_dict(),
                             'cluster_head': cluster_head[-1].state_dict(),
                             'optimizer_c': optimizer_c.state_dict(), 'students': students[-1].state_dict(),
                             'optimizer_stu': optimizer_stu.state_dict(), 'predictors': predictors[-1].state_dict(),
                             "optimizer_pre": optimizer_pre[-1].state_dict(),
                             "instance_projector": instance_projector[-1].state_dict(),
                             "optimizer_ins": optimizer_ins.state_dict(), "epoch": epoch}
                    with open('model_2_2_task_id_{}_CIFAR_3_100'.format(task_id), 'wb') as out:
                        torch.save(state, out)
                else:
                    state = {'teacher': teacher.state_dict(), "optimizer_teacher": optimizer_teacher.state_dict(),
                             'cluster_head': cluster_head[-1].state_dict(),
                             'optimizer_c': optimizer_c.state_dict(), 'students': students[-1].state_dict(),
                             'optimizer_stu': optimizer_stu.state_dict(),
                             "instance_projector": instance_projector[-1].state_dict(),
                             "optimizer_ins": optimizer_ins.state_dict(), "epoch": epoch}
        
                    with open('model_2_2_task_id_{}_CIFAR_3_100'.format(task_id), 'wb') as out:
                        torch.save(state, out)
        
            for step, ((x_i, x_j), _) in enumerate(data_loader):
                ################################ Forward Knowledge Distillation ################################
                # Unfreeze Parameters of Teacher
                teacher.train()
                for param in teacher.parameters():
                    param.requires_grad = True

                # Freeze Parameters of Students
                for ii in range(len(students)):
                    students[ii].eval()
                    for param in students[ii].parameters():
                        param.requires_grad = False
        
                x_i = x_i.to(rank)
                x_j = x_j.to(rank)
        
                optimizer_teacher.zero_grad()
                for ii in range(len(predictors)):
                    optimizer_pre[ii].zero_grad()
                optimizer_c.zero_grad()
                optimizer_ins.zero_grad()
        
                c_i, c_j, h_i, h_j = teacher(x_i, x_j)
        
                z_i = normalize(instance_projector[-1](h_i), dim=1)
                z_j = normalize(instance_projector[-1](h_j), dim=1)
        
                c_i = cluster_head[-1](c_i)
                c_j = cluster_head[-1](c_j)
        
                c_i = nn.Softmax(dim=1)(c_i)
                c_j = nn.Softmax(dim=1)(c_j)
                # Calculating L_{con} + L_{clu}
                if task_id == 0:
                    final_loss = loss_instance(z_i, z_j) + loss_cluster(c_i, c_j)
                else:
                    final_loss = loss_instance(z_i, z_j, p) + loss_cluster(c_i, c_j)
        
                flag_cnt = 0
                # Calculating L_{dis}
                for ii in range(len(students) - 2, -1, -1):
                    if flag_cnt >= hyper_params["M"]:
                        break
                    flag_cnt += 1
        
                    z_i_hat = predictors[ii](z_i)
                    z_j_hat = predictors[ii](z_j)
        
                    _, h_i_hat = students[ii](x_i)
                    _, h_j_hat = students[ii](x_j)
        
                    prev_z_i = instance_projector[ii](h_i_hat)
                    prev_z_j = instance_projector[ii](h_j_hat)
        
                    final_loss += (loss_instance(z_i_hat, prev_z_i.detach()) + loss_instance(z_j_hat,
                                                                                             prev_z_j.detach())) / (len(students) * 2)
        
                final_loss.mean().backward()

                # updating parameters of Teacher, Predictors, cluster-projector, and $l_t$-th instance-level projector
                optimizer_teacher.step()
                for i in range(len(predictors)):
                    optimizer_pre[i].step()
                optimizer_c.step()
                optimizer_ins.step()

                ################################ Backward Knowledge Distillation ################################
                # Freezing Parameters of Teacher
                teacher.eval()
                for param in teacher.parameters():
                    param.requires_grad = False

                # Unfreeze Parameters of $l_t$-th student
                students[-1].train()
                for param in students[-1].parameters():
                    param.requires_grad = True
        
                optimizer_stu.zero_grad()
        
                _, h_i_hat = students[-1](x_i)
                _, h_j_hat = students[-1](x_j)
                prev_z_i = normalize(instance_projector[-1](h_i_hat), dim=1)
                prev_z_j = normalize(instance_projector[-1](h_j_hat), dim=1)

                #Calculating L_{stu}
                loss_f = loss_instance(prev_z_i, z_i.detach()) + loss_instance(prev_z_j, z_j.detach()) + nn.MSELoss()(
                    h_i_hat, h_i.detach()) + nn.MSELoss()(h_j_hat, h_j.detach())
        
                loss_f.mean().backward()
                # Updating Parameters of $l_t$-th student
                optimizer_stu.step()
        
        # Updating Prototype Set P_{t-1}
        if task_id == 0:
            p = prototypes(hyper_params, teacher, instance_projector, cluster_head, task_id, rank, data_loader)
        else:
            p = prototypes(hyper_params,teacher,instance_projector,cluster_head,task_id,rank,data_loader,p=p)

        task_id += 1


if __name__ == "__main__":
    # Loading the Dataset
    states = torch.load("train_test_split_CIFAR100_10_task")
    train_set = states["train_set"]
    av_train = states["av_train"]
    test_set = states["test_set"]
    av_test = states["av_test"]

    # Hyperparameters
    hyper_params = {
    "batch_size": 256,
    "my_resnet" :"ResNet18",
    "feature_dim": 128,
    "training_epoch": 1000,
    "proj_hidden_dim":512,
    "nb_new_classes": [10]*10,
    "M": 5}

    world_size = torch.cuda.device_count()
    device = torch.device("cuda")

    mp.spawn(main, args=(world_size, train_set, av_train, test_set, av_test,hyper_params,), nprocs=world_size)
