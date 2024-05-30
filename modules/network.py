import torch.nn as nn
import torch
from torch.nn.functional import normalize


class Network(nn.Module):
    def __init__(self, resnet, feature_dim):
        super(Network, self).__init__()
        self.resnet = resnet
        self.feature_dim = feature_dim
        # self.cluster_num = class_num
        """self.instance_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, self.feature_dim),
        )"""
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.ReLU(),
            # nn.Linear(self.resnet.rep_dim, self.cluster_num),
            # nn.Softmax(dim=1)
        )

    def forward(self, x_i, x_j):
        h_i = self.resnet(x_i)
        h_j = self.resnet(x_j)

        

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)

        return c_i, c_j, h_i, h_j

    def forward_cluster(self, x):
        h = self.resnet(x)
        c = self.cluster_projector(h)
        # c = torch.argmax(c, dim=1)
        return c

class student_network(nn.Module):

    def __init__(self, sq, res_dim):
        super(student_network, self).__init__()

        self.sq = sq
        self.final_layer_1 = nn.Linear(1000, res_dim)


    def forward(self, x):
        h = self.sq(x)



        h = self.final_layer_1(h)


        return x, h