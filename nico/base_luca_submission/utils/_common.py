import torch.nn as nn
import torch
import os
from torch.autograd import grad
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.manual_seed(42)

dtype = torch.float
device_type = 'cuda:0' if torch.cuda.is_available() else 'cpu'



class Sin(torch.nn.Module):
    @staticmethod
    def forward(input):
        return torch.sin(input)


class NeuralNet(nn.Module):

    def __init__(self, D_hid=10):
        super(NeuralNet, self).__init__()

        self.actF = Sin()

        self.sym = 1
        self.lambda_ = torch.nn.Linear(1, 1)
        self.Lin_1 = torch.nn.Linear(2, D_hid)
        self.Lin_2 = torch.nn.Linear(D_hid+1, D_hid)
        self.out = torch.nn.Linear(D_hid+1, 1)

    def forward(self, t):
        lambda_ = self.lambda_(torch.ones_like(t))

        L1 = self.Lin_1(torch.cat((t, lambda_), 1))
        L1p = self.Lin_1(torch.cat((-t, lambda_), 1))

        h1 = self.actF(L1)
        h1p = self.actF(L1p)

        L2 = self.Lin_2(torch.cat((h1, lambda_), 1))
        L2p = self.Lin_2(torch.cat((h1p, lambda_), 1))

        h2 = self.actF(L2)
        h2p = self.actF(L2p)

        out = self.out(torch.cat((h2+ self.sym * h2p, lambda_), 1))

        return out, lambda_

    def init_weights(self):
        def help(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.00)
        self.apply(help)

#########################################################################à

class OptimizationComplete(Exception):
    pass