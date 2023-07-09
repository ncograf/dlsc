import torch
from torch import nn

def weights_init(m):
    if isinstance(m, nn.Linear) and m.weight.shape[0] != 1:
        torch.nn.init.xavier_uniform(m.weight.data)


# Define the sin() activation function
class mySin(torch.nn.Module):
    @staticmethod
    def forward(input):
        return torch.sin(input)

class qNN1(torch.nn.Module):
    def __init__(self, D_hid=10):
        super(qNN1, self).__init__()

        # Define the Activation
        #  self.actF = torch.nn.Sigmoid()
        self.actF = mySin()

        # define layers
        # self.Lin_1   = torch.nn.Linear(1, D_hid)
        # self.E_out = torch.nn.Linear(D_hid, 1)
        # self.Lin_2 = torch.nn.Linear(D_hid, D_hid)
        # self.Ein = torch.nn.Linear(1,1)
        # self.Lin_out = torch.nn.Linear(D_hid+1, 1)
        self.sym = False
        self.Ein = torch.nn.Linear(1, 1)
        self.Lin_1 = torch.nn.Linear(2, D_hid)
        self.Lin_2 = torch.nn.Linear(D_hid+1, D_hid)
        self.out = torch.nn.Linear(D_hid+1, 1)

    def forward(self, t):
        In1 = self.Ein(torch.ones_like(t))

        L1 = self.Lin_1(torch.cat((t, In1), 1))
        L1p = self.Lin_1(torch.cat((-1*t, In1), 1))

        h1 = self.actF(L1)
        h1p = self.actF(L1p)

        L2 = self.Lin_2(torch.cat((h1, In1), 1))
        L2p = self.Lin_2(torch.cat((h1p, In1), 1))

        h2 = self.actF(L2)
        h2p = self.actF(L2p)

        # out = self.out(torch.cat((h2+h2p,In1),1))
        # out = self.out(torch.cat((h2,In1),1))

        if self.sym:
            out = self.out(torch.cat((h2+h2p, In1), 1))
        else:
            out = self.out(torch.cat((h2-h2p, In1), 1))

        return out, In1
