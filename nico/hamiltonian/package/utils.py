import torch
import numpy as np
from torch.autograd import grad

dtype = torch.float
device_type = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# Define some more general functions


def dfx(x, f):
    # Calculate the derivative with auto-differention
    gopts = torch.ones(x.shape, dtype=dtype)
    if device_type == 'cuda:0':
        gopts = gopts.to(device = torch.device('cuda:0'))
    return grad([f], [x], grad_outputs=gopts, create_graph=True)[0]


def perturbPoints(grid, t0, tf, sig=0.5):
    # stochastic perturbation of the evaluation points
    # force t[0]=t0  & force points to be in the t-interval
    delta_t = grid[1] - grid[0]
    noise = delta_t * torch.randn_like(grid)*sig
    t = grid + noise
    t.data[2] = torch.ones(1, 1)*(-1)
    t.data[t < t0] = t0 - t.data[t < t0]
    t.data[t > tf] = 2*tf - t.data[t > tf]
    t.data[0] = torch.ones(1, 1)*t0

    t.data[-1] = torch.ones(1, 1)*tf
    t.requires_grad = False
    return t


def parametric_solutions(t, N1: torch.Tensor, t0, tf, x1):
    f = (1-torch.exp(-(t-t0)))*(1-torch.exp(-(t-tf)))
    psi_hat = x1 + f*N1
    return psi_hat

def potential(t : torch.Tensor, l : float, V0 : float):
    t.requires_grad = False
    high = torch.ones_like(t) * V0
    low = torch.zeros_like(t)
    condition = torch.logical_and(t < l, t > -l)
    out = torch.where(condition, low, high)
    return out

def pde_loss(t : torch.Tensor, psi : torch.tensor, E):
    
    V0 = 20
    l = np.pi / 2

    psi_dx = dfx(t, psi)
    psi_ddx = dfx(t, psi_dx)
    V = potential(t, l, V0)
    f = (psi_ddx/2.)+(E-V)*psi
    L = f.pow(2).mean()
    
    return L


class OptimizationComplete(Exception):
    pass

class OptimizationPlato(Exception):
    pass