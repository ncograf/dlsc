import torch
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


def parametricSolutions(t, nn, t0, tf, x1):
    N1, N2 = nn(t)
    dt = t-t0
    f = (1-torch.exp(-(t-t0)))*(1-torch.exp(-(t-tf)))
    psi_hat = x1 + f*N1
    return psi_hat


def hamEqs_Loss(t, psi, E):

    psi_dx = dfx(t, psi)
    psi_ddx = dfx(t, psi_dx)
    f = psi_ddx + (E**2)*psi
    L = (f.pow(2)).mean()
    return L, f, psi_ddx
