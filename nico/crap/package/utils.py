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


# function computing the solution from x and the neural net output
def para_f(x : torch.Tensor, N : torch.Tensor, L : float) -> torch.Tensor:
    """computes parametized solution form input points x and nonparametrized values N

    Args:
        x (torch.Tensor): input points
        N (torch.Tensor): nonparametrized solution 
        L (float): endpoint

    Returns:
        torch.Tensor: parametrized solution
    """
    assert(x.shape == N.shape)
    g = (1 - torch.exp(-x)) * (1 - torch.exp(-(x - L)))
        
    return N * g

def parametricSolutions(t, nn, t0, tf, x1):
    N1, _ = nn(t)
    f = (1-torch.exp(-(t-t0)))*(1-torch.exp(-(t-tf)))
    psi_hat = x1 + f*N1
    return psi_hat

def pde_residual(input: torch.Tensor, f: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """Compute residual

    Args:
        input (torch.Tensor): input points
        f (torch.Tensor): function values at input
        lam (torch.tensor): containting the eigenvalue

    Returns:
        torch.Tensor: residual values
    """
    assert(f.shape == input.shape)
    
    grad_x = torch.autograd.grad(f.sum(), input, create_graph=True)[0].reshape(-1,1)
    grad_xx = torch.autograd.grad(grad_x.sum(), input, create_graph=True)[0].reshape(-1,1)

    assert(f.shape == grad_x.shape)

    pde_residual = grad_xx + (lam.pow(2)) * f
    return  pde_residual.reshape(-1,1), grad_xx