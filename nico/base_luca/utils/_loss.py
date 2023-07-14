import torch
import numpy as np


def parametric_solutions(self, t, N1: torch.Tensor, t0, tf, x1):
    f = (1-torch.exp(-(t-t0)))*(1-torch.exp(-(tf - t)))
    psi_hat = x1 + f * N1
    return psi_hat

def dfx(self, x, f):
    gopts = torch.ones(x.shape, dtype=self.dtype)
    if self.device_type == 'cuda:0':
        gopts = gopts.to(device = torch.device('cuda:0'))
    return torch.autograd.grad([f], [x], grad_outputs=gopts, create_graph=True)[0]

def potential(self, t):
    Xsnp = t.data.numpy()
    Vnp = (np.heaviside(-Xsnp-np.pi/2, 0) + np.heaviside(Xsnp-np.pi/2, 0))*20
    Vtorch = torch.from_numpy(Vnp)
    return Vtorch

def compute_pde_residual(self, input, psi, E, V):

    psi_dx = self.dfx(input, psi)
    psi_ddx = self.dfx(input, psi_dx)
    f = psi_ddx/2 + (E - V)*psi
    loss = torch.mean(abs(f)**2)

    return loss

def compute_loss(self, inputs, w_pde, w_norm):

    inputs.requires_grad = True

    pot_n = self.potential(inputs)

    n1, lambda_n = self.network(inputs)

    pred_u = self.parametric_solutions(t=inputs, N1=n1, t0=self.x0, tf=self.xf, x1=0)

    loss_pde = self.compute_pde_residual(inputs, pred_u, lambda_n, pot_n)*w_pde[0]

    loss_norm = (torch.dot(pred_u[:, 0], pred_u[:, 0]).sqrt() - self.n_samples/(self.xf-self.x0)).pow(2) * w_norm[0]

    return loss_pde, loss_norm, n1, lambda_n, pred_u

def compute_loss_orth(self, inputs, pred_u, w_orth):
    
    loss_orth = torch.tensor([0])
    orth_sol = torch.zeros_like(pred_u.flatten())
    if self.orth_counter[0] > 0:
        for i in range(self.orth_counter[0]):
            orth_sol += self.parametric_solutions(inputs, self.dic[i][0](inputs)[0], self.x0, self.xf, 0).flatten()
        loss_orth = torch.sqrt(torch.dot(orth_sol, pred_u.flatten()).pow(2)) * w_orth[0]
    
    return loss_orth