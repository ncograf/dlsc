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
    Vnp = (np.heaviside(-Xsnp-np.pi, 0) + np.heaviside(Xsnp-np.pi, 0))*20
    Vtorch = torch.from_numpy(Vnp)
    return Vtorch

def compute_pde_residual(self, input, psi, E, V):

    assert(input.shape == psi.shape)
    psi_dx = self.dfx(input, psi)
    assert(input.shape == psi_dx.shape)
    psi_ddx = self.dfx(input, psi_dx)
    assert(input.shape == psi_ddx.shape)
    f = psi_ddx/2 + (E - V)*psi
    loss = torch.mean(abs(f)**2)

    return loss

def compute_loss(self, inputs, w_pde, w_norm, verbose = True):

    inputs.requires_grad = True

    pot_n = self.potential(inputs)

    n1, lambda_n = self.network(inputs)

    pred_u = self.parametric_solutions(t=inputs, N1=n1, t0=self.x0, tf=self.xf, x1=0)

    loss_pde = self.compute_pde_residual(inputs, pred_u, lambda_n, pot_n)*w_pde[0]

    loss_norm = (torch.dot(pred_u[:, 0], pred_u[:, 0]).sqrt() - self.n_samples/(self.xf-self.x0)).pow(2) * w_norm[0]

    loss_drive = torch.exp(-lambda_n + 0)

    loss = loss_pde + loss_norm

    """
    loss = torch.log10( 0.3*loss_int  + loss_meas )
    if verbose: print("Total loss: ", round(loss.item(), 4), "| Loss_meas: ", round(torch.log10(loss_meas).item(), 4), "| Loss_int: ", round(torch.log10(loss_int).item(), 4), "| Loss_sb_Tf: ", round(torch.log10(loss_sb_Tf).item(), 4), "| Loss_sb_Ts: ", round(torch.log10(loss_sb_Ts).item(), 4), "| Loss_tb: ", round(torch.log10(loss_tb).item(), 4))
    self.iteration = self.iteration +1
    """

    return loss, loss_pde, loss_norm, n1, lambda_n, pred_u

def compute_loss_orth(self, inputs, pred_u, w_orth):
    
    loss_orth = torch.tensor([0])

    if self.orth_counter[0] == 1:
    
        par1 = self.parametric_solutions(inputs, self.dic[0][0](inputs)[0], self.x0, self.xf, 0)

        loss_orth = torch.sqrt(torch.dot(par1[:,0], pred_u[:,0]).pow(2)) * w_orth[0]
                    

    elif self.orth_counter[0] == 2:
    
        par1 = self.parametric_solutions(inputs, self.dic[1][0](inputs)[0], self.x0, self.xf, 0)
        par2 = self.parametric_solutions(inputs, self.dic[2][0](inputs)[0], self.x0, self.xf, 0)

        loss_orth = torch.sqrt(torch.dot(par1[:,0] + par2[:,0], pred_u[:,0]).pow(2)) * w_orth[0]

        loss_tot += loss_orth

    elif self.orth_counter[0] == 3:
    
        par1 = self.parametric_solutions(inputs, self.dic[1][0](inputs)[0], self.x0, self.xf, 0)
        par2 = self.parametric_solutions(inputs, self.dic[4][0](inputs)[0], self.x0, self.xf, 0)
        par3 = self.parametric_solutions(inputs, self.dic[2][0](inputs)[0], self.x0, self.xf, 0)

        loss_orth = torch.sqrt(torch.dot(par1[:,0] + par2[:,0] + par3[:, 0], pred_u[:,0]).pow(2)) * w_orth[0]

    return loss_orth