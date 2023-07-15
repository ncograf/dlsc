import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from utils._paths import *

def save_plots(self, loss_history: list, pde_hist: list, norm_hist: list, orth_hist: list, lambda_hist: list):
    plt.close()
    plt.cla()
    plt.clf()
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    axs[0, 0].semilogy(loss_history)
    axs[0, 0].set_title('Total Loss')
    axs[0, 0].grid(True)

    axs[0, 1].semilogy(pde_hist)
    axs[0, 1].set_title('PDE Loss')
    axs[0, 1].grid(True)

    axs[0, 2].semilogy(norm_hist)
    axs[0, 2].set_title('Norm Loss')
    axs[0, 2].grid(True)

    axs[1, 0].semilogy(orth_hist)
    axs[1, 0].set_title('Orth Loss')
    axs[1, 0].grid(True)

    axs[1, 1].plot(lambda_hist)
    axs[1, 1].set_title('Lambda')
    axs[1, 1].grid(True)

    fig.tight_layout()
    fig.savefig(all_path)
    return

def save_solution(self, dic: dict, x_samples: torch.Tensor,  index: int):
    print(f'############# Save Solution_{index} ################')
    n1, lambda_ = dic[index][0](x_samples)
    pred_u = self.parametric_solutions(x_samples, n1, self.x0, self.xf, 0)
    print(f'Check Normalization: ', torch.sum(pred_u ** 2))

    plt.close()
    plt.cla()
    plt.clf()
    plt.scatter(x_samples.detach(), pred_u.detach(), label=f'E = {np.round(lambda_[0].item(),4)} ', s=2)
    plt.legend()
    plt.savefig(os.path.join(plot_dir, f'solution_{index}.png'))
    
    
def plot_all(self):
    print(f'######### plot all solutions #############')
    plt.close()
    plt.cla()
    fig, axs = plt.subplots(1, 4, figsize=(25, 6))
    t = torch.linspace(self.x0, self.xf, 3000).reshape(-1,1)
    self.load_states(True, True, True, True)
    for i in range(4):
        n1, lambda_ = self.dic[i][0](t)
        E =  np.round(lambda_[0].item(),4)
        f = self.parametric_solutions(t, n1, self.x0, self.xf, 0)
        axs[i].plot(t.flatten().detach(), f.flatten().detach(), label=f'pred E = {E}')
        axs[i].legend(loc=1)
        
    fig.savefig(all_solutions_path)