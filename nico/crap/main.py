import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import copy
import random
import json

from package.modules import *
from package.utils import *
from package.directories import *

intra_op_t = torch.get_num_interop_threads()
inter_op_t = torch.get_num_threads()

type_ = torch.float32

print(f"Intra op threds: {intra_op_t}\nInter op threds: {inter_op_t}")

seed = 120
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# for the plots
plt.rc('xtick', labelsize=16)
plt.rcParams.update({'font.size': 16})

def plot_solutions(dic: dict, L: float, nTest: int):
    plt.close();plt.cla();
    tTest = torch.linspace(0,L,nTest)
    tTest = tTest.reshape(-1,1);
    tTest.requires_grad=True
    for bin in dic.keys():
        if bin and dic[bin][0]:
            N, _ = dic[bin][0].forward(tTest)
            f = para_f(tTest, N, L)
            plt.plot(tTest.detach().numpy(), f.detach().numpy(), label=f'$\lambda$={bin}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(solutions_path)

def save_plot(item: list, xlabel: str, ylabel: str, path: str, title: str='', semilogy: int=False):
    plt.close() 
    if semilogy:
        plt.semilogy(item,'-r',alpha=0.975)
    else:
        plt.plot(item)
    plt.ylabel(ylabel);plt.xlabel(xlabel);
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path)

def run_Scan_finitewell(t0, L, neurons, first_epochs, epochs, n_points, lr):

    nn = NeuralNet(1, neurons)
    fc1 = 0
    betas = [0.999, 0.9999]
    optimizer = optim.Adam(nn.parameters(), lr=lr, betas=betas)
    best_criteria = 1e+10

    Loss_history = []
    nontriv_loss_history = []
    lam_history = [0]
    orth_losses = []
    rm_histroy = []
    internal_SE_loss = [1] # initial value to avoid index out of range

    dic = { i:(None, 1e+20) for i in range(8)}

    orth_counter = 0

    grid = torch.linspace(t0, L, n_points).reshape(-1, 1)

    # TRAINING ITERATION
    TeP0 = time.time()
    for tt in range(epochs):
        
        # permute input
        t = perturbPoints(grid, t0, L, sig=.03*L)
        idx = np.random.permutation(n_points)
        input = t[idx]
        input.requires_grad = True

        # compute apporximation
        N, lam = nn.forward(input)
        f = para_f(input, N, L)
        lam = torch.abs(lam[0])

        # PDE loss
        r_pde, H_psi = pde_residual(input, f, lam)
        l_pde = r_pde.pow(2).mean()
        loss = l_pde
        criteria_loss = l_pde

        # Normalization Loss
        l_norm = (torch.dot(lam * f.flatten(),lam * f.flatten()).sqrt() - n_points / L).pow(2)
        loss += l_norm

        window = 1000
        rm = 1
        if len(internal_SE_loss) >= window+1:
            rm = np.mean(
                np.array(internal_SE_loss[-window:])-np.array(internal_SE_loss[-window-1:-1]))
        else:
            rm = np.mean(
                np.array(internal_SE_loss[1:])-np.array(internal_SE_loss[:-1]))
        
        if rm < -0.01:
            rm = -0.01
        elif rm > 0.01:
            rm = 0.01


        if tt % 1500 == 0:
            print('\nEpoch', tt)
            print('PDE Loss', internal_SE_loss[-1])
            print('E', lam_history[-1])
            print('rm', rm)
            print('oc', orth_counter)

            plt.close()
            fig, axs = plt.subplots(2,3, figsize=(12,6))
            axs[0,0].plot(lam_history)
            axs[0,0].set_title('Lambda n')
            axs[0,1].semilogy(internal_SE_loss)
            axs[0,1].set_title('PDE Loss')
            axs[0,2].semilogy(nontriv_loss_history)
            axs[0, 2].set_title('NonTriv Loss')
            axs[1,0].semilogy(orth_losses)
            axs[1,0].set_title('Orth Loss')
            axs[1,1].plot(rm_histroy)
            axs[1,1].set_title('rm')
            fig.tight_layout()
            fig.savefig(all_path)
    

        if tt % 3000 == 0:
            plot_solutions(dic, L, 500)

        exp_thresh = -10
        if tt == first_epochs:
            nn.apply(weights_init)
            orth_counter += 1
            print('\nEpoch', tt)
            print('E', lam_history[-1])
            print('rm', rm)
            print('oc', orth_counter)
        elif rm < np.exp(exp_thresh) and rm > 0 and orth_counter > 0:
            if orth_counter in [1]:
                nn.apply(weights_init)
            if orth_counter in [3]:
                nn.sym *= 1
            orth_counter += 1
            print('\nEpoch', tt)
            print('E', lam_history[-1])
            print('rm', rm)
            print('oc', orth_counter)
        elif rm < np.exp(exp_thresh) and rm > 0 and orth_counter == 5:
            TePf = time.time()
            runTime = TePf - TeP0
            loss_histories = (Loss_history, nontriv_loss_history, internal_SE_loss,
                              nn, lam_history, dic, orth_losses, rm_histroy)
            return fc1, loss_histories, runTime, nn

        orth_idxs = []
        if orth_counter == 1:
            orth_idxs = [1]
        elif orth_counter == 2:
            orth_idxs = [1,2]
        elif orth_counter >= 3:
            orth_idxs = [1,2]

        psi_eigen = torch.zeros_like(input).type(type_)
        for i in orth_idxs:
            if dic[i][0] != None:
                nn_orth = dic[i][0]
                N_orth, _ = nn_orth.forward(input)
                f_orth = para_f(input, N_orth, L)
                psi_eigen += f_orth

        if len(orth_idxs) > 0:
            ortho_loss =  0.02 * torch.dot(psi_eigen.flatten(),f.flatten()).pow(2).sqrt()
            orth_losses.append(ortho_loss.item())
            loss += ortho_loss


        # logging
        nontriv_loss_history.append(l_norm.item())
        rm_histroy.append(rm)
        lam_history.append(lam.item())
        internal_SE_loss.append(l_pde.item())
        Loss_history.append(loss.item())

        # update weights
        loss.backward(retain_graph=False)  # True
        optimizer.step()
        if orth_counter == 2:
            optimizer = optim.Adam(nn.parameters(), lr=lr * 1.5, betas=betas)
        if orth_counter == 3:
            optimizer = optim.Adam(nn.parameters(), lr=lr * 2, betas=betas)
        optimizer.zero_grad()

        # Keep the best model (lowest loss) by using a deep copy
        if criteria_loss < best_criteria:
            fc1 = copy.deepcopy(nn)
            best_criteria = criteria_loss

        E_bin = int(lam.item() // 1)
        if criteria_loss < dic[E_bin][1]:
            dic[E_bin] = [copy.deepcopy(nn), criteria_loss,
                            (input, r_pde, H_psi, f)]

    TePf = time.time()
    runTime = TePf - TeP0
    loss_histories = (Loss_history, nontriv_loss_history, internal_SE_loss,
                     nn, lam_history, dic, orth_losses, rm_histroy)
    return fc1, loss_histories, runTime, nn

t0 =0.
L = np.pi
xBC1=0.

n_points, neurons, first_epochs, epochs, lr = 1600, 10, int(2e4), int(8e4), 1e-2
model1,loss_hists1,runTime1, latest_model = run_Scan_finitewell(t0, L, neurons, first_epochs, epochs, n_points, lr)

# Plot lambdas
save_plot(loss_hists1[4], 'Epochs', 'Lambda n', lambda_path, 'Lambda n')
save_plot(loss_hists1[2], 'Epochs', 'PDE Loss', pde_loss_path, 'PDE Loss', True)
save_plot(loss_hists1[1], 'Epochs', 'NonTriv Loss', nontriv_loss_path, 'Nontriv Loss', True)
save_plot(loss_hists1[6], 'Epochs', 'Orth Loss', orth_loss_path, 'Orth Loss', True)
save_plot(loss_hists1[7], 'Epochs', 'rm', rm_path, 'rm')

fig, axs = plt.subplots(2,3, figsize=(12,6))
axs[0,0].plot(loss_hists1[4])
axs[0,0].set_title('Lambda n')
axs[0,1].semilogy(loss_hists1[2])
axs[0,1].set_title('PDE Loss')
axs[0,2].semilogy(loss_hists1[1])
axs[0, 2].set_title('NonTriv Loss')
axs[1,0].semilogy(loss_hists1[6])
axs[1,0].set_title('Orth Loss')
axs[1,1].plot(loss_hists1[7])
axs[1,1].set_title('rm')

fig.tight_layout()
fig.savefig(all_path)

plot_solutions(loss_hists1[5], L, n_points)
