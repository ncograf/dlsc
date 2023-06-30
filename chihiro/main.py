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

seed = 120
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

# for the plots
plt.rc('xtick', labelsize=16)
plt.rcParams.update({'font.size': 16})

def save_bin(dic: dict, t0: float, tf: float, nTest: int):
    plt.close();plt.cla();
    tTest = torch.linspace(t0,tf,nTest)
    tTest = tTest.reshape(-1,1);
    tTest.requires_grad=True
    t_net = tTest.detach().numpy()
    for bin in dic.keys():
        if bin and dic[bin][0]:
            plt.plot(t_net, parametricSolutions(tTest, dic[bin][0].cpu(), t0, tf, xBC1).detach().numpy(), label=f'$\lambda$={bin}')
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

def run_Scan_finitewell(t0, tf, x1, neurons, first_epochs, epochs, n_train, lr, minibatch_number=1):
    par2 = 0
    fc0 = qNN1(neurons)
    fc1 = 0
    betas = [0.999, 0.9999]
    optimizer = optim.Adam(fc0.parameters(), lr=lr, betas=betas)
    Loss_history = []
    Llim = 1e+20
    En_loss_history = []
    boundary_loss_history = []
    nontriv_loss_history = []
    SE_loss_history = []
    Ennontriv_loss_history = []
    criteria_loss_history = []
    En_history = []
    prob_loss = []
    EWall_history = []
    orth_losses = []
    rm_histroy = []
    di = (None, 1e+20)
    dic = {}
    for i in range(50):
        dic[i] = di
    orth_counter = 0
    swith = False
    internal_SE_loss = []

    grid = torch.linspace(t0, tf, n_train).reshape(-1, 1)

    # TRAINING ITERATION
    TeP0 = time.time()
    walle = -1.5
    last_psi_L = 0
    for tt in range(epochs):
        # adjusting learning rate at epoch 3e4
        if tt == 3e4:
           optimizer = optim.Adam(fc0.parameters(), lr = 1e-2, betas = betas)
        # Perturbing the evaluation points & forcing t[0]=t0
        t = perturbPoints(grid, t0, tf, sig=.03*tf)

        # BATCHING
        batch_size = int(n_train/minibatch_number)
        batch_start, batch_end = 0, batch_size

        idx = np.random.permutation(n_train)
        t_b = t[idx]
        t_b.requires_grad = True
        t_f = t[-1]
        t_f = t_f.reshape(-1, 1)
        t_f.requires_grad = True
        loss = 0.0

        for nbatch in range(minibatch_number):
            # batch time set
            t_mb = t_b[batch_start:batch_end]

            #  Network solutions
            nn, En = fc0(t_mb)
            # + np.random.normal(0,1,1)[0]*torch.ones_like(En)/30
            En = torch.abs(En)

            En_history.append(En[0].data.tolist()[0])

            psi = parametricSolutions(t_mb, fc0, t0, tf, x1)
            Ltot, f_ret, H_psi = hamEqs_Loss(t_mb, psi, En)
            # Ltot /= En.detach()[0].data.tolist()[0]**2
            SE_loss_history.append(Ltot)
            internal_SE_loss.append(Ltot.cpu().detach().numpy())
            criteria_loss = Ltot

            # l_norm = ((n_train/(tf-t0))*1.0 -
                    #  torch.sqrt(torch.dot(psi[:, 0], psi[:, 0]))).pow(2)
            l_norm = (torch.sqrt(torch.dot((En*psi)[:,0], (En*psi)[:,0]))-n_train/(tf-t0)).pow(2)
            Ltot += l_norm

            window = 1000
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

            rm_histroy.append(rm)

            if tt % 300 == 0:
                print('\nEpoch', tt)
                print('PDE Loss', internal_SE_loss[-1])
                print('E', En_history[-1])
                print('rm', rm)
                print('oc', orth_counter)

                plt.close()
                fig, axs = plt.subplots(2,3, figsize=(12,6))
                axs[0,0].plot(En_history)
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

                save_bin(dic, t0, tf, n_train)

            exp_thresh = -10
            if tt == first_epochs:
                fc0.apply(weights_init)
                orth_counter += 1
                print('Epoch', tt)
                print('E', En_history[-1])
                print('rm', rm)
                print('oc', orth_counter)
            elif rm < np.exp(exp_thresh) and rm > 0 and orth_counter == 1:
                fc0.apply(weights_init)
                orth_counter += 1
                print('Epoch', tt)
                print('E', En_history[-1])
                print('rm', rm)
                print('oc', orth_counter)
            elif rm < np.exp(exp_thresh) and rm > 0 and orth_counter == 2:
                orth_counter += 1
                print('Epoch', tt)
                print('E', En_history[-1])
                print('rm', rm)
                print('oc', orth_counter)
            elif rm < np.exp(exp_thresh) and rm > 0 and orth_counter == 3:
                fc0.sym = True
                orth_counter += 1
                print('Epoch', tt)
                print('E', En_history[-1])
                print('rm', rm)
                print('oc', orth_counter)
            elif rm < np.exp(exp_thresh) and rm > 0 and orth_counter == 4:
                TePf = time.time()
                runTime = TePf - TeP0
                loss_histories = (Loss_history, boundary_loss_history, nontriv_loss_history, internal_SE_loss, Ennontriv_loss_history,
                                En_loss_history, criteria_loss_history, fc0, En_history, EWall_history, dic, orth_losses, rm_histroy)
                return fc1, loss_histories, runTime, fc0


            if orth_counter == 1:
                par2 = parametricSolutions(t_mb, dic[1][0], t0, tf, x1)
                ortho_loss = torch.sqrt(
                    torch.dot(par2[:, 0], psi[:, 0]).pow(2))/25
                orth_losses.append(ortho_loss.cpu().detach().numpy())
                Ltot += ortho_loss
            elif orth_counter == 2:
                par2 = parametricSolutions(t_mb, dic[1][0], t0, tf, x1)
                par3 = parametricSolutions(t_mb, dic[2][0], t0, tf, x1)
                ortho_loss = torch.sqrt(torch.dot(par2[:,0]+par3[:,0], psi[:,0]).pow(2))/25
                orth_losses.append(ortho_loss.cpu().detach().numpy())
                Ltot += ortho_loss
            elif orth_counter == 3 or orth_counter == 4:
                par2 = parametricSolutions(t_mb, dic[1][0], t0, tf, x1)
                par3 = parametricSolutions(t_mb, dic[2][0], t0, tf, x1)
                par4 = parametricSolutions(t_mb, dic[3][0], t0, tf, x1)
                ortho_loss = torch.sqrt(torch.dot(
                    par2[:, 0]+par3[:, 0]+par4[:, 0], psi[:, 0]).pow(2))/25  # get rid of sqrt
                orth_losses.append(ortho_loss.cpu().detach().numpy())
                Ltot += ortho_loss

            En_loss_history.append(torch.exp(-1*En+walle).mean())
            EWall_history.append(walle)

            # nontriv_loss_history.append(1/((psi.pow(2)).mean()+1e-6)) #
            nontriv_loss_history.append(l_norm.cpu().detach().numpy())
            # Ennontriv_loss_history.append(1/(En.pow(2).mean()+1e-6)) #
            Ennontriv_loss_history.append(1/En[0][0].pow(2))
            # OPTIMIZER
            Ltot.backward(retain_graph=False)  # True
            optimizer.step()
            loss += Ltot.cpu().data.numpy()
            optimizer.zero_grad()
            del Ltot

            batch_start += batch_size
            batch_end += batch_size

        # keep the loss function history
        Loss_history.append(loss)

        # Keep the best model (lowest loss) by using a deep copy
        if criteria_loss < Llim:
            fc1 = copy.deepcopy(fc0)
            Llim = criteria_loss

        E_bin = abs(En[0].data.tolist()[0])//1
        if criteria_loss < dic[E_bin][1]:
            dic[E_bin] = (copy.deepcopy(fc0), criteria_loss,
                          (t_mb, f_ret, H_psi, psi))

    TePf = time.time()
    runTime = TePf - TeP0
    loss_histories = (Loss_history, boundary_loss_history, nontriv_loss_history, internal_SE_loss, Ennontriv_loss_history,
                      En_loss_history, criteria_loss_history, fc0, En_history, EWall_history, dic, orth_losses, rm_histroy)
    return fc1, loss_histories, runTime, fc0

t0 =0.
tf = np.pi
xBC1=0.

n_train, neurons, first_epochs, epochs, lr,mb = 1600, 10, int(1.5e4), int(8e4), 1e-2, 1
model1,loss_hists1,runTime1, latest_model = run_Scan_finitewell(t0, tf, xBC1, neurons, first_epochs, epochs, n_train, lr, mb)

# Plot lambdas
save_plot(loss_hists1[8], 'Epochs', 'Lambda n', lambda_path, 'Lambda n')
save_plot(loss_hists1[3], 'Epochs', 'PDE Loss', pde_loss_path, 'PDE Loss', True)
save_plot(loss_hists1[2], 'Epochs', 'NonTriv Loss', nontriv_loss_path, 'Nontriv Loss', True)
save_plot(loss_hists1[11], 'Epochs', 'Orth Loss', orth_loss_path, 'Orth Loss', True)
save_plot(loss_hists1[12], 'Epochs', 'rm', rm_path, 'rm')

fig, axs = plt.subplots(2,3, figsize=(12,6))
axs[0,0].plot(loss_hists1[8])
axs[0,0].set_title('Lambda n')
axs[0,1].semilogy(loss_hists1[3])
axs[0,1].set_title('PDE Loss')
axs[0,2].semilogy(loss_hists1[2])
axs[0, 2].set_title('NonTriv Loss')
axs[1,0].semilogy(loss_hists1[11])
axs[1,0].set_title('Orth Loss')
axs[1,1].plot(loss_hists1[12])
axs[1,1].set_title('rm')

fig.tight_layout()
fig.savefig(all_path)

save_bin(loss_hists1[10], t0, tf, n_train)

print('\n############## CHECK ORTH ###################')
nTest = n_train; tTest = torch.linspace(t0,tf,nTest)
tTest = tTest.reshape(-1,1);
tTest.requires_grad=True
t_net = tTest.detach().numpy()
li = []
for bin in loss_hists1[10].keys():
    if bin and loss_hists1[10][bin][0]:
        li.append([bin, parametricSolutions(tTest, loss_hists1[10][bin][0].cpu(), t0, tf, xBC1)])

print('1x2 ', torch.dot(li[0][1][:,0],li[1][1][:,0]))
print('1x3 ', torch.dot(li[0][1][:,0],li[2][1][:,0]))
print('1x4 ', torch.dot(li[0][1][:,0],li[3][1][:,0]))