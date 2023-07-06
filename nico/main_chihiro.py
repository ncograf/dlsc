import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import matplotlib.pyplot as plt

from package.directories import *
from package.modules import *
from package.utils import *

seed = 120
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

device_type = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# for the plots
plt.rc('xtick', labelsize=16)
plt.rcParams.update({'font.size': 16})


def train(x0: Tensor, xf: Tensor, epochs: int, n_samples: int, batch_size: int):
    network = qNN1()

    soboleng = torch.quasirandom.SobolEngine(dimension=1)
    x_samples = torch.zeros(n_samples+1, 1)
    x_samples[:n_samples] = soboleng.draw(n_samples) * (xf - x0) + x0
    x_samples[-1] = np.pi

    training_set = DataLoader(TensorDataset(
        x_samples), batch_size=batch_size, shuffle=True)
    
    loss_history = list()
    loss_history_pde = list()
    loss_history_norm = list()
    loss_history_orth = list()
    history_lambda = list()
    
    best_loss_pde = [1e10, 1e10, 1e10]

    orth_counter = [0]

    dic = dict()

    for epoch in range(4):
        optimizer = torch.optim.LBFGS(network.parameters(),
                                  lr=float(0.08),
                                  # max_iter=100,
                                  max_iter=3000,
                                  max_eval=3000,
                                  history_size=800,
                                  tolerance_change=1.0 * np.finfo(float).eps)
        for j, (x_train,) in enumerate(training_set):
            if j > 0:
                break
            def closure() -> float:
                optimizer.zero_grad()

                x_train.requires_grad = True

                n1, lambda_n = network(x_train)
                pred_u = parametric_solutions(x_train, n1, x0, xf, 0)

                loss_pde = pde_loss(x_train, pred_u, lambda_n)
                
                if torch.isnan(loss_pde):
                    print(f"Nan-detected epoch: {len(loss_history)}")
                    print(f"pred_u head: {pred_u[0:10]}")
                    print(f"pred_u max: {pred_u.max()}")
                    print(f"pred_u min: {pred_u.min()}")
                    print(f"pred_u mean: {pred_u.mean()}")
                    print(f"pred_u sum: {pred_u.sum()}")
                    return False

                # Impose squared integral = 1
                # loss_norm = (torch.sum(pred_u**2) - 1)**2 / 25
                loss_norm = (torch.sqrt(torch.dot(pred_u[:,0], pred_u[:,0])) - 1).pow(2)
                # loss_norm = torch.Tensor([0])

                loss_tot = loss_pde + loss_norm

                loss_orth = torch.tensor([0])
                if orth_counter[0]:
                    if orth_counter[0] == 1:
                        loss_orth = torch.sqrt(torch.dot(parametric_solutions(x_train, dic[0](x_train)[0], x0, xf, 0)[:,0], 
                                              pred_u[:,0]).pow(2))/25
                        loss_tot += loss_orth
                    elif orth_counter[0] == 2:
                        f_orth = parametric_solutions(x_train, dic[0](x_train)[0], x0, xf, 0)[:,0]
                        f_orth += parametric_solutions(x_train, dic[1](x_train)[0], x0, xf, 0)[:,0]
                        loss_orth = torch.sqrt(torch.dot(f_orth.flatten(), pred_u.flatten()).pow(2)) * 2
                        loss_tot += loss_orth
                    # elif orth_counter[0] == 2 and 1 in dic.keys():
                    #     par1 = parametric_solutions(x_train, dic[0](x_train)[0], x0, xf, 0)
                    #     par2 = parametric_solutions(x_train, dic[1](x_train)[0], x0, xf, 0)
                    #     loss_orth = torch.sqrt(torch.dot(par1[:,0] + par2[:,0], 
                    #                           pred_u[:,0]).pow(2))/25
                    #     loss_tot += loss_orth
                loss_tot.backward()

                # orth_counter[0] += 1

                loss_history_pde.append(loss_pde.item())
                loss_history_norm.append(loss_norm.item())
                loss_history.append(loss_tot.item())
                loss_history_orth.append(loss_orth.item())
                history_lambda.append(lambda_n[0].item())

                if not len(loss_history) % 300:
                    print(f'################# Epoch {len(loss_history)} ################')
                    print('PDE loss: ', loss_pde.item())
                    print('Norm loss: ', loss_norm.item())
                    print('Orth loss: ', loss_orth.item())
                    print('Lambda: ', lambda_n[0].item())
                    print('Orth counter: ', orth_counter[0])

                import copy
                if orth_counter[0] == 0:
                    dic[0] = copy.deepcopy(network)
                    best_loss_pde[0] = loss_pde

                elif lambda_n[0] > 1.88 and orth_counter[0] == 1:
                    dic[1] = copy.deepcopy(network)
                    best_loss_pde[1] = loss_pde

                elif lambda_n[0] > 2.88 and orth_counter[0] == 2 and loss_pde < best_loss_pde[2]:
                    dic[2] = copy.deepcopy(network)
                    best_loss_pde[2] = loss_pde
                        
                return loss_tot.item()
            
            optimizer.step(closure=closure)
            # plt.plot(history_lambda)
            # plt.show()
            orth_counter[0] += 1

    # plt.semilogy(loss_history_norm)
    # plt.semilogy(loss_history_pde[:50])
    # plt.show()

    plt.plot(history_lambda)
    plt.savefig("chihiro/attempt2/plots/history.png")
    plt.clf()

    n1 = dic[0](x_samples)[0]
    pred_u = parametric_solutions(x_samples, n1, x0, xf, 0)
    print(torch.sum(pred_u **2), torch.dot(pred_u[:,0], pred_u[:,0]))
    plt.scatter(x_samples.detach(), pred_u.detach(), label='Pred u')
    plt.scatter(x_samples.detach(), 0.04*torch.sin(x_samples).detach(), label='sin(x)')
    plt.legend()
    plt.savefig("chihiro/attempt2/plots/solution_1.png")
    plt.clf()

    n1 = dic[1](x_samples)[0]
    pred_u = parametric_solutions(x_samples, n1, x0, xf, 0)
    print(torch.sum(pred_u **2), torch.dot(pred_u[:,0], pred_u[:,0]))
    plt.scatter(x_samples.detach(), pred_u.detach(), label='pred u')
    plt.scatter(x_samples.detach(), 0.04*torch.sin(-2*x_samples).detach(), label='sin(2x)')
    plt.legend()
    plt.savefig("chihiro/attempt2/plots/solution_2.png")
    plt.clf()

    n1 = dic[2](x_samples)[0]
    pred_u = parametric_solutions(x_samples, n1, x0, xf, 0)
    print(torch.sum(pred_u **2), torch.dot(pred_u[:,0], pred_u[:,0]))
    plt.scatter(x_samples.detach(), pred_u.detach(), label='pred u')
    plt.scatter(x_samples.detach(), 0.04*torch.sin(-3*x_samples).detach(), label='sin(3x)')
    plt.legend()
    plt.savefig("chihiro/attempt2/plots/solution_3.png")
    plt.clf()

x0, xf = 0., np.pi
epochs, n_samples = int(1), 1200
batch_size = n_samples
train(x0, xf, epochs, n_samples, batch_size)
