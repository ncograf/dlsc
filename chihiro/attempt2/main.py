import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import matplotlib.pyplot as plt
import copy

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


def save_plots(loss_history:list, pde_hist: list, norm_hist: list, orth_hist: list, lambda_hist: list):
    plt.close();plt.cla();
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    axs[0,0].semilogy(loss_history)
    axs[0,0].set_title('Total Loss')
    axs[0,0].grid(True)

    axs[0,1].semilogy(pde_hist)
    axs[0,1].set_title('PDE Loss')
    axs[0,1].grid(True)

    axs[0,2].semilogy(norm_hist)
    axs[0,2].set_title('Norm Loss')
    axs[0,2].grid(True)

    axs[1,0].semilogy(orth_hist)
    axs[1,0].set_title('Orth Loss')
    axs[1,0].grid(True)

    axs[1,1].plot(lambda_hist)
    axs[1,1].set_title('Lambda')
    axs[1,1].grid(True)

    fig.tight_layout()
    fig.savefig(all_path)
    return


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

    orth_counter = [0]

    dic = dict()
    dic[1] = (None, 1e4)
    dic[2] = (None, 1e4)
    dic[3] = (None, 1e4)
    dic[4] = (None, 1e4)

    for epoch in range(4):
        optimizer = torch.optim.LBFGS(network.parameters(),
                                  lr=float(0.05) if orth_counter[0] == 0 else float(0.04),
                                  # max_iter=100,
                                  max_iter=1950,
                                  max_eval=1950,
                                  history_size=800,
                                  tolerance_change=1.0 * np.finfo(float).eps)
        
        try:
            for j, (x_train,) in enumerate(training_set):
                def closure() -> float:
                    optimizer.zero_grad()

                    x_train.requires_grad = True

                    n1, lambda_n = network(x_train)
                    pred_u = parametric_solutions(x_train, n1, x0, xf, 0)

                    loss_pde = pde_loss(x_train, pred_u, lambda_n)

                    # Impose squared integral = 1
                    # loss_norm = (torch.sum(pred_u**2) - 1)**2 / 25
                    loss_norm = (torch.sqrt(torch.dot(pred_u[:,0], pred_u[:,0])) - 1).pow(2)
                    # loss_norm = torch.Tensor([0])

                    loss_tot = loss_pde + loss_norm

                    loss_orth = Tensor([0])
                    if orth_counter[0]:
                        if orth_counter[0] == 1:
                            par1 = parametric_solutions(x_train, dic[1][0](x_train)[0], x0, xf, 0)[:, 0]
                            loss_orth = torch.sqrt(torch.dot(par1, pred_u[:,0].pow(2)))/25
                            loss_tot += loss_orth
                        elif orth_counter[0] == 2:
                            par1 = parametric_solutions(x_train, dic[1][0](x_train)[0], x0, xf, 0)
                            par2 = parametric_solutions(x_train, dic[3][0](x_train)[0], x0, xf, 0)
                            loss_orth = torch.sqrt(torch.dot(par1[:,0] + par2[:,0], 
                                                  pred_u[:,0]).pow(2))/25
                            loss_tot += loss_orth
                        elif orth_counter[0] == 3:
                            par1 = parametric_solutions(x_train, dic[1][0](x_train)[0], x0, xf, 0)
                            par2 = parametric_solutions(x_train, dic[3][0](x_train)[0], x0, xf, 0)
                            par3 = parametric_solutions(x_train, dic[2][0](x_train)[0], x0, xf, 0)
                            loss_orth = torch.sqrt(torch.dot(par1[:,0] + par2[:,0] + par3[:,0], 
                                                  pred_u[:,0]).pow(2))
                            loss_tot += loss_orth

                        loss_history_orth.append(loss_orth.item())
                        
                    loss_tot.backward()

                    # orth_counter[0] += 1

                    loss_history_pde.append(loss_pde.item())
                    loss_history_norm.append(loss_norm.item())
                    loss_history.append(loss_tot.item())
                    history_lambda.append(lambda_n[0].item())

                    if not len(loss_history) % 300:
                        print(f'################# Epoch {len(loss_history)} ################')
                        print('Total Loss: ', loss_tot.item())
                        print('PDE loss: ', loss_pde.item())
                        print('Norm loss: ', loss_norm.item())
                        print('Orth loss: ', loss_orth.item())
                        print('Lambda: ', lambda_n[0].item())
                        print('Orth counter: ', orth_counter[0])

                        save_plots(loss_history, loss_history_pde, loss_history_norm, loss_history_orth, history_lambda)

                    bin = round(lambda_n[0].data.tolist()[0])
                    print(orth_counter[0], bin, lambda_n[0])
                    if bin in [1,2,3,4] and loss_pde < dic[bin][1]:
                        dic[bin] = (copy.deepcopy(network), loss_pde)

                    if len(loss_history) > 3:
                        decreasing = (loss_history[-3] > loss_history[-2]) and (loss_history[-2] > loss_history[-1])
                    
                    if orth_counter[0] == 0 and loss_tot.item() < 2e-5:
                        save_plots(loss_history, loss_history_pde, loss_history_norm, loss_history_orth, history_lambda)
                        print(f'Orth {orth_counter[0]} complete. Total Loss: {loss_tot.item()}')
                        raise OptimizationComplete
                    if orth_counter[0] == 1 and decreasing and loss_tot.item() < 6.7e-3:
                        save_plots(loss_history, loss_history_pde, loss_history_norm, loss_history_orth, history_lambda)
                        print(f'Orth {orth_counter[0]} complete. Total Loss: {loss_tot.item()}')
                        raise OptimizationComplete

                    return loss_tot.item()
                
                optimizer.step(closure=closure)
        except OptimizationComplete:
            pass

        orth_counter[0] += 1

    # plt.semilogy(loss_history_norm)
    # plt.semilogy(loss_history_pde[:50])
    # plt.show()

    # plt.plot(history_lambda)
    # plt.show()
    
    n1 = dic[1][0](x_samples)[0]
    pred_u = parametric_solutions(x_samples, n1, x0, xf, 0)
    print(torch.sum(pred_u **2), torch.dot(pred_u[:,0], pred_u[:,0]))
    plt.close();plt.cla();
    plt.scatter(x_samples.detach(), pred_u.detach(), label='Pred u')
    plt.scatter(x_samples.detach(), 0.04*torch.sin(x_samples).detach(), label='sin(x)')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'solution1.png'))

    n1 = dic[3][0](x_samples)[0]
    pred_u = parametric_solutions(x_samples, n1, x0, xf, 0)
    plt.close();plt.cla();
    print(torch.sum(pred_u **2), torch.dot(pred_u[:,0], pred_u[:,0]))
    plt.scatter(x_samples.detach(), pred_u.detach(), label='pred u')
    plt.scatter(x_samples.detach(), 0.04*torch.sin(-3*x_samples).detach(), label='sin(-3x)')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'solution3.png'))

    n1 = dic[2][0](x_samples)[0]
    pred_u = parametric_solutions(x_samples, n1, x0, xf, 0)
    plt.close();plt.cla();
    print(torch.sum(pred_u **2), torch.dot(pred_u[:,0], pred_u[:,0]))
    plt.scatter(x_samples.detach(), pred_u.detach(), label='pred u')
    plt.scatter(x_samples.detach(), 0.04*torch.sin(2*x_samples).detach(), label='sin(2x)')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'solution2.png'))

    n1 = dic[4][0](x_samples)[0]
    pred_u = parametric_solutions(x_samples, n1, x0, xf, 0)
    plt.close();plt.cla();
    print(torch.sum(pred_u **2), torch.dot(pred_u[:,0], pred_u[:,0]))
    plt.scatter(x_samples.detach(), pred_u.detach(), label='pred u')
    plt.scatter(x_samples.detach(), 0.04*torch.sin(4*x_samples).detach(), label='sin(4x)')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'solution4.png'))

x0, xf = 0., np.pi
epochs, n_samples = int(1), 1200
batch_size = n_samples
train(x0, xf, epochs, n_samples, batch_size)
