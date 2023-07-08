from scipy import optimize
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


def save_plots(loss_history: list, pde_hist: list, norm_hist: list, orth_hist: list, lambda_hist: list):
    plt.close()
    plt.cla()
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


def save_solution(dic: dict, x_samples: Tensor,  index: int):
    print(f'############# Save Solution_{index} ################')
    n1 = dic[index][0](x_samples)[0]
    pred_u = parametric_solutions(x_samples, n1, x0, xf, 0)
    print(f'Check Normalization: ', torch.sum(pred_u ** 2))

    c_2 = n_samples / xf / \
        torch.sqrt(torch.dot(torch.sin(x_samples)[
                   :, 0], torch.sin(x_samples)[:, 0]))

    plt.close()
    plt.cla()
    plt.scatter(x_samples.detach(), pred_u.detach(), label='pred u', s=2)
    if index == 3:
        plt.scatter(x_samples.detach(), c_2*torch.sin(-3 *
                    x_samples).detach(), label=f'$sin(-3x)$', s=2)
    if index in [1,2]:
        plt.scatter(x_samples.detach(), c_2 *
                    torch.sin(-index*x_samples).detach(), label=f'$sin(-{2 if index==2 else ""}x)$', s=2)
    else:
        plt.scatter(x_samples.detach(), c_2*torch.sin(index *
                    x_samples).detach(), label=f'$sin({index}x)$', s=2)
    plt.legend()
    plt.savefig(os.path.join(plot_dir, f'solution_{index}.png'))


def train(x0: Tensor, xf: Tensor, epochs: int, n_samples: int, batch_size: int,
          load1: bool = False, load2: bool = False, load3: bool = False):
    network = qNN1(100)

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

    epoch_beg = 0
    if load1:
        net1 = qNN1(100)
        net1.load_state_dict(torch.load(model1_path))
        dic[1] = (net1, 1e-20)
        epoch_beg = 1
        orth_counter[0] = 1
    if load2:
        net2 = qNN1(100)
        net2.load_state_dict(torch.load(model2_path))
        dic[2] = (net1, 1e-20)
        epoch_beg = 2
        orth_counter[0] = 2
    if load3:
        net3 = qNN1(100)
        net3.load_state_dict(torch.load(model3_path))
        dic[3] = (net1, 1e-20)
        epoch_beg = 3
        orth_counter[0] = 3

    for epoch in range(epoch_beg, 2):
        w = 1/25
        w_pde = 1
        if orth_counter[0] == 0:
            lr = float(0.05)
        elif orth_counter[0] == 3:
            lr = float(0.1)
            w = 1
        else:
            lr = float(0.04)
            w = 0.6
            w_pde = 2

        optimizer = torch.optim.LBFGS(network.parameters(),
                                      lr=lr,
                                      max_iter=5000,
                                      max_eval=5000,
                                      history_size=800,
                                      tolerance_change=1.0 * np.finfo(float).eps)

        try:
            for j, (x_train,) in enumerate(training_set):
                def closure() -> float:
                    optimizer.zero_grad()

                    x_train.requires_grad = True

                    n1, lambda_n = network(x_train)
                    pred_u = parametric_solutions(x_train, n1, x0, xf, 0)

                    loss_pde = pde_loss(x_train, pred_u, lambda_n) * w_pde

                    # Impose squared integral = 1
                    # loss_norm = (torch.sum(pred_u**2) - 1)**2 / 25
                    loss_norm = (torch.sqrt(
                        torch.dot(pred_u[:, 0], pred_u[:, 0])) - n_samples/xf).pow(2)
                    # loss_norm = torch.Tensor([0])

                    loss_tot = loss_pde + loss_norm

                    loss_orth = Tensor([0])
                    if orth_counter[0]:
                        if orth_counter[0] == 1:
                            par1 = parametric_solutions(
                                x_train, dic[1][0](x_train)[0], x0, xf, 0)[:, 0]
                            loss_orth = torch.sqrt(
                                torch.dot(par1, pred_u[:, 0]).pow(2)) * w
                            loss_tot += loss_orth
                        elif orth_counter[0] == 2:
                            par1 = parametric_solutions(
                                x_train, dic[1][0](x_train)[0], x0, xf, 0)
                            par2 = parametric_solutions(
                                x_train, dic[2][0](x_train)[0], x0, xf, 0)
                            loss_orth = torch.sqrt(torch.dot(par1[:, 0] + par2[:, 0],
                                                             pred_u[:, 0]).pow(2)) * w
                            loss_tot += loss_orth
                        elif orth_counter[0] == 3:
                            par1 = parametric_solutions(
                                x_train, dic[1][0](x_train)[0], x0, xf, 0)
                            par2 = parametric_solutions(
                                x_train, dic[3][0](x_train)[0], x0, xf, 0)
                            par3 = parametric_solutions(
                                x_train, dic[2][0](x_train)[0], x0, xf, 0)
                            loss_orth = torch.sqrt(torch.dot(par1[:, 0] + par2[:, 0] + par3[:, 0],
                                                             pred_u[:, 0]).pow(2)) * w
                            loss_tot += loss_orth

                        loss_history_orth.append(loss_orth.item())

                    loss_tot.backward()

                    loss_history_pde.append(loss_pde.item())
                    loss_history_norm.append(loss_norm.item())
                    loss_history.append(loss_tot.item())
                    history_lambda.append(lambda_n[0].item())

                    if not len(loss_history) % 300:
                        print(
                            f'################# Epoch {len(loss_history)} ################')
                        print('Total Loss: ', loss_tot.item())
                        print('PDE loss: ', loss_pde.item())
                        print('Norm loss: ', loss_norm.item())
                        print('Orth loss: ', loss_orth.item())
                        print('Lambda: ', lambda_n[0].item())
                        print('Orth counter: ', orth_counter[0])

                        save_plots(loss_history, loss_history_pde,
                                   loss_history_norm, loss_history_orth, history_lambda)

                    bin = round(lambda_n[0].data.tolist()[0])
                    print(len(loss_history), orth_counter[0], bin, round(lambda_n[0].item(), 4),
                          round(loss_pde.item(), 6), round(loss_norm.item(), 6), round(loss_orth.item(), 6))
                    if bin in [1, 2, 3, 4] and loss_pde < dic[bin][1]:
                        dic[bin] = (copy.deepcopy(network), loss_pde)

                    decreasing = False
                    if len(loss_history) > 3:
                        decreasing = (
                            loss_history[-3] > loss_history[-2]) and (loss_history[-2] > loss_history[-1])

                    if orth_counter[0] == 0 and loss_tot.item() < 0.009:
                        torch.save(network.state_dict(), model1_path)
                        save_plots(loss_history, loss_history_pde,
                                   loss_history_norm, loss_history_orth, history_lambda)
                        print(
                            f'Orth {orth_counter[0]} complete. Total Loss: {loss_tot.item()}')
                        raise OptimizationComplete
                    if orth_counter[0] == 1 and decreasing and loss_tot.item() < 5.1:
                        save_plots(loss_history, loss_history_pde,
                                   loss_history_norm, loss_history_orth, history_lambda)
                        print(
                            f'Orth {orth_counter[0]} complete. Total Loss: {loss_tot.item()}')
                        raise OptimizationComplete
                    if orth_counter[0] == 2 and decreasing and loss_tot.item() < 1.99e-4:
                        save_plots(loss_history, loss_history_pde,
                                   loss_history_norm, loss_history_orth, history_lambda)
                        print(
                            f'Orth {orth_counter[0]} complete. Total Loss: {loss_tot.item()}')
                        raise OptimizationComplete

                    return loss_tot.item()

                optimizer.step(closure=closure)
        except OptimizationComplete:
            pass

        orth_counter[0] += 1

    save_solution(dic, x_samples, 1)
    save_solution(dic, x_samples, 2)
    save_solution(dic, x_samples, 3)


x0, xf = 0., np.pi
epochs, n_samples = int(1), 1200
batch_size = n_samples
train(x0, xf, epochs, n_samples, batch_size, load1=True)
