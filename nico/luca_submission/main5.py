import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
from torch import Tensor, nn
import random
import matplotlib.pyplot as plt
from torch.autograd import grad
from torch.utils.data import DataLoader, TensorDataset
from common import *
import time
import copy
from scipy import optimize

torch.autograd.set_detect_anomaly(True)

seed = 120
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)


base_dir = os.path.dirname(os.path.realpath(__file__))

plot_dir = os.path.join(base_dir, 'plots')
if not os.path.isdir(plot_dir):
    os.mkdir(plot_dir)

lambda_path = os.path.join(plot_dir, 'lambda.png')
pde_loss_path = os.path.join(plot_dir, 'pde_loss.png')
nontriv_loss_path = os.path.join(plot_dir, 'nontriv.png')
orth_loss_path = os.path.join(plot_dir, 'orth.png')
rm_path = os.path.join(plot_dir, 'rm.png')
all_path = os.path.join(plot_dir, 'all.png')
all_solutions_path = os.path.join(plot_dir, 'all_solutions.png')
solutions_path = os.path.join(plot_dir, 'solutions.png')

model_dir = os.path.join(base_dir, 'models')
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
model_path = []
for i in range(8):
    model_path.append(os.path.join(model_dir, f'model{i}.pth'))

class Pinns:
    def __init__(self, neurons_, n_samples_, batch_size_, x0_, xf_, load1, load2, load3, load4):
        self.n_samples = n_samples_
        self.batch_size = batch_size_
        self.x0 = x0_
        self.xf = xf_
        self.neurons = neurons_

        self.network = NeuralNet(self.neurons)

        self.network.init_weights()

        self.loss_history = list()
        self.loss_history_pde = list()
        self.loss_history_norm = list()
        self.loss_history_orth = list()
        self.history_lambda = list()

        self.load1 = load1
        self.load2 = load2
        self.load3 = load3
        self.load4 = load4

        self.orth_counter = [0]
        self.epoch_beg = 0

        self.dic = dict()
        self.dic[0] = (None, 1e4)
        self.dic[1] = (None, 1e4)
        self.dic[2] = (None, 1e4)
        self.dic[3] = (None, 1e4)

        self.soboleng = torch.quasirandom.SobolEngine(dimension=1)

        self.x_samples = self.add_points()
        self.training_set = self.assemble_datasets(self.x_samples)

        self.load_states(self.load1, self.load2, self.load3, self.load4)

        def sanity_check():
            t = self.x_samples.detach()
            t, _ = t.sort(dim=0)
            pot = self.potential(t=t)
            plt.close()
            plt.cla()
            plt.plot(t, pot)
            par = self.parametric_solutions(t, 1, self.x0, self.xf, 0)
            plt.plot(t, par)
            plt.show()
        #sanity_check()
        

        self.device_type = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float


    ################################################################################################
    ################################################################################################

    def load_states(self, load1, load2, load3, load4):

        if load1:
            self.network = NeuralNet(self.neurons)
            self.network.load_state_dict(torch.load(model_path[0]))
            self.network.sym = 1
            self.dic[0] = (copy.deepcopy(self.network), 0)
            self.epoch_beg = 1
            self.orth_counter[0] = 1
        if load2:
            self.network = NeuralNet(self.neurons)
            self.network.load_state_dict(torch.load(model_path[1]))
            self.network.sym = 1
            self.dic[1] = (copy.deepcopy(self.network), 0)
            self.epoch_beg = 2
            self.orth_counter[0] = 2
        if load3:
            self.network = NeuralNet(self.neurons)
            self.network.load_state_dict(torch.load(model_path[2]))
            self.network.sym = -1
            self.dic[2] = (copy.deepcopy(self.network), 0)
            self.epoch_beg = 3
            self.orth_counter[0] = 3
        if load4:
            self.network = NeuralNet(self.neurons)
            self.network.load_state_dict(torch.load(model_path[3]))
            self.network.sym = -1
            self.dic[3] = (copy.deepcopy(self.network), 0)
            self.epoch_beg = 4
            self.orth_counter[0] = 4
        self.network.init_weights()   
    
    def load_states_stair(self, load1, load2, load3, load4):

        if load1:
            self.network = NeuralNet(self.neurons)
            self.network.load_state_dict(torch.load(model_path[0]))
            self.network.sym = 1
            self.dic[0] = (copy.deepcopy(self.network), 0)
            self.epoch_beg = 1
            self.orth_counter[0] = 1
            if load2:
                self.network = NeuralNet(self.neurons)
                self.network.load_state_dict(torch.load(model_path[1]))
                self.network.sym = 1
                self.dic[1] = (copy.deepcopy(self.network), 0)
                self.epoch_beg = 2
                self.orth_counter[0] = 2
                if load3:
                    self.network = NeuralNet(self.neurons)
                    self.network.load_state_dict(torch.load(model_path[2]))
                    self.network.sym = -1
                    self.dic[2] = (copy.deepcopy(self.network), 0)
                    self.epoch_beg = 3
                    self.orth_counter[0] = 3
                    if load4:
                        self.network = NeuralNet(self.neurons)
                        self.network.load_state_dict(torch.load(model_path[3]))
                        self.network.sym = -1
                        self.dic[3] = (copy.deepcopy(self.network), 0)
                        self.epoch_beg = 4
                        self.orth_counter[0] = 4
        self.network.init_weights()   
    ################################################################################################
    ################################################################################################

    def add_points(self, verbose=False):
        x_samples = torch.zeros(self.n_samples+1, 1)
        x_samples[:self.n_samples] = self.soboleng.draw(self.n_samples) * (self.xf - self.x0) + self.x0
        x_samples[-1] = self.xf
    
        if verbose:
            print(x_samples[0])
            print(x_samples[-1])

        return x_samples

    def assemble_datasets(self, inputs):
        training_set = DataLoader(TensorDataset(inputs), batch_size=self.batch_size+1, shuffle=True)
        return training_set
    
    ################################################################################################
    ################################################################################################

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
        Vnp = (np.heaviside(-Xsnp-1.7, 0) + np.heaviside(Xsnp-1.7, 0))*20
        Vtorch = torch.from_numpy(Vnp)
        return Vtorch
    
    ################################################################################################
    ################################################################################################

    def compute_pde_residual(self, input, psi, E, V):

        psi_dx = self.dfx(input, psi)
        psi_ddx = self.dfx(input, psi_dx)
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

        return loss_pde, loss_norm, n1, lambda_n, pred_u

    def compute_loss_orth(self, inputs, pred_u, w_orth):
        
        loss_orth = torch.tensor([0])
        orth_sol = torch.zeros_like(pred_u.flatten())
        if self.orth_counter[0] > 0:
            for i in range(self.orth_counter[0]):
                orth_sol += self.parametric_solutions(inputs, self.dic[i][0](inputs)[0], self.x0, self.xf, 0).flatten()
            loss_orth = torch.sqrt(torch.dot(orth_sol, pred_u.flatten()).pow(2)) * w_orth[0]
        
        return loss_orth            
    
    ################################################################################################
    ################################################################################################

    def weights_init(self, m):
        if isinstance(m, nn.Linear) and m.weight.shape[0] != 1:
            torch.nn.init.xavier_uniform(m.weight.data)

    def network_initialization(self, cnt):
        lr = float(0.1)
        w_orth = [0.01]
        w_pde = [1]
        w_norm = [1]
        if cnt == 0:
            lr = float(0.2)
        elif cnt == 3:
            lr = float(0.15)
            w_orth = [0.02]

        return lr, w_pde, w_norm, w_orth

    """def network_initialization(self, int):
        if int == 0:
            lr = float(0.2)
            w_pde = [1]
            w_norm = [1]
            w_orth = [0]
            self.network1.init_weights()

        elif int == 1:
            lr = float(0.2)
            w_pde = [1]
            w_norm = [1]
            w_orth = [0.01]
            self.network2.init_weights()

        elif int == 2:
            lr = float(0.3)
            w_pde = [1]
            w_norm = [1]
            w_orth = [0.01]
            self.network3.init_weights()
            #self.network.sym = -1

        elif int == 3:
            lr = float(0.3)
            w_pde = [1]
            w_norm = [1]
            w_orth = [0.02]
            self.network4.init_weights()
            #self.network.sym = -1

        return lr, w_pde, w_norm, w_orth"""


    def save_or_switch_nico(self, loss, local_loss_history, E):

        #if (not self.dic[self.orth_counter[0]][0]) or loss < self.dic[self.orth_counter[0]][1]:
        #    self.dic[self.orth_counter[0]] = (copy.deepcopy(self.network), loss)

        self.dic[self.orth_counter[0]] = (copy.deepcopy(self.network), loss)

        loss_thresh = [10, 10, 10, 35]
        loss_thresh = loss_thresh[self.orth_counter[0]]
        
        rm_thresh = 1e-4

        window = 1000
        if len(local_loss_history) >= window+1:
            rm = np.mean(np.array(local_loss_history[-window:])-np.array(local_loss_history[-window-1:-1]))
            #rm = np.mean(np.array(local_loss_history[-window-1:-1])-np.array(local_loss_history[-window:]))

        else:
            rm = np.mean(np.array(local_loss_history[1:]) - np.array(local_loss_history[:-1]))
            #rm = np.mean(np.array(local_loss_history[:-1]) - np.array(local_loss_history[1:]))
        
        patience = False
        if rm >= 0 and rm < rm_thresh:
            patience = True

        #if len(local_loss_history) > 3:
        #    patience = rm >= 0 and rm < rm_thresh

        decreasing = False
        if len(local_loss_history) > 3:
            decreasing = (local_loss_history[-3] > local_loss_history[-2]) and (local_loss_history[-2] > local_loss_history[-1])

            
        loss_condition = loss < loss_thresh

        if patience:
            if loss_condition:

                torch.save(self.dic[self.orth_counter[0]][0].state_dict(), model_path[self.orth_counter[0]])
                self.save_solution(self.dic, torch.linspace(self.x0,self.xf, 1500).reshape(-1,1), self.orth_counter[0])
        
                print("\n")
                self.save_plots(self.loss_history, self.loss_history_pde, self.loss_history_norm, self.loss_history_orth, self.history_lambda)
                print(f'Orth {self.orth_counter[0]} complete. Total Loss: {loss.item()}, \
                    eigenvalue = {E}')
                print("\n\n\n")

                self.orth_counter[0] += 1
                self.network.init_weights()
                #self.network.apply(self.weights_init) #stessa cosa che sopra ma con funzione della classe 


                return True, rm
                
            else: 
                self.network.sym  *= -1
                print(f"\n\nPatinece condition satisfied but loss too hight switch symmetry to\
                    {self.network.sym}")
                self.network.init_weights()         #?? se patience soddisfatta ma non loss, non aumento orth counter, cambio simmetria e ricerco la soluzione ?

                return True, rm

        return False, rm


    def save_or_switch_only_loss(self, loss, local_loss_history, E):

        self.dic[self.orth_counter[0]] = (copy.deepcopy(self.network), loss)

        loss_thresh = [10, 10, 10, 35]
        loss_thresh = loss_thresh[self.orth_counter[0]]
        
        rm_thresh = 1e-4

        window = 1000
        if len(local_loss_history) >= window+1:
            rm = np.mean(np.array(local_loss_history[-window-1:-1])-np.array(local_loss_history[-window:]))
        else:
            rm = np.mean(np.array(local_loss_history[:-1]) - np.array(local_loss_history[1:]))
        
        patience = False
        if rm <= 0 and abs(rm) < rm_thresh:
            patience = True
            
        loss_condition = loss < loss_thresh

        decreasing = False
        if len(local_loss_history) > 3:
            decreasing = (
                local_loss_history[-3] > local_loss_history[-2]) and (local_loss_history[-2] > local_loss_history[-1])


        if loss_condition:

            torch.save(self.dic[self.orth_counter[0]][0].state_dict(), model_path[self.orth_counter[0]])
            self.save_solution(self.dic, torch.linspace(self.x0,self.xf, 1500).reshape(-1,1), self.orth_counter[0])
    
            print("\n")
            self.save_plots(self.loss_history, self.loss_history_pde, self.loss_history_norm, self.loss_history_orth, self.history_lambda)
            print(f'Orth {self.orth_counter[0]} complete. Total Loss: {loss.item()}, \
                eigenvalue = {E}')
            print("\n\n\n")

            self.orth_counter[0] += 1
            self.network.init_weights()

            return True, rm
        



    ################################################################################################
    def fit(self):

        for _ in range(100):
            
            # after finding 4 soluitons stop
            if self.orth_counter[0] >= 4:
                break
            local_loss_history = list()

            lr, w_pde, w_norm, w_orth = self.network_initialization(self.orth_counter[0])
            
            optimizer = torch.optim.LBFGS(self.network.parameters(), lr=lr,
                                        max_iter=50000,
                                        max_eval=50000,
                                        history_size=800,
                                        line_search_fn="strong_wolfe",
                                        tolerance_change=1.0 * np.finfo(float).eps)


            try:
                for j, (x_train,) in enumerate(self.training_set):
                    def closure() -> float:

                        optimizer.zero_grad()

                        loss_pde, loss_norm, n1, lambda_n, pred_u = self.compute_loss(x_train, w_pde, w_norm)
                        loss = loss_pde + loss_norm


                        loss_orth = self.compute_loss_orth(x_train, pred_u, w_orth)
                        if loss_orth < 1e-4:
                            loss_orth = torch.tensor([0])

                        loss_tot = loss + loss_orth

                        self.loss_history_pde.append(loss_pde.item())
                        self.loss_history_norm.append(loss_norm.item())
                        self.loss_history_orth.append(loss_orth.item())
                        self.loss_history.append(loss_tot.item())
                        local_loss_history.append(loss_tot.item())
                        self.history_lambda.append(lambda_n[0].item())
                        
                        loss_log = loss_tot.log()
                        loss_log.backward()
          
                        save_or_switch, rm = self.save_or_switch_nico(loss_tot, local_loss_history, lambda_n[0].item())

                        if not len(self.loss_history) % 300:
                            print(
                                f'################# Epoch {len(self.loss_history)} ################')
                            print('Total Loss: ', loss_tot.item())
                            print('PDE loss: ', loss_pde.item())
                            print('Norm loss: ', loss_norm.item())
                            print('Orth loss: ', loss_orth.item())
                            print('Lambda: ', lambda_n[0].item())
                            print('Orth counter: ', self.orth_counter[0])

                            self.save_plots(self.loss_history, self.loss_history_pde, self.loss_history_norm, self.loss_history_orth, self.history_lambda)

                        print("Len loss_history: ", len(self.loss_history),
                              " | orth_counter: ", self.orth_counter[0],
                              " | lambda: ", round(lambda_n[0].item(), 4),
                              " | loss tot: ", round(loss_tot.item(), 6),
                              " | loss pde: ", round(loss_pde.item(), 6),
                              " | loss norm: ", round(loss_norm.item(), 6),
                              " | loss orth: ", round(loss_orth.item(), 6),
                              " | rm: ", round(rm, 6),
                              " | sym: ", self.network.sym)
                        
                        
                        if save_or_switch: 
                            print(" \n Optimization Complete \n")
                            raise OptimizationComplete
                        #elif check == 1:
                        #    print("\n Only patience condition met \n Switching Symmetry \n")

                        return loss_tot.item()

                    optimizer.step(closure=closure)
            except OptimizationComplete:
                pass
            

        self.plot_all()

    ################################################################################################

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
        fig, axs = plt.subplots(1, 4, figsize=(25, 5))
        t = torch.linspace(self.x0, self.xf, 3000).reshape(-1,1)
        self.load_states(True, True, True, True)
        for i in range(4):
            n1, lambda_ = self.dic[i][0](t)
            f = self.parametric_solutions(t, n1, self.x0, self.xf, 0)
            E = np.round(lambda_[0].item(),4)
            axs[i].plot(t.flatten().detach(), f.flatten().detach(), label=f'pred E = {E}')
            axs[i].legend(loc=1)
            
        fig.savefig(all_solutions_path)


if __name__ == "__main__":
    x0, xf = -6., 6.
    epochs = 4
    n_samples = 2000
    batch_size = n_samples
    neurons = 50

    pinn = Pinns(neurons, n_samples, batch_size, x0, xf, load1 = False, load2 = False, load3 = False, load4 = False)

    pinn.fit()
