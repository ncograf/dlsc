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

device_type = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dtype = torch.float


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
solutions_path = os.path.join(plot_dir, 'solutions.png')

model_dir = os.path.join(base_dir, 'models')
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
model1_path = os.path.join(model_dir, 'model1.pth')
model2_path = os.path.join(model_dir, 'model2.pth')
model3_path = os.path.join(model_dir, 'model3.pth')
model4_path = os.path.join(model_dir, 'model4.pth')



class Pinns:
    def __init__(self, neurons_, n_samples_, batch_size_, x0_, xf_, load1, load2, load3, load4):
        self.n_samples = n_samples_
        self.batch_size = batch_size_
        self.x0 = x0_
        self.xf = xf_
        self.neurons = neurons_

        self.network1 = NeuralNet(self.neurons)
        self.network2 = NeuralNet(self.neurons)
        self.network3 = NeuralNet(self.neurons)
        self.network4 = NeuralNet(self.neurons)

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
        self.dic[1] = (None, 1e4)
        self.dic[2] = (None, 1e4)
        self.dic[3] = (None, 1e4)
        self.dic[4] = (None, 1e4)

        self.load_states(self.load1, self.load2, self.load3, self.load4)

        self.soboleng = torch.quasirandom.SobolEngine(dimension=1)

        self.x_samples = self.add_points()
        self.training_set = self.assemble_datasets(self.x_samples)


        def sanity_check():
            t = self.x_samples.detach()
            t, _ = t.sort(dim=0)
            pot = self.potential(t=t)
            plt.plot(t, pot)
            par = self.parametric_solutions(t, 1, self.x0, self.xf, 0)
            plt.plot(t, par)
            plt.show()
        sanity_check()

        self.device_type = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float


    ################################################################################################
    ################################################################################################

    def load_states(self, load1, load2, load3, load4):

        if load1:
            self.network1 = NeuralNet(self.neurons)
            self.network1.load_state_dict(torch.load(model1_path))
            self.dic[1] = (copy.deepcopy(self.network1), 0)
            self.epoch_beg = 1
            self.orth_counter[0] = 1
        if load2:
            self.network2 = NeuralNet(self.neurons)
            self.network2.load_state_dict(torch.load(model2_path))
            self.dic[2] = (copy.deepcopy(self.network2), 0)
            self.epoch_beg = 2
            self.orth_counter[0] = 2
        if load3:
            self.network3 = NeuralNet(self.neurons)
            self.network3.load_state_dict(torch.load(model3_path))
            self.dic[3] = (copy.deepcopy(self.network3), 0)
            self.epoch_beg = 3
            self.orth_counter[0] = 3
        if load4:
            self.network4 = NeuralNet(self.neurons)
            self.network4.load_state_dict(torch.load(model4_path))
            self.dic[4] = (copy.deepcopy(self.network4), 0)
            self.epoch_beg = 4
            self.orth_counter[0] = 4

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
        f = (1-torch.exp(-(t-t0)))*(1-torch.exp(-(t-tf)))
        psi_hat = x1 + f*N1
        return psi_hat

    def dfx(self, x, f):
        gopts = torch.ones(x.shape, dtype=dtype)
        if device_type == 'cuda:0':
            gopts = gopts.to(device = torch.device('cuda:0'))
        return grad([f], [x], grad_outputs=gopts, create_graph=True)[0]

    def potential(self, t):
        Xsnp = t.data.numpy()
        Vnp = (np.heaviside(-Xsnp-np.pi, 0) + np.heaviside(Xsnp-np.pi, 0))*20
        Vtorch = torch.from_numpy(Vnp)
        return Vtorch
    
    ################################################################################################
    ################################################################################################

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
        
        if self.orth_counter[0] == 0:
            n1, lambda_n = self.network1(inputs)
        if self.orth_counter[0] == 1:
            n1, lambda_n = self.network2(inputs)
        if self.orth_counter[0] == 2:
            n1, lambda_n = self.network3(inputs)
        if self.orth_counter[0] == 3:
            n1, lambda_n = self.network4(inputs)

        pred_u = self.parametric_solutions(t=inputs, N1=n1, t0=self.x0, tf=self.xf, x1=0)

        loss_pde = self.compute_pde_residual(inputs, pred_u, lambda_n, pot_n)*w_pde[0]

        loss_norm = (torch.sqrt(torch.dot(pred_u[:, 0], pred_u[:, 0])) - self.n_samples/(self.xf-self.x0)).pow(2) * w_norm[0]
                    
        loss_drive = torch.exp(-lambda_n + 0)

        loss = loss_pde + loss_norm

        return loss, loss_pde, loss_norm, n1, lambda_n, pred_u

    def compute_loss_orth(self, inputs, pred_u, w_orth):
        
        loss_orth = Tensor([0])

        if self.orth_counter[0] == 1:
        
            par1 = self.parametric_solutions(inputs, self.dic[1][0](inputs)[0], self.x0, self.xf, 0)

            loss_orth = torch.sqrt(torch.dot(par1[:,0], pred_u[:,0]).pow(2)) * w_orth[0]
                        

        elif self.orth_counter[0] == 2:
        
            par1 = self.parametric_solutions(inputs, self.dic[1][0](inputs)[0], self.x0, self.xf, 0)
            par2 = self.parametric_solutions(inputs, self.dic[2][0](inputs)[0], self.x0, self.xf, 0)

            loss_orth = torch.sqrt(torch.dot(par1[:,0] + par2[:,0], pred_u[:,0]).pow(2)) * w_orth[0]


        elif self.orth_counter[0] == 3:
        
            par1 = self.parametric_solutions(inputs, self.dic[1][0](inputs)[0], self.x0, self.xf, 0)
            par2 = self.parametric_solutions(inputs, self.dic[4][0](inputs)[0], self.x0, self.xf, 0)
            par3 = self.parametric_solutions(inputs, self.dic[2][0](inputs)[0], self.x0, self.xf, 0)

            loss_orth = torch.sqrt(torch.dot(par1[:,0] + par2[:,0] + par3[:, 0], pred_u[:,0]).pow(2)) * w_orth[0]

        return loss_orth
            
    ################################################################################################
    ################################################################################################

    def weights_init(self, m):
        if isinstance(m, nn.Linear) and m.weight.shape[0] != 1:
            torch.nn.init.xavier_uniform(m.weight.data)

    def weights_initialization(self, int):
        if int == 0:
            lr = float(0.1)
            w_pde = [1]
            w_norm = [1]
            w_orth = [1/25]

        elif int == 1:
            lr = float(0.04)
            w_pde = [1]
            w_norm = [1]
            w_orth = [1/25]

        elif int == 2:
            lr = float(0.05)
            w_pde = [1]
            w_norm = [1]
            w_orth = [1/25]

        elif int == 3:
            lr = float(0.04)
            w_pde = [1]
            w_norm = [1]
            w_orth = [1/25]

        return lr, w_pde, w_norm, w_orth

    def save_or_switch(self, loss, decreasing, rm):

        if self.orth_counter[0] == 1:
            self.dic[self.orth_counter[0]] = (copy.deepcopy(self.network1), loss)

        if self.orth_counter[0] == 2:
            self.dic[self.orth_counter[0]] = (copy.deepcopy(self.network2), loss)

        if self.orth_counter[0] == 3:
            self.dic[self.orth_counter[0]] = (copy.deepcopy(self.network3), loss)

        if self.orth_counter[0] == 4:
            self.dic[self.orth_counter[0]] = (copy.deepcopy(self.network4), loss)


        exp_thresh = -14 #for the rolling mean rm treshold to satisfy the patience condition
        loss_thresh = 10


        if rm < np.exp(exp_thresh) and rm > 0 and self.orth_counter[0] == 0:
            self.network1.apply(self.weights_init)
            print(" \n Patience Codition Satisfied Sol 1 \n")
            
            if loss < loss_thresh: 

                torch.save(self.network1.state_dict(), model1_path)

                print(" \n Loss Codition Satisfied Sol 1 \n")

                print("\n\n\n")
                self.save_plots(self.loss_history, self.loss_history_pde, self.loss_history_norm, self.loss_history_orth, self.history_lambda)
                print(f'Orth {self.orth_counter[0]} complete. Total Loss: {loss.item()}')
                print("\n\n\n")

                self.dic[1] = (copy.deepcopy(self.network1), loss)
                self.save_solution(self.dic, self.x_samples, 1)

                return 0

            self.network1.sym = not self.network1.sym

            return 1

        if rm < np.exp(exp_thresh) and rm > 0 and self.orth_counter[0] == 1:
            self.network2.apply(self.weights_init)
            print(" \n Patience Codition Satisfied Sol 2 \n")
            
            if loss < loss_thresh: 

                torch.save(self.network2.state_dict(), model2_path)

                print(" \n Loss Codition Satisfied Sol 2 \n")

                print("\n\n\n")
                self.save_plots(self.loss_history, self.loss_history_pde, self.loss_history_norm, self.loss_history_orth, self.history_lambda)
                print(f'Orth {self.orth_counter[0]} complete. Total Loss: {loss.item()}')
                print("\n\n\n")

                self.dic[2] = (copy.deepcopy(self.network2), loss)
                self.save_solution(self.dic, self.x_samples, 2)

                return 0

            self.network2.sym = not self.network2.sym

            return 1

        if rm < np.exp(exp_thresh) and rm > 0 and self.orth_counter[0] == 2:
            self.network3.apply(self.weights_init)
            print(" \n Patience Codition Satisfied Sol 3 \n")
            
            if loss < loss_thresh: 

                torch.save(self.network3.state_dict(), model3_path)

                print(" \n Loss Codition Satisfied Sol 3 \n")

                print("\n\n\n")
                self.save_plots(self.loss_history, self.loss_history_pde, self.loss_history_norm, self.loss_history_orth, self.history_lambda)
                print(f'Orth {self.orth_counter[0]} complete. Total Loss: {loss.item()}')
                print("\n\n\n")

                self.dic[3] = (copy.deepcopy(self.network3), loss)
                self.save_solution(self.dic, self.x_samples, 3)

                return 0

            self.network3.sym = not self.network3.sym

            return 1

        if rm < np.exp(exp_thresh) and rm > 0 and self.orth_counter[0] == 3:
            self.network4.apply(self.weights_init)
            print(" \n Patience Codition Satisfied Sol 4 \n")
            
            if loss < loss_thresh: 

                torch.save(self.network4.state_dict(), model4_path)

                print(" \n Loss Codition Satisfied Sol 4 \n")

                print("\n\n\n")
                self.save_plots(self.loss_history, self.loss_history_pde, self.loss_history_norm, self.loss_history_orth, self.history_lambda)
                print(f'Orth {self.orth_counter[0]} complete. Total Loss: {loss.item()}')
                print("\n\n\n")

                self.dic[4] = (copy.deepcopy(self.network4), loss)
                self.save_solution(self.dic, self.x_samples, 4)

                return 0

            self.network4.sym = not self.network4.sym

            return 1



    def select_optimizer(self, lr):
        if self.orth_counter[0] == 0:
            optimizer = torch.optim.LBFGS(self.network1.parameters(), lr=lr,
                                        max_iter=50000,
                                        max_eval=50000,
                                        history_size=800,
                                        line_search_fn="strong_wolfe",
                                        tolerance_change=1.0 * np.finfo(float).eps)
        if self.orth_counter[0] == 1:
            optimizer = torch.optim.LBFGS(self.network2.parameters(), lr=lr,
                                        max_iter=50000,
                                        max_eval=50000,
                                        history_size=800,
                                        line_search_fn="strong_wolfe",
                                        tolerance_change=1.0 * np.finfo(float).eps)
        if self.orth_counter[0] == 2:
            optimizer = torch.optim.LBFGS(self.network3.parameters(), lr=lr,
                                        max_iter=50000,
                                        max_eval=50000,
                                        history_size=800,
                                        line_search_fn="strong_wolfe",
                                        tolerance_change=1.0 * np.finfo(float).eps)
        if self.orth_counter[0] == 3:
            optimizer = torch.optim.LBFGS(self.network4.parameters(), lr=lr,
                                        max_iter=50000,
                                        max_eval=50000,
                                        history_size=800,
                                        line_search_fn="strong_wolfe",
                                        tolerance_change=1.0 * np.finfo(float).eps)
        return optimizer

    ################################################################################################
    def fit(self, num_epochs, verbose=True):


        for epoch in range(self.epoch_beg, 4):
            if verbose: print("################################ ", epoch, " ################################")

            local_loss_history = list()

            lr, w_pde, w_norm, w_orth = self.weights_initialization(self.orth_counter[0])


            optimizer = self.select_optimizer(lr) 


            try:
                for j, (x_train,) in enumerate(self.training_set):
                    def closure() -> float:

                        optimizer.zero_grad()

                        loss, loss_pde, loss_norm, n1, lambda_n, pred_u = self.compute_loss(x_train, w_pde, w_norm, verbose=verbose)

                        loss_orth = self.compute_loss_orth(x_train, pred_u, w_orth)
                        #loss_orth = torch.zeros(1)

                        loss_tot = loss #+ loss_orth

                        self.loss_history_pde.append(loss_pde.item())
                        self.loss_history_norm.append(loss_norm.item())
                        self.loss_history_orth.append(loss_orth.item())
                        self.loss_history.append(loss_tot.item())
                        local_loss_history.append(loss_tot.item())
                        self.history_lambda.append(lambda_n[0].item())

                        loss_tot.backward()
          

                        if not len(self.loss_history) % 300:
                            print(
                                f'################# Epoch {len(self.loss_history)} ################')
                            print('Total Loss: ', loss_tot)
                            print('PDE loss: ', loss_pde)
                            print('Norm loss: ', loss_norm)
                            print('Orth loss: ', loss_orth)
                            print('Lambda: ', lambda_n[0])
                            print('Orth counter: ', self.orth_counter[0])

                            self.save_plots(self.loss_history, self.loss_history_pde, self.loss_history_norm, self.loss_history_orth, self.history_lambda)

                        print("Len loss_history: ", len(self.loss_history),
                              " | orth_counter: ", self.orth_counter[0],
                              " | lambda: ", round(lambda_n[0].item(), 4),
                              " | loss tot: ", round(loss_tot.item(), 6),
                              " | loss pde: ", round(loss_pde.item(), 6),
                              " | loss norm: ", round(loss_norm.item(), 6),
                              " | loss orth: ", round(loss_orth.item(), 6))
                        
                        
                        
                        window = 100
                        if len(local_loss_history) >= window+1:
                            rm = np.mean(np.array(local_loss_history[-window:])-np.array(local_loss_history[-window-1:-1]))
                        else:
                            rm = np.mean(np.array(local_loss_history[1:])-np.array(local_loss_history[:-1]))
                        
                        decreasing = False
                        if len(local_loss_history) > 3:
                            decreasing = (
                                local_loss_history[-3] > local_loss_history[-2]) and (local_loss_history[-2] > local_loss_history[-1])
                      
                        check = self.save_or_switch(loss_tot, decreasing, rm)

                        if check == 0: 
                            print(" \n Optimization Complete \n")
                            raise OptimizationComplete
                        elif check == 1:
                            print("\n Only patience condition met \n Switching Symmetry \n")

                        return loss_tot.item()

                    optimizer.step(closure=closure)
            except OptimizationComplete:
                pass
            
            print("\n Changing Orth Counter \n")
            self.orth_counter[0] += 1

        self.save_solution(self.dic, self.x_samples, 1)
        self.save_solution(self.dic, self.x_samples, 2)
        self.save_solution(self.dic, self.x_samples, 3)
        self.save_solution(self.dic, self.x_samples, 4)


    ################################################################################################
    def save_plots(self, loss_history: list, pde_hist: list, norm_hist: list, orth_hist: list, lambda_hist: list):
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

    def save_solution(self, dic: dict, x_samples: Tensor,  index: int):
        print(f'############# Save Solution_{index} ################')
        n1 = dic[index][0](x_samples)[0]
        pred_u = self.parametric_solutions(x_samples, n1, self.x0, self.xf, 0)
        print(f'Check Normalization: ', torch.sum(pred_u ** 2))

        c_2 = self.n_samples / self.xf / \
            torch.sqrt(torch.dot(torch.sin(x_samples)[
                    :, 0], torch.sin(x_samples)[:, 0]))

        plt.close()
        plt.cla()
        plt.scatter(x_samples.detach(), pred_u.detach(), label='pred u', s=2)

        plt.legend()
        plt.savefig(os.path.join(plot_dir, f'solution_{index}.png'))



x0, xf = -1., 1.
epochs = 4
n_samples = 2000
batch_size = n_samples
neurons = 200

pinn = Pinns(neurons, n_samples, batch_size, x0, xf, load1 = False, load2 = False, load3 = False, load4 = False)

pinn.fit(num_epochs=epochs, verbose=True)
