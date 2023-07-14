import numpy as np
import torch
import random
from torch.utils.data import DataLoader, TensorDataset
import utils._common as _common
from utils._paths import *
import copy
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)

seed = 120
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)


class Pinns:
    def __init__(self, neurons_, n_samples_, batch_size_, x0_, xf_, load1, load2, load3, load4):
        self.n_samples = n_samples_
        self.batch_size = batch_size_
        self.x0 = x0_
        self.xf = xf_
        self.neurons = neurons_

        self.network = _common.NeuralNet(self.neurons)
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
        
    
    # import outsourced functions
    from utils._plotting import save_plots, save_solution, plot_all
    from utils._load import load_states
    from utils._loss import compute_loss, compute_loss_orth, compute_pde_residual, \
        dfx, potential, parametric_solutions
    from utils._save_switch import save_or_switch

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

    def network_initialization(self, int):
        if int == 0:
            lr = float(0.2)
            w_pde = [1]
            w_norm = [1]
            w_orth = [0]
            self.network.init_weights()

        elif int == 1:
            lr = float(0.2)
            w_pde = [1]
            w_norm = [1]
            w_orth = [0.01]
            self.network.init_weights()

        elif int == 2:
            lr = float(0.3)
            w_pde = [1]
            w_norm = [1]
            w_orth = [0.01]
            self.network.init_weights()
            self.network.sym = -1

        elif int == 3:
            lr = float(0.3)
            w_pde = [1]
            w_norm = [1]
            w_orth = [0.02]
            self.network.init_weights()
            self.network.sym = -1

        return lr, w_pde, w_norm, w_orth



    ################################################################################################
    def fit(self):


        for _ in range(self.epoch_beg, 4):

            local_loss_history = list()

            lr, w_pde, w_norm, w_orth = self.network_initialization(self.orth_counter[0])

            optimizer = torch.optim.LBFGS(self.network.parameters(),
                                        lr=lr,
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

                        window = 1000
                        if len(local_loss_history) >= window+1:
                            rm = np.mean(np.array(local_loss_history[-window:])-np.array(local_loss_history[-window-1:-1]))
                        else:
                            rm = np.mean(np.array(local_loss_history[1:])-np.array(local_loss_history[:-1]))
                        
                        decreasing = False
                        if len(local_loss_history) > 3:
                            decreasing = (
                                local_loss_history[-3] > local_loss_history[-2]) and (local_loss_history[-2] > local_loss_history[-1])
                      
                        check = self.save_or_switch(loss_tot, decreasing, rm)

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
                              " | rm: ", round(rm, 6))

                        if check == 0: 
                            raise _common.OptimizationComplete

                        return loss_tot.item()

                    optimizer.step(closure=closure)
            except _common.OptimizationComplete:
                pass
            
            print("\n Changing Orth Counter \n")
            torch.save(self.dic[self.orth_counter[0]][0].state_dict(), model_path[self.orth_counter[0]])
            self.save_solution(self.dic, torch.linspace(self.x0,self.xf, 3000).reshape(-1,1), self.orth_counter[0])
            self.orth_counter[0] += 1
        
        self.plot_all()

    ################################################################################################
    



x0, xf = -6, 6
n_samples = 2000
batch_size = n_samples
neurons = 50

pinn = Pinns(neurons, n_samples, batch_size, x0, xf, load1 = True, load2 = True, load3 = True, load4 = True)

pinn.fit()
