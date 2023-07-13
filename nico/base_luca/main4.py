import numpy as np
import torch
import random
from torch.autograd import grad
from torch.utils.data import DataLoader, TensorDataset
import utils.common as common
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

        self.network = common.NeuralNet(self.neurons)
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
        
    
    # import outsourced functions
    from utils._plotting import save_plots, save_solution
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



    ################################################################################################
    def fit(self, num_epochs, load1: bool = False, load2: bool = False, load3: bool = False, load4: bool = False, verbose=True):


        for epoch in range(self.epoch_beg, 1):
            if verbose: print("################################ ", epoch, " ################################")

            local_loss_history = list()

            lr, w_pde, w_norm, w_orth = self.weights_initialization(self.orth_counter[0])

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

                        loss, loss_pde, loss_norm, n1, lambda_n, pred_u = self.compute_loss(x_train, w_pde, w_norm, verbose=verbose)

                        loss_orth = torch.zeros(1)# self.compute_loss_orth(x_train, pred_u, w_orth)

                        loss_tot = loss #+ loss_orth

                        self.loss_history_pde.append(loss_pde.item())
                        self.loss_history_norm.append(loss_norm.item())
                        self.loss_history_orth.append(loss_orth.item())
                        self.loss_history.append(loss_tot.item())
                        local_loss_history.append(loss_tot.item())
                        self.history_lambda.append(lambda_n[0].item())
                        
                        #loss_log = loss_tot.log()
                        #loss_log.backward()
                        loss_tot.backward()
          

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
                              " | loss orth: ", round(loss_orth.item(), 6))
                        
                        
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

                        if check == 0: 
                            raise common.OptimizationComplete

                        return loss_tot.item()

                    optimizer.step(closure=closure)
            except common.OptimizationComplete:
                pass
            
            print("\n Changing Orth Counter \n")
            self.orth_counter[0] += 1

        self.save_solution(self.dic, self.x_samples, 0)
        self.save_solution(self.dic, self.x_samples, 1)
        self.save_solution(self.dic, self.x_samples, 2)
        self.save_solution(self.dic, self.x_samples, 3)


    ################################################################################################



x0, xf = -6, 6
epochs = 4
n_samples = 2000
batch_size = n_samples
neurons = 50

pinn = Pinns(neurons, n_samples, batch_size, x0, xf, load1 = False, load2 = False, load3 = False, load4 = False)

pinn.fit(num_epochs=epochs, verbose=True)
