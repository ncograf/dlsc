import torch
import copy
from utils._paths import *
import numpy as np

def save_or_switch(self, loss, local_loss_history, E):

    if (not self.dic[self.orth_counter[0]][0]) or loss < self.dic[self.orth_counter[0]][1]:
        self.dic[self.orth_counter[0]] = (copy.deepcopy(self.network), loss)

    loss_thresh = [10, 10, 10, 35]
    loss_thresh = loss_thresh[self.orth_counter[0]]
    
    rm_thresh = 1e-4

    window = 1000
    if len(local_loss_history) >= window+1:
        rm = np.mean(np.array(local_loss_history[-window:])-np.array(local_loss_history[-window-1:-1]))
    else:
        rm = np.mean(np.array(local_loss_history[1:]) - np.array(local_loss_history[:-1]))
    
    decreasing = False
    if len(local_loss_history) > 3:
        decreasing = rm >= 0 and rm < rm_thresh
        
    loss_condition = loss < loss_thresh

    if decreasing:
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
            
        else: 
            self.network.sym  *= -1
            print(f"\n\nPatinece condition satisfied but loss too hight switch symmetry to\
                {self.network.sym}")
            return True, rm

    return False, rm