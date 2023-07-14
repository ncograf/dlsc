import torch
import copy
from utils._paths import *

def save_or_switch(self, loss, decreasing, rm):

    if (not self.dic[self.orth_counter[0]][0]) or loss < self.dic[self.orth_counter[0]][1]:
        self.dic[self.orth_counter[0]] = (copy.deepcopy(self.network), loss)

    loss_thresh = [0.8, 0.1, 0.8, 0.8, 0.1]

    if ((self.orth_counter[0] >= 1 and decreasing) or self.orth_counter[0] == 1) and loss < loss_thresh[self.orth_counter[0]]:

        torch.save(self.dic[self.orth_counter[0]][0], model_path[self.orth_counter[0]])
        self.save_solution(self.dic, self.x_samples, self.orth_counter[0])
 
        print("\n\n\n")
        self.save_plots(self.loss_history, self.loss_history_pde, self.loss_history_norm, self.loss_history_orth, self.history_lambda)
        print(f'Orth {self.orth_counter[0]} complete. Total Loss: {loss.item()}')
        print("\n\n\n")

        return 0