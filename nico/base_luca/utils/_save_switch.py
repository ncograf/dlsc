import torch
import copy
from utils._paths import *

def save_or_switch(self, loss, decreasing, rm):

    self.dic[self.orth_counter[0]] = (copy.deepcopy(self.network), loss)

    exp_thresh = -14 #for the rolling mean rm treshold to satisfy the patience condition

    if self.orth_counter[0] == 0 and loss < 0.1:
        torch.save(self.network.state_dict(), model1_path)

        print("\n\n\n")
        self.save_plots(self.loss_history, self.loss_history_pde, self.loss_history_norm, self.loss_history_orth, self.history_lambda)
        print(f'Orth {self.orth_counter[0]} complete. Total Loss: {loss.item()}')
        print("\n\n\n")

        #self.dic[1] = (copy.deepcopy(self.network), loss)
        #self.save_solution(self.dic, self.x_samples, 1)

        return 0

    if self.orth_counter[0] == 1 and decreasing and loss < 0.1:

        torch.save(self.network.state_dict(), model2_path)
        
        print("\n\n\n")
        self.save_plots(self.loss_history, self.loss_history_pde, self.loss_history_norm, self.loss_history_orth, self.history_lambda)
        print(f'Orth {self.orth_counter[0]} complete. Total Loss: {loss.item()}')
        print("\n\n\n")

        return 0
    
    if self.orth_counter[0] == 2 and decreasing and loss < 1.4:

        torch.save(self.network.state_dict(), model3_path)

        print("\n\n\n")
        self.save_plots(self.loss_history, self.loss_history_pde, self.loss_history_norm, self.loss_history_orth, self.history_lambda)
        print(f'Orth {self.orth_counter[0]} complete. Total Loss: {loss.item()}')
        print("\n\n\n")

        return 0
    
    if self.orth_counter[0] == 3 and decreasing and loss < 0.45:

        torch.save(self.network.state_dict(), model3_path)

        print("\n\n\n")
        self.save_plots(self.loss_history, self.loss_history_pde, self.loss_history_norm, self.loss_history_orth, self.history_lambda)
        print(f'Orth {self.orth_counter[0]} complete. Total Loss: {loss.item()}')
        print("\n\n\n")

        return 0