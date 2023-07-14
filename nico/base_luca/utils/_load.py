import utils._common as _common
import torch
import copy
from utils._paths import *

def load_states(self, load1, load2, load3, load4):

    if load1:
        self.network = _common.NeuralNet(self.neurons)
        self.network.load_state_dict(torch.load(model_path[0]))
        self.network.sym = 1
        self.dic[0] = (copy.deepcopy(self.network), 0)
        self.epoch_beg = 1
        self.orth_counter[0] = 1
        if load2:
            self.network = _common.NeuralNet(self.neurons)
            self.network.load_state_dict(torch.load(model_path[1]))
            self.network.sym = 1
            self.dic[1] = (copy.deepcopy(self.network), 0)
            self.epoch_beg = 2
            self.orth_counter[0] = 2
            if load3:
                self.network = _common.NeuralNet(self.neurons)
                self.network.load_state_dict(torch.load(model_path[2]))
                self.network.sym = -1
                self.dic[2] = (copy.deepcopy(self.network), 0)
                self.epoch_beg = 3
                self.orth_counter[0] = 3
                if load4:
                    self.network = _common.NeuralNet(self.neurons)
                    self.network.load_state_dict(torch.load(model_path[3]))
                    self.network.sym = -1
                    self.dic[3] = (copy.deepcopy(self.network), 0)
                    self.epoch_beg = 4
                    self.orth_counter[0] = 4