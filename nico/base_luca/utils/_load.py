import utils.common as common
import torch
import copy
import utils._paths as path

def load_states(self, load1, load2, load3, load4):

    if load1:
        self.network = common.NeuralNet(self.neurons)
        self.network.load_state_dict(torch.load(path.model1_path))
        self.dic[1] = (copy.deepcopy(self.network), 0)
        self.epoch_beg = 1
        self.orth_counter[0] = 1
        if load2:
            self.network = common.NeuralNet(self.neurons)
            self.network.load_state_dict(torch.load(path.model2_path))
            self.dic[2] = (copy.deepcopy(self.network), 0)
            self.epoch_beg = 2
            self.orth_counter[0] = 2
            if load3:
                self.network = common.NeuralNet(self.neurons)
                self.network.load_state_dict(torch.load(path.model3_path))
                self.dic[3] = (copy.deepcopy(self.network), 0)
                epoch_beg = 3
                self.orth_counter[0] = 3
                if load4:
                    self.network = common.NeuralNet(self.neurons)
                    self.network.load_state_dict(torch.load(path.model4_path))
                    self.dic[4] = (copy.deepcopy(self.network), 0)
                    self.epoch_beg = 4
                    self.orth_counter[0] = 4