import torch
from torch import nn


def weights_init(m):
    if isinstance(m, nn.Linear) and m.weight.shape[0] != 1:
        torch.nn.init.xavier_uniform(m.weight.data)
        
class Sin(torch.nn.Module):
    @staticmethod
    def forward(input):
        return torch.sin(input)

class NeuralNet(nn.Module):

    def __init__(self, n_hidden_layers, neurons):
        super(NeuralNet, self).__init__()
        self.neurons = neurons
        self.n_hidden_layers = n_hidden_layers
        # Activation function
        self.activation = Sin()
        
        # determine symmetry
        self.sym = -1
        
        self.layer_I = nn.Linear(1,1)
        self.input_layer = nn.Linear(2, self.neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons + 1, self.neurons) for _ in range(n_hidden_layers)])
        self.output_layer = nn.Linear(self.neurons + 1, 1)

    def forward(self, x):
        x = x.reshape(-1,1)
        lambda_ = self.layer_I(torch.ones_like(x)).reshape(-1,1)
        lambda_ = lambda_

        x_plus = x
        x_plus = torch.cat((x_plus, lambda_), dim=1)
        x_plus = self.activation(self.input_layer(x_plus))
        for _, l in enumerate(self.hidden_layers):
            x_plus = torch.cat((x_plus, lambda_), dim=1)
            x_plus = self.activation(l(x_plus))

        x_minus = -x
        x_minus = torch.cat((x_minus, lambda_), dim=1)
        x_minus = self.activation(self.input_layer(x_minus))
        for _, l in enumerate(self.hidden_layers):
            x_minus = torch.cat((x_minus, lambda_), dim=1)
            x_minus = self.activation(l(x_minus))
        
        hub_layer = torch.cat((x_plus + self.sym * x_minus, lambda_), dim=1)
        return self.output_layer(hub_layer), lambda_

    def init_xavier(self):
        def init_weights(m):
            # dont reset the lambda layer
            if type(m) == nn.Linear and m.weight.shape[0] != 1:
                torch.nn.init.xavier_uniform_(m.weight.data)

        self.apply(init_weights)
