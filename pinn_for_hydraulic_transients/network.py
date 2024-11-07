import torch
import torch.nn as nn
from collections import OrderedDict


class Network(nn.Module):
    def __init__(self,layers):
        super(Network, self).__init__()
        num_layers = len(layers) - 1

        input_layer = nn.Linear(layers[0], layers[1])
        nn.init.xavier_normal_(input_layer.weight, gain=1)
        # nn.init.kaiming_normal_(input_layer.weight, mode='fan_in', nonlinearity='relu')
        layer = [('input', input_layer), ('input_activation', nn.Tanh())]

        for i in range(1, num_layers-1):
            hidden = nn.Linear(layers[i], layers[i+1])
            nn.init.xavier_normal_(hidden.weight, gain=1)
            # nn.init.kaiming_normal_(hidden.weight, mode='fan_in', nonlinearity='relu')
            layer.append(('hidden_%d' % i, hidden))
            layer.append(('activation_%d' % i, nn.Tanh()))

        out_layer = nn.Linear(layers[-2], layers[-1])
        nn.init.xavier_normal_(out_layer.weight, gain=1)
        # nn.init.kaiming_normal_(out_layer.weight, mode='fan_in', nonlinearity='relu')
        layer.append(('output', out_layer))

        layerDict = OrderedDict(layer)
        self.layers = nn.Sequential(layerDict)

    def forward(self, x):
        # X = torch.nn.functional.normalize(x, dim=0)
        return self.layers(x)
