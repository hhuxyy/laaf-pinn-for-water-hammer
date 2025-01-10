import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN_LAAF(nn.Module):
    def __init__(self, layers):
        super(DNN_LAAF, self).__init__()
        self.num_layers = len(layers) - 1
        self.layers = nn.ModuleList()

        for l in range(self.num_layers):
            in_features = layers[l]
            out_features = layers[l + 1]
            layer = nn.Linear(in_features, out_features)
            self.layers.append(layer)

            nn.init.xavier_normal_(layer.weight)

        self.a = nn.Parameter(torch.empty(size=(self.num_layers - 1, layers[1])))
        nn.init.xavier_uniform_(self.a.data, gain=1.4)

    def forward(self, x):

        i = 0
        for layer in self.layers[:-1]:
            x = F.tanh(10 * torch.mul(self.a[i, :], layer(x)))
            i += 1

        x = self.layers[-1](x)
        return x
