import torch
import torch.nn as nn
import torch.nn.functional as F


class mMLP(nn.Module):
    def __init__(self, layers):
        super(mMLP, self).__init__()
        self.num_layers = len(layers) - 1
        self.layers = nn.ModuleList()

        # hidden layer
        for l in range(self.num_layers):
            in_features = layers[l] 
            out_features = layers[l + 1]
            layer = nn.Linear(in_features, out_features)
            self.layers.append(layer)

            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

        # encoder parameters
        self.encoder_weights_1 = nn.Parameter(torch.Tensor(layers[1], layers[0]))
        self.encoder_biases_1 = nn.Parameter(torch.Tensor(layers[1]))
        self.encoder_weights_2 = nn.Parameter(torch.Tensor(layers[1], layers[0]))
        self.encoder_biases_2 = nn.Parameter(torch.Tensor(layers[1]))

        nn.init.xavier_normal_(self.encoder_weights_1)
        nn.init.xavier_normal_(self.encoder_biases_1.unsqueeze(0))
        nn.init.xavier_normal_(self.encoder_weights_2)
        nn.init.xavier_normal_(self.encoder_biases_2.unsqueeze(0))

    def forward(self, x):
        encoder_1 = torch.tanh(torch.matmul(x, self.encoder_weights_1.t()) + self.encoder_biases_1)
        encoder_2 = torch.tanh(torch.matmul(x, self.encoder_weights_2.t()) + self.encoder_biases_2)

        x = self.layers[0](x)

        for layer in self.layers[1:-1]:
            W = layer.weight
            b = layer.bias
            x = torch.mul(F.tanh(torch.matmul(x, W) + b), encoder_1) + \
                torch.mul(1 - F.tanh(torch.matmul(x, W) + b), encoder_2)

        x = self.layers[-1](x)
        return x
