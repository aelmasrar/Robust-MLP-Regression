"""MLP commun aux quatre fonctions de coût."""
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, hidden_layers=10):
        super().__init__()
        if hidden_layers < 1:
            raise ValueError('hidden_layers doit être >= 1')
        layers = []
        d = input_dim
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(d, hidden_dim), nn.ReLU()])
            d = hidden_dim
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
