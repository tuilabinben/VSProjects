"""
model.py - a simple MLP for regression (1 input -> 1 output).
"""
import torch.nn as nn


class FlexibleRegressionMLP(nn.Module):
    def __init__(self, hidden_nodes=64, num_layers=2, activation_fn=nn.ReLU()):
        super().__init__()

        layers = [nn.Linear(1, hidden_nodes), activation_fn]   # input + 1st hidden
        for _ in range(num_layers - 1):                        # extra hidden layers
            layers += [nn.Linear(hidden_nodes, hidden_nodes), activation_fn]
        layers += [nn.Linear(hidden_nodes, 1)]                 # output: no activation

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
