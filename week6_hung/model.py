import torch
import torch.nn as nn

#Define the flexible MLP Architecture
class FlexibleRegressionMLP(nn.Module):
    def __init__(self, hidden_nodes=64, num_layers=2, activation_fn=nn.ReLU()):
        super(FlexibleRegressionMLP, self).__init__()
        
        layers = []
        
        # Input layer (Takes 1 feature: x) -> First hidden layer
        layers.append(nn.Linear(1, hidden_nodes))
        layers.append(activation_fn)
        
        # Dynamically add extra hidden layers based on configuration
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_nodes, hidden_nodes))
            layers.append(activation_fn)
            
        # Output layer (Outputs 1 feature: y) - NO activation function for regression
        layers.append(nn.Linear(hidden_nodes, 1))
        
        # Package everything into a Sequential container
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)
