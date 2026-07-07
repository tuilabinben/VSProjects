"""
train.py - train the MLP once and return the model + loss history.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from data_generation import x, y_quadratic, y_cubic
from model import FlexibleRegressionMLP
from lossplot import plot_loss, plot_prediction


def train_model(dataset=y_quadratic, hidden_nodes=64, num_layers=2,
                activation_fn=nn.ReLU(), lr=0.001, epochs=100, seed=42):
    torch.manual_seed(seed)               # same result every run

    X = torch.FloatTensor(x)
    y = torch.FloatTensor(dataset)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = FlexibleRegressionMLP(hidden_nodes, num_layers, activation_fn)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    for _ in range(epochs):
        loss = criterion(model(X_train), y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val)
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

    return model, train_losses, val_losses


if __name__ == "__main__":
    model, tr, va = train_model(dataset=y_quadratic)
    print(f"Final: Train={tr[-1]:.5f}, Val={va[-1]:.5f}")
    plot_loss(tr, va, save_path="loss.png")
    with torch.no_grad():
        prediction = model(torch.FloatTensor(x)).numpy()
    plot_prediction(x, y_quadratic, prediction, save_path="prediction.png")
