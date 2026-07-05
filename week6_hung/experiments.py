import torch.nn as nn

from train import train_model
from data_generation import y_quadratic, y_cubic

# ==========================================
# Choose Dataset
# ==========================================

dataset = y_quadratic      # Change to y_cubic if needed


# ==========================================
# Activation Function Experiment
# ==========================================

print("\n===== Activation Function =====")

for activation in [nn.ReLU(), nn.Tanh(), nn.Sigmoid()]:

    train_loss, val_loss = train_model(
        dataset=dataset,
        activation_fn=activation
    )

    print(
        f"{activation.__class__.__name__:8} | "
        f"Train Loss: {train_loss:.5f} | "
        f"Validation Loss: {val_loss:.5f}"
    )


# ==========================================
# Learning Rate Experiment
# ==========================================

print("\n===== Learning Rate =====")

for lr in [0.1, 0.01, 0.001, 0.0001]:

    train_loss, val_loss = train_model(
        dataset=dataset,
        learning_rate=lr
    )

    print(
        f"LR = {lr:<7} | "
        f"Train Loss: {train_loss:.5f} | "
        f"Validation Loss: {val_loss:.5f}"
    )


# ==========================================
# Hidden Layers Experiment
# ==========================================

print("\n===== Hidden Layers =====")

for layers in [1, 2, 3]:

    train_loss, val_loss = train_model(
        dataset=dataset,
        num_layers=layers
    )

    print(
        f"{layers} Layers | "
        f"Train Loss: {train_loss:.5f} | "
        f"Validation Loss: {val_loss:.5f}"
    )


# ==========================================
# Hidden Nodes Experiment
# ==========================================

print("\n===== Hidden Nodes =====")

for nodes in [16, 32, 64, 128]:

    train_loss, val_loss = train_model(
        dataset=dataset,
        hidden_nodes=nodes
    )

    print(
        f"{nodes:3} Nodes | "
        f"Train Loss: {train_loss:.5f} | "
        f"Validation Loss: {val_loss:.5f}"
    )


# ==========================================
# Epoch Experiment
# ==========================================

print("\n===== Epochs =====")

for epochs in [50, 100, 200]:

    train_loss, val_loss = train_model(
        dataset=dataset,
        epochs=epochs
    )

    print(
        f"{epochs:3} Epochs | "
        f"Train Loss: {train_loss:.5f} | "
        f"Validation Loss: {val_loss:.5f}"
    )