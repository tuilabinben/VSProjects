import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
import time

import optuna
import matplotlib.pyplot as plt

#Import the data
data = pd.read_csv("D:\\Thinkalpha\\Source code\\Training_week6_Khôi\\cubic_data.csv")

# Create tensor of input features
x = torch.tensor(data['x'].values, dtype = torch.float).view(-1,1)
y = torch.tensor(data['y'].values, dtype = torch.float).view(-1,1)

train_size = 0.80
val_size = 0.5

# Train-test split
x_train, x_temp, y_train, y_temp = train_test_split(
    x, y,
    train_size = train_size,
    random_state = 2
)

x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, 
    train_size = val_size, 
    random_state = 2   
)

# Define the model
class NN_Regression(nn.Module):
    def __init__(self, numHiddenNodes):
        super(NN_Regression, self).__init__()
        self.layer1 = nn.Linear(1, numHiddenNodes) 
        self.layer2 = nn.Linear(numHiddenNodes, 1)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# Bayesian optimization
def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    nodes_per_layer = trial.suggest_int("nodes_per_layer", 4, 64)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])

    torch.manual_seed(42)
    model = NN_Regression(nodes_per_layer)
    loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    n_train = x_train.shape[0]
    for epoch in range(100):
        permutation = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            idx = permutation[i : i + batch_size]
            predictions = model(x_train[idx])
            MSE = loss(predictions, y_train[idx])
            MSE.backward()
            optimizer.step()
            optimizer.zero_grad()

    model.eval()
    with torch.no_grad():
        val_MSE = loss(model(x_val), y_val).item()
    return val_MSE

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=15)

print("Best hyperparameters:", study.best_params)
print("Best validation loss:", study.best_value)

# Training
best = study.best_params
torch.manual_seed(42)
final_model = NN_Regression(best["nodes_per_layer"])
loss = nn.MSELoss()
optimizer = optim.Adam(final_model.parameters(), lr=best["lr"])

n_train = x_train.shape[0]
train_losses = []
val_losses = []
start_time = time.time()

for epoch in range(1000):
    permutation = torch.randperm(n_train)
    epoch_loss = 0.0
    n_batches = 0

    for i in range(0, n_train, best["batch_size"]):
        idx = permutation[i : i + best["batch_size"]]
        predictions = final_model(x_train[idx])
        MSE = loss(predictions, y_train[idx])
        MSE.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += MSE.item()
        n_batches += 1

    train_losses.append(epoch_loss / n_batches)

    final_model.eval()
    with torch.no_grad():
        val_MSE = loss(final_model(x_val), y_val)
    final_model.train()
    val_losses.append(val_MSE.item())

train_time_s = time.time() - start_time

# Print the results
print(f"Final train loss: {train_losses[-1]:.4f}")  
print(f"Final val loss: {val_losses[-1]:.4f}")       
print(f"Best val loss: {min(val_losses):.4f}")         
print(f"Train time: {train_time_s:.2f}\n")

# Model evaluation
final_model.eval()
with torch.no_grad():
    test_MSE = loss(final_model(x_test), y_test)
print(f"Final test MSE: {test_MSE.item():.4f}")

# Get predictions on the test set
final_model.eval()
with torch.no_grad():
    test_predictions = final_model(x_test)

# Convert tensors to numpy for plotting
x_test_np = x_test.numpy().flatten()
y_test_np = y_test.numpy().flatten()
pred_np = test_predictions.numpy().flatten()

# Sort by x so the predicted curve draws as a smooth line, not a zigzag
sort_idx = np.argsort(x_test_np)
x_sorted = x_test_np[sort_idx]
pred_sorted = pred_np[sort_idx]

plt.figure(figsize=(10, 6))
plt.scatter(x_test_np, y_test_np, label="Actual data", alpha=0.5, color="blue")
plt.plot(x_sorted, pred_sorted, label="Model prediction", color="red", linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Model Prediction vs Actual Data")
plt.legend()
plt.show()



