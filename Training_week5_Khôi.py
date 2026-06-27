import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

#Import the data
data = pd.read_csv("linear_data.csv")

# Create tensor of input features
x = torch.tensor(data['x'].values, dtype = torch.float).view(-1,1)
y = torch.tensor(data['y'].values, dtype = torch.float).view(-1,1)

# Train-test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size = 0.80,
    test_size = 0.20,
    random_state = 2
)

# Set random seed
torch.manual_seed(42)

# Define the model
model = nn.Sequential(
    nn.Linear(1, 1)
)

# MSE Loss function + optimizer
loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

# Training
num_epochs = 1000
for epoch in range(num_epochs):
    predictions = model(x_train)
    MSE = loss(predictions, y_train)
    MSE.backward()
    optimizer.step()
    optimizer.zero_grad()

    if(epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], MSE Loss: {MSE.item()}")

# Model evaluation
model.eval()
with torch.no_grad():
    predictions = model(x_test)
    test_MSE = loss(predictions, y_test)

print(f"Test MSE is: {test_MSE.item()}")

# Save model
torch.save(model, "model.pth")

# Plotting predictions vs actual values
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, 
            label='Predictions', alpha=0.5, color='blue')

plt.xlabel('Actual Values (y_test)')
plt.ylabel('Predicted Values')

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 
        linestyle='-', color='gray', linewidth=2,
        label="Actual Rent")
plt.legend()
plt.title('Predictions vs Actual Values: y = 2x + 1')
plt.show()





