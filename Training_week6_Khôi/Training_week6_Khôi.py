import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
import time

#Import the data
data = pd.read_csv("quadratic_data.csv")

# Create tensor of input features
x = torch.tensor(data['x'].values, dtype = torch.float).view(-1,1)
y = torch.tensor(data['y'].values, dtype = torch.float).view(-1,1)

results = []
hyperparameter = "batch_size"

for i in (8, 32, 128):
    # Set random seed
    torch.manual_seed(42)

    # Set hyperparameters
    lr = 0.01
    batch_size = i
    num_epochs = 300
    train_size = 0.80
    val_size = 0.5
    nodes_per_layers = 16

    print(f"Case {hyperparameter} = {i}")
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
            #self.layer2 = nn.Linear(numHiddenNodes, numHiddenNodes//2)
            #self.layer3 = nn.Linear(numHiddenNodes//2, 1)
            #self.layer3 = nn.Linear(numHiddenNodes//2, numHiddenNodes//4)
            #self.layer4 = nn.Linear(numHiddenNodes//4, 1)
            
            self.relu = nn.ReLU()
            #self.sigmoid = nn.Sigmoid()
            #self.tanh = nn.Tanh()

        def forward(self, x):
            x = self.layer1(x)
            x = self.relu(x)
            #x = self.sigmoid(x)
            #x = self.tanh(x)
            x = self.layer2(x)
            #x = self.relu(x)
            #x = self.layer3(x)
            #x = self.relu(x)
            #x = self.layer4(x) 
            return x

    model = NN_Regression(nodes_per_layers)

    # MSE Loss function + optimizer
    loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr)

    # Training
    n_train = x_train.shape[0]

    train_losses = []   
    val_losses = []      
    start_time = time.time()  

    for epoch in range(num_epochs):
        permutation = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for j in range(0, n_train, batch_size):
            idx = permutation[j : j + batch_size]
            xb, yb = x_train[idx], y_train[idx]

            predictions = model(xb)
            MSE = loss(predictions, yb)
            MSE.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += MSE.item()
            n_batches += 1

        train_losses.append(epoch_loss / n_batches)   

        model.eval()
        with torch.no_grad():
            val_predictions = model(x_val)
            val_MSE = loss(val_predictions, y_val)
        model.train()

        val_losses.append(val_MSE.item())   

        if(epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}]")
            print(f"Train MSE: {epoch_loss / n_batches:.4f}")
            print(f"Val MSE: {val_MSE.item():.4f}")
            print("------------------------------")

    train_time_s = time.time() - start_time   

    # Model evaluation
    model.eval()
    with torch.no_grad():
        predictions = model(x_test)
        test_MSE = loss(predictions, y_test)

    print(f"Test MSE is: {test_MSE.item()}")

    # Print the summary
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final val loss: {val_losses[-1]:.4f}")
    print(f"Best val loss: {min(val_losses):.4f}")
    print(f"Train time: {train_time_s:.2f}\n")

    # Saving results
    results.append({
    f"{hyperparameter}": i,
    "Final train loss": round(train_losses[-1], 4),
    "Final val loss": round(val_losses[-1], 4),
    "Best val loss": round(min(val_losses), 4),
    "Test MSE": round(test_MSE.item(), 4),
    "Train time": round(train_time_s, 4),
    })

df = pd.DataFrame(results)
#df.to_csv(f"quadratic_{hyperparameter}_results.csv", index=False)
print(df)





