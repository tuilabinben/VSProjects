import torch
import torch.nn as nn

x = torch.arange(-10, 11, dtype=torch.float32).reshape(-1,1)
y = 2*x + 1

model = nn.Linear(1,1)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 1000

for epoch in range(epochs):
    predictions = model(x)
    loss = criterion(predictions, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss = {loss.item():.6f}")

print("\nTraining Finished\n")

print("Weight:")
print(model.weight)

print("\nBias:")
print(model.bias)

print("\nPrediction for x = 20")

test = torch.tensor([[20.0]])

print(model(test))
