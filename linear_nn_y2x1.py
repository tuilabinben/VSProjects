import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split  # type: ignore
import matplotlib.pyplot as plt

# ── 1. Generate synthetic data ─────────────────────────────────────────────
np.random.seed(42)
n = 100
x_raw = np.random.uniform(-10, 10, n)
noise  = np.random.normal(0, 1, n)
y_raw  = 2 * x_raw + 1 + noise

df = pd.DataFrame({'x': x_raw, 'y': y_raw})
df.to_csv('linear_data.csv', index=False)
print("✓ linear_data.csv created")

# ── 2. Load data ───────────────────────────────────────────────────────────
data = pd.read_csv("linear_data.csv")

# ── 3. Build tensors ───────────────────────────────────────────────────────
x = torch.tensor(data['x'].values, dtype=torch.float).view(-1, 1)
y = torch.tensor(data['y'].values, dtype=torch.float).view(-1, 1)

# ── 4. Train-test split ────────────────────────────────────────────────────
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.80,
    test_size=0.20,
    random_state=2
)

# ── 5. Define model ────────────────────────────────────────────────────────
torch.manual_seed(42)

model = nn.Sequential(
    nn.Linear(1, 1)
)

# ── 6. Loss function + optimizer ───────────────────────────────────────────
loss_fn   = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ── 7. Training loop ───────────────────────────────────────────────────────
num_epochs = 1000
for epoch in range(num_epochs):
    predictions = model(x_train)
    MSE = loss_fn(predictions, y_train)

    MSE.backward()
    optimizer.step()
    optimizer.zero_grad()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], MSE Loss: {MSE.item():.6f}")

# ── 8. Evaluation ──────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    predictions = model(x_test)
    test_MSE = loss_fn(predictions, y_test)

print(f"\nTest MSE is: {test_MSE.item():.6f}")

# ── 9. Save model ──────────────────────────────────────────────────────────
torch.save(model, "model.pth")
print("✓ model.pth saved")

# ── 10. Plot predictions vs actual ─────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')

fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(y_test, predictions,
           color='#E53935',
           alpha=0.7,
           s=60,
           edgecolors='white',
           linewidths=0.5,
           label='Predictions',
           zorder=3)

ax.plot([min(y_test), max(y_test)],
        [min(y_test), max(y_test)],
        color="#2E82CC",
        linewidth=2,
        linestyle='--',
        label='Perfect Prediction',
        zorder=2)

ax.set_xlabel('Actual Values (y_test)', fontsize=12)
ax.set_ylabel('Predicted Values', fontsize=12)
ax.set_title('Predictions vs Actual Values: y = 2x + 1', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig('predictions_plot.png', dpi=150, bbox_inches='tight')
plt.show()