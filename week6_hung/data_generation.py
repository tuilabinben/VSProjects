import numpy as np
import matplotlib.pyplot as plt

# 1. Set seed for reproducibility
np.random.seed(42)

# 2. Generate 1000 input features (x) between -3 and 3
x = np.linspace(-3, 3, 1000).reshape(-1, 1)

# 3. Generate random Gaussian noise
noise = np.random.normal(0, 0.5, size=x.shape) # mean=0, std_dev=0.5

# 4. Calculate targets (y) based on the formulas + noise
y_quadratic = (x**2 + 2*x + 3) + noise
y_cubic = (x**3 + 3*(x**2) - x + 2) + noise

if __name__ == "__main__":
    # Plot the generated data
    plt.figure(figsize=(12, 5))

    # Plot Quadratic Data
    plt.subplot(1, 2, 1)
    plt.scatter(x, y_quadratic, color='blue', alpha=0.5, s=10)
    plt.title('Quadratic Data: $y = x^2 + 2x + 3 + noise$')
    plt.xlabel('x')
    plt.ylabel('y')

    # Plot Cubic Data
    plt.subplot(1, 2, 2)
    plt.scatter(x, y_cubic, color='red', alpha=0.5, s=10)
    plt.title('Cubic Data: $y = x^3 + 3x^2 - x + 2 + noise$')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.tight_layout()
    plt.show()