"""
data_generation.py - create the quadratic and cubic datasets.
Run this file directly to draw & save the data figure (Figure 1).
"""
import numpy as np

np.random.seed(42)                        # same data every run
x = np.linspace(-3, 3, 1000).reshape(-1, 1)
noise = np.random.normal(0, 0.5, size=x.shape)   # mean 0, std 0.5

y_quadratic = (x**2 + 2*x + 3) + noise
y_cubic     = (x**3 + 3*(x**2) - x + 2) + noise


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    QUAD_COLOR  = "#2CA02C"   # green
    CUBIC_COLOR = "#9467BD"   # purple

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(x, y_quadratic, color=QUAD_COLOR, alpha=0.5, s=10)
    plt.title("Quadratic Data: $y = x^2 + 2x + 3 + noise$")
    plt.xlabel("x"); plt.ylabel("y")

    plt.subplot(1, 2, 2)
    plt.scatter(x, y_cubic, color=CUBIC_COLOR, alpha=0.5, s=10)
    plt.title("Cubic Data: $y = x^3 + 3x^2 - x + 2 + noise$")
    plt.xlabel("x"); plt.ylabel("y")

    plt.tight_layout()
    plt.savefig("fig1_data.png", dpi=130, bbox_inches="tight")
    print("Saved fig1_data.png")
    plt.show()
