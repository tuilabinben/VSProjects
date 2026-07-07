"""
lossplot.py - drawing helpers.
Change all plot colors by editing the COLORS dict below.
"""
import matplotlib
matplotlib.use("Agg")               # save figures without opening a window
import matplotlib.pyplot as plt

COLORS = {
    "train": "#2CA02C",   # green   - Training Loss
    "val":   "#9467BD",   # purple  - Validation Loss
    "data":  "#17BECF",   # teal    - data points
    "pred":  "#C71585",   # magenta - prediction line
}


def _loss(ax, train_losses, val_losses):
    ax.plot(train_losses, color=COLORS["train"], label="Training Loss")
    ax.plot(val_losses,   color=COLORS["val"],   label="Validation Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Training History")
    ax.legend(); ax.grid(True)


def _pred(ax, x, y, prediction):
    ax.scatter(x, y, s=10, alpha=0.5, color=COLORS["data"], label="Data")
    ax.plot(x, prediction, color=COLORS["pred"], linewidth=2, label="Prediction")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title("Model Prediction")
    ax.legend(); ax.grid(True)


def plot_loss(train_losses, val_losses, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 5)); _loss(ax, train_losses, val_losses); _save(fig, save_path)


def plot_prediction(x, y, prediction, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 5)); _pred(ax, x, y, prediction); _save(fig, save_path)


def plot_combined(train_losses, val_losses, x, y, prediction, save_path=None):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    _loss(a1, train_losses, val_losses); _pred(a2, x, y, prediction); _save(fig, save_path)


def _save(fig, save_path):
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
