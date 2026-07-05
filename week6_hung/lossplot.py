import matplotlib.pyplot as plt

def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(8, 5))

    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.legend()
    plt.grid(True)

    plt.show()
    
def plot_prediction(X, y, prediction):
    plt.figure(figsize=(8, 5))

    plt.scatter(X, y, s=10, alpha=0.5, label="Data")
    plt.plot(X, prediction, color="red", linewidth=2, label="Prediction")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Model Prediction")
    plt.legend()
    plt.grid(True)

    plt.show()