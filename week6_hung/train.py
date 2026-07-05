import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from lossplot import plot_loss, plot_prediction
from data_generation import x, y_quadratic, y_cubic
from model import FlexibleRegressionMLP


def train_model(
    dataset=y_cubic,
    hidden_nodes=64,
    num_layers=2,
    activation_fn=nn.ReLU(),
    learning_rate=0.001,
    epochs=100,
    show_plot=False, #set to True to visualize the training and prediction plots
    verbose=False    #set to True to print training progress
):

    X = torch.FloatTensor(x)
    y = torch.FloatTensor(dataset)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = FlexibleRegressionMLP(
        hidden_nodes=hidden_nodes,
        num_layers=num_layers,
        activation_fn=activation_fn
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):

        prediction = model(X_train)
        loss = criterion(prediction, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train={loss:.5f}, Val={val_loss:.5f}")

    if verbose:
        print("Training Finished!")

    if show_plot:
        plot_loss(train_losses, val_losses)

        with torch.no_grad():
            prediction = model(X).numpy()

        plot_prediction(
            x,
            dataset,
            prediction
        )

    return loss.item(), val_loss.item()

if __name__ == "__main__":
    train_model()