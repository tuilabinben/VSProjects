"""
main.py  ---  RUN THIS FILE TO USE THE PROGRAM:   python main.py

It will:
  - check the required libraries and, if any are missing, show how to install them;
  - show a menu so you can choose: quick train, or run all experiments.
"""
import importlib.util
import sys

# ---------- 1) Check libraries ----------
NEEDED = {"torch": "torch", "numpy": "numpy",
          "sklearn": "scikit-learn", "matplotlib": "matplotlib"}
missing = [pip_name for module, pip_name in NEEDED.items()
           if importlib.util.find_spec(module) is None]

if missing:
    print("=" * 55)
    print(" MISSING LIBRARIES:", ", ".join(missing))
    print("=" * 55)
    print(" Install using one of these two ways:")
    print()
    print(" (Recommended) use a virtual environment (venv):")
    print("     python3 -m venv venv")
    print("     source venv/bin/activate")
    print("     pip install -r requirements.txt")
    print()
    print(" (Quick) install directly:")
    print("     pip install --break-system-packages -r requirements.txt")
    print("=" * 55)
    sys.exit(1)

# ---------- 2) Import the rest ----------
import torch
from data_generation import x, y_quadratic, y_cubic
from train import train_model
from lossplot import plot_loss, plot_prediction
from run_all_experiments import run_all

DATASETS = {"1": ("Quadratic", y_quadratic), "2": ("Cubic", y_cubic)}


def train_once():
    print("\nChoose dataset:  1) Quadratic    2) Cubic")
    name, data = DATASETS.get(input("Enter 1 or 2: ").strip(), ("Cubic", y_cubic))
    print(f"Training on {name} ...")
    model, tr, va = train_model(dataset=data)
    print(f"RESULT:  Train={tr[-1]:.5f} | Val={va[-1]:.5f}")
    plot_loss(tr, va, save_path="loss.png")
    with torch.no_grad():
        prediction = model(torch.FloatTensor(x)).numpy()
    plot_prediction(x, data, prediction, save_path="prediction.png")
    print("Saved 2 figures: loss.png and prediction.png")


def menu():
    while True:
        print("\n" + "=" * 30)
        print("   WEEK 6 PROGRAM - MLP")
        print("=" * 30)
        print("  1. Quick train (once)")
        print("  2. Run ALL experiments (build report)")
        print("  0. Exit")
        choice = input("Choose (0/1/2): ").strip()
        if choice == "1":
            train_once()
        elif choice == "2":
            run_all()
        elif choice == "0":
            print("Goodbye!")
            break
        else:
            print("Invalid choice, try again.")


if __name__ == "__main__":
    menu()
