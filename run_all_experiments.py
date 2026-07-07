"""
run_all_experiments.py - run ALL experiments; create every figure + number for the report.
Run:  python run_all_experiments.py   (figures are saved into the figures/ folder)
"""
import os
import torch
import torch.nn as nn

from data_generation import x, y_quadratic, y_cubic
from train import train_model
from lossplot import plot_loss, plot_prediction, plot_combined


def _predict(model):
    with torch.no_grad():
        return model(torch.FloatTensor(x)).numpy()


def run_all(fig_dir="figures", results_file="results.txt"):
    os.makedirs(fig_dir, exist_ok=True)
    datasets = {"Quadratic": y_quadratic, "Cubic": y_cubic}
    lines = []

    def log(t):
        print(t); lines.append(t)

    log("===== BASELINE =====")
    for name, data in datasets.items():
        model, tr, va = train_model(dataset=data)
        plot_loss(tr, va, save_path=f"{fig_dir}/base_loss_{name}.png")
        plot_prediction(x, data, _predict(model), save_path=f"{fig_dir}/base_pred_{name}.png")
        log(f"[{name}] FINAL Train={tr[-1]:.5f} Val={va[-1]:.5f}")

    log("\n===== EXPERIMENTS =====")
    activations = {"ReLU": nn.ReLU(), "Tanh": nn.Tanh(), "Sigmoid": nn.Sigmoid()}
    nodes = {"Quadratic": [16, 32, 64, 128], "Cubic": [32, 64, 128, 512]}

    def one(name, data, tag, label, **kw):
        model, tr, va = train_model(dataset=data, **kw)
        plot_combined(tr, va, x, data, _predict(model), save_path=f"{fig_dir}/{tag}.png")
        log(f"  {label:12s}| Train={tr[-1]:.5f} | Val={va[-1]:.5f}")

    for name, data in datasets.items():
        log(f"\n----- {name} -----")
        for a, fn in activations.items():
            one(name, data, f"act_{name}_{a}", a, activation_fn=fn)
        for lr in [0.1, 0.01, 0.001, 0.0001]:
            one(name, data, f"lr_{name}_{lr}", f"LR={lr}", lr=lr)
        for nl in [1, 2, 3, 4]:
            one(name, data, f"lay_{name}_{nl}", f"{nl} layers", num_layers=nl)
        for nd in nodes[name]:
            one(name, data, f"nod_{name}_{nd}", f"{nd} nodes", hidden_nodes=nd)
        for ep in [50, 100, 300, 800]:
            one(name, data, f"epo_{name}_{ep}", f"{ep} epochs", epochs=ep)

    with open(results_file, "w") as fh:
        fh.write("\n".join(lines))
    print(f"\nDone! Images in '{fig_dir}/', numbers in '{results_file}'.")


if __name__ == "__main__":
    run_all()
