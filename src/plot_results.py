"""Figures from the training results: predicted vs actual per model, MLP loss curve.

Usage: python src/plot_results.py   ->  figures/pred_vs_actual.png, figures/mlp_loss.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "figures"
plt.style.use(Path(__file__).with_name("sage.mplstyle"))
NAMES = {"baseline": "Linear regression", "knn": "k-NN", "rf": "Random Forest", "mlp": "MLP"}
ACCENT, DARK, SAGE = "#C98B4E", "#2F4A3E", "#6B9A80"


def pred_vs_actual():
    keys = [k for k in NAMES if (RES / f"pred_{k}.npz").exists()]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 3.6), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, keys):
        d = np.load(RES / f"pred_{k}.npz")
        r2 = json.loads((RES / f"{k}.json").read_text())["test"]["r2_log"]
        ax.hexbin(d["y"], d["pred"], gridsize=60, cmap="Greens", mincnt=2, bins="log", linewidths=0, extent=(9, 16, 9, 16))
        ax.plot([9, 16], [9, 16], color=ACCENT, lw=1.8, ls="--")
        ax.set_title(f"{NAMES[k]}  (R² {r2:.3f})", fontsize=12)
        ax.set_xlabel("actual log value")
        ax.grid(False)
    axes[0].set_ylabel("predicted log value")
    fig.tight_layout()
    fig.savefig(FIG / "pred_vs_actual.png")
    plt.close(fig)


def mlp_loss():
    p = RES / "mlp.json"
    if not p.exists():
        return
    m = json.loads(p.read_text())
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    loss = m["loss_curve"]
    ax.plot(range(1, len(loss) + 1), loss, color=DARK, label="training loss (squared error / 2)")
    ax.set_ylim(min(loss) * 0.95, max(loss[1:]) * 1.15)  # epoch 1 is far higher; zoom on the rest
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    if m.get("validation_scores"):
        ax2 = ax.twinx()
        ax2.plot(range(1, len(m["validation_scores"]) + 1), m["validation_scores"], color=ACCENT, label="validation R² (early stopping)")
        ax2.set_ylabel("validation R²"); ax2.grid(False)
        ax2.spines["right"].set_visible(True)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=9)
    ax.set_title("MLP training curve (best configuration)", fontsize=12)
    fig.savefig(FIG / "mlp_loss.png")
    plt.close(fig)


if __name__ == "__main__":
    pred_vs_actual()
    mlp_loss()
