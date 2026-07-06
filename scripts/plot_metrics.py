"""Trace la courbe de loss train/val depuis checkpoints/metrics.csv.

Usage :
    python scripts/plot_metrics.py                       # -> assets/loss_curve.png
    python scripts/plot_metrics.py --csv path --out path

Dépendance : matplotlib (requirements-dev.txt). Le fichier metrics.csv est
produit automatiquement par src/train.py à chaque évaluation.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # rendu fichier, pas de fenêtre
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot train/val loss curves.")
    parser.add_argument("--csv", default="checkpoints/metrics.csv")
    parser.add_argument("--out", default="assets/loss_curve.png")
    args = parser.parse_args()

    iters, train_loss, val_loss = [], [], []
    with open(args.csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            iters.append(int(row["iter"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    ax.plot(iters, train_loss, label="train", color="#7aa2f7", linewidth=2)
    ax.plot(iters, val_loss, label="val", color="#e0616a", linewidth=2)
    ax.set_xlabel("itération")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Mini-GPT — courbe d'apprentissage (tinyshakespeare, char-level)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"courbe sauvegardée : {out} (best val = {min(val_loss):.4f})")


if __name__ == "__main__":
    main()
