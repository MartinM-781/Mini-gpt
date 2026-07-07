"""train.py - boucle d'entraînement du Mini-GPT.

Usage standard :
    cd C:\\dev\\mini-gpt
    python src/train.py --preset cpu-small      # premier run rapide sur CPU
    python src/train.py --preset cpu-medium     # ~30 min sur CPU, meilleure loss
    python src/train.py --preset gpu            # config complète (GPU conseillé)

Suppose qu'un corpus existe à `data/input.txt`. Sauvegarde :
- checkpoints/ckpt.pt      : poids + optimiseur + config + iter
- checkpoints/vocab.json   : tokenizer (nécessaire à sample.py)
- checkpoints/metrics.csv  : iter, train_loss, val_loss, lr, temps écoulé
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time

import torch

from config import GPTConfig
from dataset import CharDataset
from model import GPT
from tokenizer import CharTokenizer

# Presets : un compromis taille/temps par machine cible. Les presets cpu-*
# rallongent le warmup à 200 iters ; gpu garde les defaults de GPTConfig.
PRESETS: dict[str, dict] = {
    # Petit modèle, premier run rapide sur CPU portable. val_loss ≈ 1.9.
    "cpu-small": dict(
        n_layer=4, n_head=4, n_embd=128, block_size=64,
        batch_size=16, max_iters=4000, eval_interval=400, eval_iters=50,
        warmup_iters=200, device="cpu",
    ),
    # Architecture complète mais contexte réduit, ~30 min sur CPU portable.
    # val_loss 1.88 (seedé : deux runs identiques donnent le même résultat).
    "cpu-medium": dict(
        n_layer=6, n_head=6, n_embd=192, block_size=96,
        batch_size=12, max_iters=3000, eval_interval=300, eval_iters=50,
        warmup_iters=200, device="cpu",
    ),
    # Config complète (defaults de GPTConfig). Quelques minutes sur GPU.
    "gpu": dict(),
}


def resolve_device(requested: str) -> str:
    """Downgrade cuda -> cpu si indisponible ; ne force JAMAIS cuda.

    Un utilisateur qui demande explicitement --device cpu sur une machine
    équipée d'un GPU doit être respecté (debug, benchmark, VRAM occupée).
    """
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def get_lr(it: int, config: GPTConfig) -> float:
    """LR schedule : warmup linéaire puis cosine decay vers min_lr.

    - iter 0 .. warmup_iters       : montée linéaire 0 -> learning_rate
    - iter warmup_iters .. decay   : descente en cosinus vers min_lr
    - iter > decay                  : reste à min_lr

    Intuition : steps prudents au début (gradients bruyants car poids
    aléatoires), grands pas en milieu d'entraînement, steps fins en fin
    (raffinement, pas d'oscillation).
    """
    if it < config.warmup_iters:
        return config.learning_rate * (it + 1) / config.warmup_iters
    decay_iters = config.lr_decay_iters if config.lr_decay_iters > 0 else config.max_iters
    if it > decay_iters:
        return config.min_lr
    decay_ratio = (it - config.warmup_iters) / max(1, decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


def configure_optimizer(
    model: GPT, config: GPTConfig
) -> torch.optim.Optimizer:
    """AdamW avec weight decay sélectif (GPT-2 style).

    On applique weight_decay UNIQUEMENT aux paramètres 2D (matrices de
    transformation : Linear.weight, Embedding.weight). Les paramètres 1D
    (biais, LayerNorm gamma/beta) reçoivent decay = 0.

    Raison : la L2 a du sens sur les matrices (on veut les garder petites
    pour éviter le surapprentissage), pas sur les biais ni sur les
    facteurs de scaling LayerNorm.
    """
    decay_params, no_decay_params = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay_params if p.dim() >= 2 else no_decay_params).append(p)

    param_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    n_decay = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)
    print(
        f"optim : {len(decay_params)} tenseurs decayes ({n_decay:,} params), "
        f"{len(no_decay_params)} tenseurs no-decay ({n_no_decay:,} params)"
    )

    return torch.optim.AdamW(
        param_groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )


@torch.no_grad()
def estimate_loss(
    model: GPT, dataset: CharDataset, config: GPTConfig
) -> dict[str, float]:
    """Moyenne la loss sur eval_iters batches, pour train et val.

    Critique : passer en mode eval() pour désactiver dropout, puis remettre
    train() à la fin (sinon le dropout reste OFF dans la suite et on
    sur-apprend bêtement).
    """
    out: dict[str, float] = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(config.eval_iters)
        for i in range(config.eval_iters):
            x, y = dataset.get_batch(split)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main(config: GPTConfig | None = None) -> None:
    if config is None:
        config = GPTConfig()

    # ----- Device : respecter le choix explicite, downgrade si cuda absent -----
    config.device = resolve_device(config.device)
    print(f"device = {config.device}")

    # ----- Seed -----
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # ----- Corpus + tokenizer + dataset -----
    if not os.path.exists(config.data_path):
        raise FileNotFoundError(
            f"corpus introuvable : {config.data_path}\n"
            f"  Place un fichier .txt à cet endroit (ex: tinyshakespeare)."
        )
    with open(config.data_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"corpus : {len(text):,} caracteres")

    tokenizer = CharTokenizer.from_text(text)
    config.vocab_size = tokenizer.vocab_size
    print(f"vocab_size = {tokenizer.vocab_size}")

    dataset = CharDataset(config.data_path, tokenizer, config)
    print(
        f"train = {len(dataset.train_data):,} tokens, "
        f"val = {len(dataset.val_data):,} tokens"
    )

    # ----- Modèle -----
    model = GPT(config).to(config.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params = {n_params:,}")

    # ----- Optimiseur -----
    optimizer = configure_optimizer(model, config)

    # ----- Préparation du dossier de checkpoints + sauvegarde du vocab -----
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    tokenizer.save(os.path.join(config.checkpoint_dir, "vocab.json"))
    ckpt_path = os.path.join(config.checkpoint_dir, "ckpt.pt")

    # ----- Log CSV des métriques (une ligne par évaluation) -----
    metrics_path = os.path.join(config.checkpoint_dir, "metrics.csv")
    metrics_file = open(metrics_path, "w", newline="", encoding="utf-8")
    metrics = csv.writer(metrics_file)
    metrics.writerow(["iter", "train_loss", "val_loss", "lr", "elapsed_s"])

    # ----- Boucle d'entraînement -----
    print(
        f"\n--- training {config.max_iters} iters, "
        f"batch {config.batch_size}, block_size {config.block_size} ---"
    )
    t0 = time.time()
    best_val = float("inf")

    for it in range(config.max_iters):
        # Mise à jour du learning rate (warmup + cosine).
        lr = get_lr(it, config)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Évaluation périodique (et au tout dernier step).
        if it % config.eval_interval == 0 or it == config.max_iters - 1:
            losses = estimate_loss(model, dataset, config)
            dt = time.time() - t0
            print(
                f"iter {it:5d} | train {losses['train']:.4f} "
                f"| val {losses['val']:.4f} | lr {lr:.2e} | {dt:.1f}s",
                flush=True,
            )
            metrics.writerow(
                [it, f"{losses['train']:.4f}", f"{losses['val']:.4f}",
                 f"{lr:.6e}", f"{dt:.1f}"]
            )
            metrics_file.flush()
            # On ne sauvegarde que si la val s'améliore (best checkpoint).
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": vars(config),
                        "iter": it,
                        "val_loss": losses["val"],
                    },
                    ckpt_path,
                )

        # Step d'entraînement.
        x, y = dataset.get_batch("train")
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Clipping de la norme L2 globale du gradient (anti-explosion).
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

    metrics_file.close()
    print(f"\nfini en {time.time() - t0:.1f}s | best val loss = {best_val:.4f}")
    print(f"checkpoint : {ckpt_path}")
    print(f"metriques  : {metrics_path}")


def parse_args() -> GPTConfig:
    """Construit la config depuis --preset + overrides CLI éventuels."""
    parser = argparse.ArgumentParser(description="Train a Mini-GPT from scratch.")
    parser.add_argument(
        "--preset", choices=sorted(PRESETS), default="cpu-small",
        help="Config prédéfinie (défaut : cpu-small).",
    )
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = GPTConfig(**PRESETS[args.preset])
    # Les overrides CLI priment sur le preset.
    for field in ("max_iters", "device", "data_path", "seed"):
        value = getattr(args, field)
        if value is not None:
            setattr(config, field, value)
    return config


if __name__ == "__main__":
    main(parse_args())
