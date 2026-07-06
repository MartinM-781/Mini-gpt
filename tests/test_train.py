"""Tests des utilitaires d'entraînement : LR schedule, groupes d'optimisation."""

import torch

from config import GPTConfig
from model import GPT
from train import PRESETS, configure_optimizer, get_lr


def _cfg(**kwargs):
    base = dict(
        vocab_size=27, block_size=16, n_embd=64, n_head=4, n_layer=2,
        device="cpu", learning_rate=3e-4, min_lr=3e-5,
        warmup_iters=100, max_iters=1000, lr_decay_iters=0,
    )
    base.update(kwargs)
    return GPTConfig(**base)


def test_lr_warmup_starts_low_and_reaches_peak():
    cfg = _cfg()
    assert get_lr(0, cfg) < cfg.learning_rate / 10
    # Fin du warmup : lr == learning_rate (à epsilon près).
    assert abs(get_lr(cfg.warmup_iters - 1, cfg) - cfg.learning_rate) < 1e-9


def test_lr_decays_to_min_lr():
    cfg = _cfg()
    assert abs(get_lr(cfg.max_iters, cfg) - cfg.min_lr) < 1e-9
    assert get_lr(cfg.max_iters + 500, cfg) == cfg.min_lr


def test_lr_monotonically_decreases_after_warmup():
    cfg = _cfg()
    lrs = [get_lr(it, cfg) for it in range(cfg.warmup_iters, cfg.max_iters, 50)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))


def test_optimizer_decay_groups():
    cfg = _cfg()
    model = GPT(cfg)
    optimizer = configure_optimizer(model, cfg)

    assert len(optimizer.param_groups) == 2
    decay, no_decay = optimizer.param_groups
    assert decay["weight_decay"] == cfg.weight_decay
    assert no_decay["weight_decay"] == 0.0
    # Les matrices (>= 2D) sont décayées, les vecteurs (biais, LayerNorm) non.
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() < 2 for p in no_decay["params"])
    # Aucun paramètre oublié.
    n_opt = sum(p.numel() for g in optimizer.param_groups for p in g["params"])
    n_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_opt == n_model


def test_presets_are_valid_configs():
    for name, overrides in PRESETS.items():
        cfg = GPTConfig(**overrides)  # __post_init__ vérifie n_embd % n_head
        assert cfg.max_iters > 0, name
