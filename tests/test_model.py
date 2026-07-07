"""Tests du modèle GPT : causalité, shapes, init, weight tying, génération."""

import math

import pytest
import torch

from config import GPTConfig
from model import GPT, Block, MultiHeadSelfAttention


@pytest.fixture()
def cfg():
    return GPTConfig(
        vocab_size=27, block_size=16, n_embd=64, n_head=4, n_layer=2,
        dropout=0.0, device="cpu",
    )


@pytest.fixture()
def model(cfg):
    torch.manual_seed(0)
    return GPT(cfg).eval()


def _assert_causal(module, B=2, T=8, C=64):
    """Modifier le dernier token ne doit pas changer les sorties précédentes."""
    torch.manual_seed(0)
    x1 = torch.randn(B, T, C)
    x2 = x1.clone()
    x2[:, -1, :] = torch.randn(B, C)
    with torch.no_grad():
        y1, y2 = module(x1), module(x2)
    assert torch.allclose(y1[:, :-1], y2[:, :-1], atol=1e-6)


def test_attention_is_causal(cfg):
    _assert_causal(MultiHeadSelfAttention(cfg).eval())


def test_block_is_causal(cfg):
    _assert_causal(Block(cfg).eval())


def test_forward_shapes_and_optional_loss(model, cfg):
    B, T = 2, 8
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))

    logits, loss = model(idx)
    assert logits.shape == (B, T, cfg.vocab_size)
    assert loss is None

    logits, loss = model(idx, targets)
    assert logits.shape == (B, T, cfg.vocab_size)
    assert loss.dim() == 0


def test_initial_loss_close_to_uniform(model, cfg):
    # À l'init, le modèle doit être ~uniforme sur le vocab : loss ~ ln(V).
    idx = torch.randint(0, cfg.vocab_size, (4, 16))
    targets = torch.randint(0, cfg.vocab_size, (4, 16))
    _, loss = model(idx, targets)
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 0.5


def test_weight_tying_shares_memory(model):
    assert model.lm_head.weight.data_ptr() == model.token_embed.weight.data_ptr()


def test_sequence_longer_than_block_size_raises(model, cfg):
    idx = torch.randint(0, cfg.vocab_size, (1, cfg.block_size + 1))
    with pytest.raises(AssertionError):
        model(idx)


def test_generate_output_shape(model):
    start = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(start, max_new_tokens=10, top_k=5)
    assert out.shape == (1, 11)
    # Le prompt doit être préservé en tête de séquence.
    assert torch.equal(out[:, :1], start)


def test_generate_beyond_block_size(model, cfg):
    # La génération doit tronquer le contexte et ne jamais planter,
    # même quand la séquence dépasse block_size.
    start = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(start, max_new_tokens=cfg.block_size * 2)
    assert out.shape == (1, 1 + cfg.block_size * 2)


def test_generate_ids_within_vocab(model, cfg):
    start = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(start, max_new_tokens=20)
    assert out.min() >= 0
    assert out.max() < cfg.vocab_size


def test_generate_stream_yields_each_token(model, cfg):
    start = torch.zeros((2, 3), dtype=torch.long)
    tokens = list(model.generate_stream(start, max_new_tokens=5, top_k=3))
    assert len(tokens) == 5
    for t in tokens:
        assert t.shape == (2, 1)
        assert 0 <= t.min() and t.max() < cfg.vocab_size


def test_generate_matches_stream_with_same_seed(model):
    start = torch.zeros((1, 1), dtype=torch.long)
    torch.manual_seed(42)
    full = model.generate(start, max_new_tokens=8, temperature=0.9, top_k=5)
    torch.manual_seed(42)
    streamed = torch.cat(
        [start]
        + list(model.generate_stream(start, max_new_tokens=8, temperature=0.9, top_k=5)),
        dim=1,
    )
    assert torch.equal(full, streamed)


def test_generate_top_k_zero_means_disabled(model):
    # top_k=0 (slider UI à zéro) doit être traité comme top_k=None, pas planter.
    start = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(start, max_new_tokens=5, top_k=0)
    assert out.shape == (1, 6)


def test_generate_rejects_zero_temperature(model):
    start = torch.zeros((1, 1), dtype=torch.long)
    with pytest.raises(AssertionError, match="temperature"):
        model.generate(start, max_new_tokens=1, temperature=0.0)


def test_generate_rejects_empty_prompt(model):
    empty = torch.zeros((1, 0), dtype=torch.long)
    with pytest.raises(AssertionError, match="amorce"):
        model.generate(empty, max_new_tokens=1)


def test_generate_restores_training_mode(model):
    model.train()
    start = torch.zeros((1, 1), dtype=torch.long)
    model.generate(start, max_new_tokens=2)
    assert model.training
    model.eval()
    model.generate(start, max_new_tokens=2)
    assert not model.training
