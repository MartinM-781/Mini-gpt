"""Tests du CharDataset : split, shapes des batches, décalage x/y."""

import pytest
import torch

from config import GPTConfig
from dataset import CharDataset
from tokenizer import CharTokenizer

SAMPLE = "Le chat dort sur le tapis. " * 100  # ~2700 chars


@pytest.fixture()
def dataset(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text(SAMPLE, encoding="utf-8")
    cfg = GPTConfig(batch_size=4, block_size=16, device="cpu")
    tok = CharTokenizer.from_text(SAMPLE)
    cfg.vocab_size = tok.vocab_size
    return CharDataset(path, tok, cfg)


def test_split_proportions(dataset):
    n_total = len(dataset.train_data) + len(dataset.val_data)
    assert n_total == len(SAMPLE)
    assert len(dataset.train_data) == int(0.9 * n_total)


def test_batch_shapes(dataset):
    for split in ("train", "val"):
        x, y = dataset.get_batch(split)
        assert x.shape == (4, 16)
        assert y.shape == (4, 16)
        assert x.dtype == torch.long
        assert y.dtype == torch.long


def test_targets_are_inputs_shifted_by_one(dataset):
    x, y = dataset.get_batch("train")
    # Pour chaque ligne du batch : y[t] == x[t+1].
    assert torch.equal(x[:, 1:], y[:, :-1])
