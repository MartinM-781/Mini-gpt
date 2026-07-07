# mini-gpt

[![CI](https://github.com/MartinM-781/Mini-gpt/actions/workflows/ci.yml/badge.svg)](https://github.com/MartinM-781/Mini-gpt/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**English** | [Français](README.fr.md)

A GPT-style decoder-only Transformer **implemented from scratch in pure PyTorch** —
no `nn.Transformer`, no `nn.MultiheadAttention`, no HuggingFace, no tiktoken.
Trained at character level on tinyshakespeare, it learns to write pseudo-Shakespeare
in ~20 minutes on a laptop CPU.

The goal is to understand every component of a GPT by rebuilding it by hand:
tokenizer, causal multi-head attention, pre-norm residual blocks, training loop,
LR scheduling, sampling. Every file is extensively commented (in French — the
project doubles as French-language teaching material).

![playground screenshot](assets/playground.png)

## What's inside

| Component | Details |
|---|---|
| [`tokenizer.py`](src/tokenizer.py) | Character-level tokenizer, deterministic vocab, JSON persistence |
| [`model.py`](src/model.py) | Multi-head causal self-attention (fused QKV), pre-norm blocks, GELU FFN, learned positional embeddings, **weight tying**, GPT-2 init with 1/√(2·n_layer) residual scaling |
| [`train.py`](src/train.py) | AdamW with selective weight decay (2-D params only), linear warmup + cosine decay, gradient clipping, best-checkpoint saving, CSV metrics |
| [`sample.py`](src/sample.py) | Temperature + top-k sampling, **streaming generation** (token-by-token generator) |
| [`serve.py`](src/serve.py) | Web playground with live token streaming — stdlib `http.server` only, zero web dependencies |
| [`tests/`](tests/) | 31 pytest tests (causality, shapes, init statistics, weight tying, LR schedule, streaming, edge cases) — run in CI |

## Quickstart

```bash
git clone https://github.com/MartinM-781/Mini-gpt.git
cd Mini-gpt
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows — or: source .venv/bin/activate
pip install -r requirements.txt
```

**Train** (corpus included — [tinyshakespeare](data/input.txt), 1.1 MB of public-domain Shakespeare):

```bash
python src/train.py --preset cpu-small     # smallest model, quickest first run
python src/train.py --preset cpu-medium    # ~30 min on a laptop CPU — the model shown below
python src/train.py --preset gpu           # full config, a few minutes with CUDA
```

| Preset | Layers | Heads | Embed | Context | Params | Val loss |
|---|---|---|---|---|---|---|
| `cpu-small` | 4 | 4 | 128 | 64 | 0.8 M | ≈ 1.9 |
| `cpu-medium` | 6 | 6 | 192 | 96 | 2.7 M | **1.88** |
| `gpu` | 6 | 6 | 192 | 128 | 2.7 M | — |

**Generate** from the command line:

```bash
python src/sample.py --prompt "ROMEO:" --max_new_tokens 300 --temperature 0.8 --top_k 40
```

**Or launch the web playground** (streams characters as they are sampled, ChatGPT-style —
~100 tokens/s on a laptop CPU):

```bash
python src/serve.py        # -> http://127.0.0.1:8000
```

## Results

Training curve for `cpu-medium` (2.7 M params, 3 000 iterations, ~30 min on a laptop CPU).
Runs are seeded end-to-end: two identical runs converge to the exact same 1.8824 val loss.

![loss curve](assets/loss_curve.png)

Sample output after training (`--temperature 0.75 --top_k 40`):

```text
KING HENRY:
Prick with your hearth the arry his well,
And good many king with dike it at her,
And frectious cancecious that hath frierl th.
Orw knotherer be one the that not do,
Sondire of mide hall all thou have prestenced,
This to curting and, the the soul for and her.

QUEEN ELIZABET:
Wo, what here be it somet:
Shall the chent to shir and him where ploke...
```

Not Shakespeare yet — but for a 2.7 M-parameter model that started from random
weights half an hour earlier, on a laptop, with a character-level tokenizer: real
English words, iambic-ish line lengths, theatre structure, and a spontaneous
QUEEN ELIZABET answering a KING HENRY.

## Architecture

```text
idx (B, T) integer tokens
  ├─ token embedding      (B, T) -> (B, T, C)
  ├─ position embedding   + (1, T, C)   learned, additive
  ▼
N × Block:                pre-norm, residual
  x = x + MHSA(LN(x))     causal multi-head self-attention
  x = x + FFN(LN(x))      Linear(C→4C) → GELU → Linear(4C→C)
  ▼
LayerNorm
  ▼
lm_head                   (B, T, C) -> (B, T, vocab)   weight-tied with token embedding
```

Design choices follow GPT-2: pre-norm blocks (stable gradients at depth),
fused QKV projection, weight tying between the input embedding and the output
head, N(0, 0.02) init with residual projections scaled by 1/√(2·n_layer)
to keep the residual-stream variance flat, AdamW β₂ = 0.95.
Each choice is motivated in the code comments.

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Covers: attention causality (perturbing the last token must not change earlier
outputs), initial loss ≈ ln(vocab_size), weight tying (same memory address),
generation beyond block_size, stream/batch generation equivalence, LR schedule
boundaries, optimizer decay groups.

## Project structure

```text
mini-gpt/
├── src/
│   ├── config.py          # GPTConfig dataclass (all hyperparameters)
│   ├── tokenizer.py       # CharTokenizer
│   ├── dataset.py         # train/val split + random batching
│   ├── model.py           # MultiHeadSelfAttention, Block, GPT
│   ├── train.py           # training loop + presets CLI
│   ├── sample.py          # CLI generation
│   └── serve.py           # web playground (stdlib only)
├── web/index.html         # playground front-end
├── tests/                 # pytest suite
├── scripts/plot_metrics.py
├── data/input.txt         # tinyshakespeare corpus (public domain)
└── .github/workflows/ci.yml
```

## License

[MIT](LICENSE)
