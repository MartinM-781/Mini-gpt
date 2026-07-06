"""sample.py - generation de texte avec un Mini-GPT entraine.

Usage :
    cd C:\\dev\\mini-gpt
    python src/sample.py --prompt "Le chat" --max_new_tokens 200 --temperature 0.8 --top_k 40

Recharge le checkpoint le plus recent dans `checkpoints/ckpt.pt` et le
tokenizer dans `checkpoints/vocab.json`, puis genere `num_samples` echantillons.
"""

from __future__ import annotations

import argparse
import sys

import torch

from config import GPTConfig
from model import GPT
from tokenizer import CharTokenizer


def load_checkpoint(
    ckpt_path: str, vocab_path: str, device: str
) -> tuple[GPT, CharTokenizer, dict]:
    """Recharge modele + tokenizer depuis disque.

    Le tokenizer et le modele sont sauves SEPAREMENT :
    - vocab.json est independant du modele (utile aussi pour re-encoder
      des prompts ou re-entrainer)
    - ckpt.pt contient les poids + config + etat de l'optimiseur (pour
      pouvoir reprendre l'entrainement)
    """
    # 1) Tokenizer
    tokenizer = CharTokenizer.from_file(vocab_path)

    # 2) Checkpoint : weights_only=False car on a serialise un dict de config
    # en plus des tenseurs. SAFE ici car c'est notre propre fichier ; ne
    # jamais faire weights_only=False sur un .pt telecharge d'internet.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # 3) Reconstruire la GPTConfig depuis le dict serialise.
    # GPTConfig(**dict) repasse par __init__ et donc par __post_init__
    # (verification n_embd % n_head == 0).
    config = GPTConfig(**ckpt["config"])
    config.device = device  # override le device serialise

    # 4) Verifier la coherence entre tokenizer et config.
    assert config.vocab_size == tokenizer.vocab_size, (
        f"mismatch tokenizer/config : "
        f"vocab.json a {tokenizer.vocab_size} chars, "
        f"ckpt a vocab_size={config.vocab_size}"
    )

    # 5) Instancier et charger les poids.
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(
        f"loaded : {ckpt_path} | iter {ckpt['iter']} "
        f"| val_loss {ckpt['val_loss']:.4f} | device {device}"
    )
    # Métadonnées utiles à l'appelant (affichage UI, logs).
    meta = {
        "iter": ckpt["iter"],
        "val_loss": ckpt["val_loss"],
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    return model, tokenizer, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a trained Mini-GPT.")
    parser.add_argument("--ckpt", default="checkpoints/ckpt.pt")
    parser.add_argument("--vocab", default="checkpoints/vocab.json")
    parser.add_argument(
        "--prompt", default="\n",
        help="Texte d'amorce. Defaut : un newline (genere from scratch).",
    )
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="< 1 plus deterministe, > 1 plus aleatoire.",
    )
    parser.add_argument(
        "--top_k", type=int, default=None,
        help="Si donne, ne sample que parmi les k tokens les plus probables.",
    )
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    model, tokenizer, _ = load_checkpoint(args.ckpt, args.vocab, device)

    # Encoder le prompt. Erreur explicite si un char est absent du vocab.
    try:
        idx_list = tokenizer.encode(args.prompt)
    except KeyError as e:
        print(f"Erreur d'encodage du prompt : {e}", file=sys.stderr)
        sys.exit(1)
    idx = torch.tensor([idx_list], dtype=torch.long, device=device)

    print(f"\nprompt = {args.prompt!r}")
    print(f"--- {args.num_samples} sample(s) ---\n")
    for i in range(args.num_samples):
        out = model.generate(
            idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        text = tokenizer.decode(out[0].tolist())
        print(f"=== sample {i + 1} ===")
        print(text)
        print()


if __name__ == "__main__":
    main()
