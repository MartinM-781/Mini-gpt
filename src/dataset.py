"""dataset.py - chargement du corpus, split train/val, génération de batches.

Le modèle apprend à prédire le token suivant. Ce module produit donc des paires
(x, y) où :
    x = [B, o, n, j, o, u, r]
    y = [o, n, j, o, u, r, !]
            ↑
       y[t] = x[t+1] : la cible à la position t est le token suivant.

Une séquence de longueur L donne L exemples d'apprentissage simultanés
(un par position) — d'où l'efficacité du Transformer décodeur.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from config import GPTConfig
from tokenizer import CharTokenizer


class CharDataset:
    """Lit un corpus brut, l'encode, fournit des batches (x, y).

    Tout le corpus est gardé en mémoire sous forme d'un long tenseur d'entiers.
    Pour des corpus de quelques Mo (Shakespeare = ~1 Mo), c'est plus rapide
    et plus simple qu'un DataLoader streaming.
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: CharTokenizer,
        config: GPTConfig,
    ) -> None:
        # 1) Lecture brute du corpus.
        #    encoding="utf-8" explicite : Windows défaute à cp1252.
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()

        # 2) Encodage en un long vecteur d'entiers.
        #    dtype=torch.long (int64) : nn.Embedding exige des index 64 bits,
        #    même si vocab_size < 256.
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

        # 3) Split train/val par POSITION (pas mélange aléatoire).
        #    Le corpus est UN seul document continu : on coupe à 90 % pour
        #    que la val contienne du texte que le modèle n'a jamais vu, et
        #    pour préserver les dépendances séquentielles dans chaque split.
        n = int(config.train_split * len(data))
        self.train_data: Tensor = data[:n]
        self.val_data: Tensor = data[n:]

        # 4) Mémoriser les hyperparams utiles à get_batch.
        self.block_size = config.block_size
        self.batch_size = config.batch_size
        self.device = config.device

    def get_batch(self, split: str) -> tuple[Tensor, Tensor]:
        """Tire un batch (x, y) où y est x décalé d'une position.

        Returns:
            x: (batch_size, block_size) tokens d'entrée
            y: (batch_size, block_size) tokens cibles (next-token prediction)

        Pour tout t : y[b, t] == x[b, t+1].

        Stratégie : positions de départ UNIFORMES dans le corpus.
        - Le corpus est une longue séquence, pas un set d'exemples i.i.d.
        - Tirer aléatoirement à chaque step approxime l'hypothèse i.i.d.
          que SGD attend pour des gradients non biaisés.
        - Conséquence : la notion d'« epoch » perd son sens ici ; on raisonne
          en nombre d'itérations. Des chevauchements entre batches peuvent
          arriver, sans conséquence.
        """
        data = self.train_data if split == "train" else self.val_data

        # Positions de départ valides : il faut pouvoir lire block_size + 1
        # tokens à partir de ix (block_size pour x, +1 pour la cible finale de y).
        # Dernier départ valide : len(data) - block_size - 1, donc randint
        # exclusif sur len(data) - block_size.
        assert len(data) > self.block_size, (
            f"split '{split}' trop court : {len(data)} tokens pour un "
            f"block_size de {self.block_size}. Corpus trop petit ou "
            f"train_split trop extrême."
        )
        ix = torch.randint(0, len(data) - self.block_size, (self.batch_size,))

        # Pour chaque position tirée, slice block_size tokens consécutifs,
        # puis stack en un tenseur 2D de forme (batch_size, block_size).
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in ix])

        # Transfert sur device. pin_memory + non_blocking possibles plus tard
        # comme optimisation (overlap CPU→GPU avec compute).
        x, y = x.to(self.device), y.to(self.device)
        return x, y


if __name__ == "__main__":
    # Test : on crée un mini-corpus en mémoire, on l'écrit dans un fichier
    # temporaire, on instancie le dataset, on tire un batch, on vérifie
    # que y est bien x décalé d'une position.
    import os
    import tempfile

    sample = "Le chat dort sur le tapis. " * 100  # ~2700 chars

    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, suffix=".txt"
    ) as f:
        f.write(sample)
        path = f.name

    try:
        cfg = GPTConfig(batch_size=4, block_size=16, device="cpu")
        tok = CharTokenizer.from_text(sample)
        cfg.vocab_size = tok.vocab_size

        ds = CharDataset(path, tok, cfg)
        print(f"len(train) = {len(ds.train_data)}, len(val) = {len(ds.val_data)}")

        x, y = ds.get_batch("train")
        print(f"x.shape = {tuple(x.shape)}, y.shape = {tuple(y.shape)}")
        print(f"x[0] = {x[0].tolist()}")
        print(f"y[0] = {y[0].tolist()}")
        print(f"decode x[0] = {tok.decode(x[0].tolist())!r}")
        print(f"decode y[0] = {tok.decode(y[0].tolist())!r}")

        # Sanity check : y[:, :-1] doit valoir x[:, 1:]
        assert torch.equal(
            x[0, 1:], y[0, :-1]
        ), "y doit être x décalé d'une position"
        print("OK : x[:, 1:] == y[:, :-1] (décalage vérifié)")
    finally:
        os.unlink(path)
