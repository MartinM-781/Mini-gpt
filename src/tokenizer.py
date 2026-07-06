"""CharTokenizer : tokenization au niveau du caractère.

Convertit du texte en suite d'indices entiers (pour le modèle) et inversement.

Au niveau caractère :
- chaque caractère unique du corpus reçoit un indice entre 0 et vocab_size - 1
- vocab_size = nombre de caractères distincts du corpus
- très simple, vocab petit (~80-200 chars pour FR/EN), pas besoin d'UNK

Limite : ne capture pas la structure morphémique des mots, les séquences sont
plus longues qu'avec du BPE. On passera à BPE plus tard si nécessaire.
"""

from __future__ import annotations

import json
from pathlib import Path


class CharTokenizer:
    """Tokenizer au niveau caractère.

    Construit un vocabulaire trié depuis un corpus, et expose :
    - encode(text) -> list[int]
    - decode(ids)  -> str
    """

    def __init__(self, chars: list[str]) -> None:
        # `chars` : liste triée et dédupliquée de caractères (le vocabulaire).
        # On construit les deux mappings nécessaires :
        #   stoi : char  -> index (utilisé par encode)
        #   itos : index -> char  (utilisé par decode)
        self.chars: list[str] = list(chars)
        self.vocab_size: int = len(self.chars)
        self.stoi: dict[str, int] = {ch: i for i, ch in enumerate(self.chars)}
        self.itos: dict[int, str] = {i: ch for i, ch in enumerate(self.chars)}

    # --- Constructions alternatives -------------------------------------------------

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Construit le tokenizer depuis un corpus.

        sorted(set(text)) :
        - set() déduplique les caractères
        - sorted() rend l'ordre déterministe (par codepoint Unicode).
          Indispensable : sinon les indices changeraient entre deux runs Python
          et un modèle entraîné lundi serait illisible mardi.
        """
        chars = sorted(set(text))
        return cls(chars)

    @classmethod
    def from_file(cls, path: str | Path) -> "CharTokenizer":
        """Charge un vocabulaire sauvegardé (utile à la génération)."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["chars"])

    # --- Persistance ----------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Sauvegarde le vocabulaire en JSON.

        À appeler après l'entraînement. Sans ce fichier, on ne peut pas
        re-decoder les sorties du modèle : les indices n'ont aucun sens
        sans le mapping qui les a produits.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"chars": self.chars}, f, ensure_ascii=False, indent=2)

    # --- Encode / decode ------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Texte -> liste d'indices.

        Lève une KeyError explicite si le texte contient un caractère absent
        du vocabulaire (cas typique : un prompt à l'inférence contenant un
        caractère jamais vu à l'entraînement). On préfère échouer fort plutôt
        que de remplacer silencieusement.
        """
        try:
            return [self.stoi[ch] for ch in text]
        except KeyError as e:
            ch = e.args[0]
            raise KeyError(
                f"Caractère absent du vocabulaire : {ch!r} (codepoint {ord(ch)}). "
                f"Le tokenizer ne connaît que les caractères vus dans le corpus "
                f"d'entraînement."
            ) from None

    def decode(self, ids: list[int]) -> str:
        """Liste d'indices -> texte."""
        return "".join(self.itos[i] for i in ids)


if __name__ == "__main__":
    # Démo / test de fumée : on construit un tokenizer sur une mini-phrase,
    # on vérifie que le round-trip encode -> decode est l'identité.
    sample = "Bonjour, GPT ! Apprends-moi quelque chose.\n"
    tok = CharTokenizer.from_text(sample)
    print(f"vocab_size = {tok.vocab_size}")
    print(f"vocab      = {tok.chars}")
    ids = tok.encode(sample)
    print(f"encoded    = {ids}")
    back = tok.decode(ids)
    print(f"decoded    = {back!r}")
    assert back == sample, "round-trip encode/decode doit être l'identité"
    print("OK : round-trip identique")
