"""Configuration centralisée des hyperparamètres du Mini-GPT.

Tout regrouper dans une dataclass facilite :
- le passage entre modules (train.py -> model.py, etc.)
- la sérialisation dans un checkpoint
- la modification d'un seul endroit
"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    # ----- Architecture -----
    vocab_size: int = 0           # défini par le tokenizer au runtime (dépend du corpus)
    block_size: int = 128         # longueur max de contexte (en tokens)
    n_layer: int = 6              # nombre de blocs Transformer empilés
    n_head: int = 6               # nombre de têtes d'attention par bloc
    n_embd: int = 192             # dimension des embeddings ; doit être divisible par n_head
    dropout: float = 0.1          # dropout (attention + FFN + résiduel)

    # ----- Optimisation -----
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_iters: int = 5000         # itérations totales
    eval_interval: int = 500      # éval toutes les N itérations
    eval_iters: int = 100         # nombre de batches utilisés pour estimer la loss
    weight_decay: float = 0.1     # régularisation L2 via AdamW
    beta1: float = 0.9            # AdamW
    beta2: float = 0.95           # AdamW (0.95 plutôt que 0.999, choix issu de GPT-3)
    grad_clip: float = 1.0        # clipping de la norme L2 des gradients

    # ----- LR schedule (warmup linéaire + cosine decay) -----
    warmup_iters: int = 100       # iters pendant lesquels lr monte linéairement de 0 -> learning_rate
    lr_decay_iters: int = 0       # iters de fin de decay ; 0 = utiliser max_iters
    min_lr: float = 3e-5          # lr minimum après decay (~ 1/10 du learning_rate)

    # ----- Données -----
    data_path: str = "data/input.txt"
    train_split: float = 0.9      # 90% train, 10% val

    # ----- Système -----
    device: str = "cuda"          # remplacé par "cpu" dans train.py si pas de GPU
    seed: int = 1337
    checkpoint_dir: str = "checkpoints"

    def __post_init__(self) -> None:
        # n_embd doit être divisible par n_head :
        # chaque tête reçoit n_embd // n_head dimensions.
        assert self.n_embd % self.n_head == 0, (
            f"n_embd ({self.n_embd}) doit être divisible par n_head ({self.n_head})"
        )

    @property
    def head_dim(self) -> int:
        """Dimension par tête d'attention (souvent noté d_k)."""
        return self.n_embd // self.n_head
