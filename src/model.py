"""model.py - Transformer décodeur complet (style GPT).

Construit pas à pas :
1. MultiHeadSelfAttention : attention causale multi-têtes        ✓
2. FeedForward + Block   : un bloc Transformer complet (pré-norm) ✓
3. GPT                   : modèle complet (embeddings + N blocs + head + loss + generate) ✓
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from config import GPTConfig


class MultiHeadSelfAttention(nn.Module):
    """Attention causale multi-têtes, façon GPT-2.

    Pour chaque position t, le module :
    - calcule une query q_t à partir de x_t
    - compare q_t aux keys k_0..k_t (positions passées + courante)
    - récupère une moyenne pondérée des values v_0..v_t

    En parallèle sur `n_head` têtes : chaque tête a ses propres projections
    Q/K/V et peut se spécialiser dans une relation différente.

    Causal : la position t n'a jamais accès à t+1, t+2, ... (sinon le modèle
    apprendrait à tricher en regardant la cible).
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim

        # Projection unique pour Q, K, V (plus efficace que 3 Linears séparés).
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Projection de sortie : après concat des têtes, re-projete dans n_embd.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Masque causal triangulaire inférieur. register_buffer : non entraînable
        # mais déplacé avec .to(device) et inclus dans state_dict.
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer(
            "causal_mask",
            mask.view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape

        # Projeter Q, K, V puis split en 3 tenseurs (B, T, C).
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Séparer les têtes : (B, T, C) -> (B, n_head, T, head_dim).
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scores scaled : 1/sqrt(d_k) pour éviter la saturation de softmax.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Masque causal : positions futures -> -inf.
        scores = scores.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0, float("-inf")
        )
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        # Moyenne pondérée des valeurs.
        out = weights @ v  # (B, n_head, T, head_dim)

        # Re-merge des têtes : contiguous() obligatoire avant view après transpose.
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.c_proj(out))
        return out


class FeedForward(nn.Module):
    """FFN : Linear(C -> 4C) -> GELU -> Linear(4C -> C) -> Dropout.

    Expansion 4x : convention GPT, sweet spot capacité / coût.
    GELU plutôt que ReLU : courbe plus douce, meilleure pour Transformers.
    Position-wise : le même MLP appliqué à chaque position indépendamment.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Bloc Transformer décodeur pré-norm :
        x = x + attn(LN(x))
        x = x + ffn(LN(x))

    Pré-norm (LN avant la sous-couche) : courant résiduel jamais normalisé,
    gradients qui traversent N blocs sans bottleneck. Convention GPT-2+.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class GPT(nn.Module):
    """Mini-GPT décodeur complet.

    Architecture :
        idx (B, T) entiers
          ├─ token_embed   (B, T) -> (B, T, C)
          ├─ pos_embed     broadcast (1, T, C)  (additionné, pas concaténé)
          ▼
        N x Block          (B, T, C)
          ▼
        LayerNorm finale   (B, T, C)
          ▼
        lm_head            (B, T, C) -> (B, T, V) = logits
          ▼
        cross_entropy si targets fourni

    Weight tying : lm_head.weight = token_embed.weight (même matrice).
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.vocab_size > 0, (
            "config.vocab_size doit être défini AVANT d'instancier GPT "
            "(le tokenizer le donne au runtime)."
        )
        self.config = config

        # Embeddings token et position (tous les deux entraînables).
        # Positionnels APPRIS (pas sinusoïdes du papier original) : choix GPT-2.
        self.token_embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embed = nn.Embedding(config.block_size, config.n_embd)

        # Dropout sur (token + pos) avant le premier bloc.
        self.drop = nn.Dropout(config.dropout)

        # La pile de N blocs.
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # LayerNorm finale (indispensable en pré-norm : sinon les sorties du
        # dernier bloc n'ont jamais été normalisées).
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Tête linguistique : projection vers vocab_size.
        # bias=False : standard en weight tying, et empiriquement neutre.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying : MÊME matrice de poids pour embedding et projection finale.
        # Économise V*C paramètres, aide la généralisation. C'est la même mémoire :
        # les gradients des deux usages s'accumulent automatiquement.
        self.lm_head.weight = self.token_embed.weight

        # Init des poids façon GPT-2.
        self.apply(self._init_weights)

        # Scaling spécifique des projections résiduelles (c_proj de attn et ffn).
        # Chaque bloc ajoute 2 termes au courant résiduel ; sans cette division
        # supplémentaire par sqrt(2 * n_layer), la variance grossit linéairement
        # avec la profondeur → activations qui explosent → NaN.
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module: nn.Module) -> None:
        """Init façon GPT-2 : N(0, 0.02) pour weights, 0 pour biais."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # LayerNorm : on garde les defaults PyTorch (gamma=1, beta=0).

    def forward(
        self,
        idx: Tensor,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Args:
            idx:     (B, T) entiers
            targets: (B, T) entiers, optionnel
        Returns:
            logits: (B, T, vocab_size)
            loss:   scalaire si targets fourni, sinon None
        """
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"séquence {T} > block_size {self.config.block_size}"
        )

        # 1) Token + position embeddings, additionnés.
        # pos = [0, 1, ..., T-1] ; pos_embed[pos] donne (T, C), unsqueeze pour broadcast.
        pos = torch.arange(T, device=idx.device).unsqueeze(0)  # (1, T)
        tok_emb = self.token_embed(idx)                          # (B, T, C)
        pos_emb = self.pos_embed(pos)                            # (1, T, C)
        x = self.drop(tok_emb + pos_emb)                         # broadcast somme + dropout

        # 2) N blocs.
        for block in self.blocks:
            x = block(x)

        # 3) LayerNorm finale + lm_head.
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, V)

        # 4) Loss si on entraîne.
        loss: Tensor | None = None
        if targets is not None:
            # cross_entropy attend (N, C) et (N,), pas (B, T, V) et (B, T).
            # Flatten -> (B*T, V) et (B*T,). Calcule la perte sur les B*T positions
            # simultanément (c'est ça qui rend l'entraînement efficient).
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate_stream(
        self,
        idx: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ):
        """Génération auto-régressive, un token yield à la fois.

        Générateur : permet à l'appelant (CLI, serveur web) de décoder et
        afficher chaque token dès qu'il est échantillonné, sans attendre la
        fin de la génération.

        Args:
            idx: (B, T_in) tokens d'amorce (le prompt encodé).
            max_new_tokens: nombre de tokens à générer.
            temperature: < 1 plus déterministe, > 1 plus aléatoire.
            top_k: si donné, on ne sample que parmi les k tokens les + probables.
        Yields:
            (B, 1) : l'id du token fraîchement échantillonné.
        """
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                # Tronquer le contexte à block_size : le modèle n'a pas
                # d'embedding pour les positions >= block_size.
                idx_cond = (
                    idx
                    if idx.size(1) <= self.config.block_size
                    else idx[:, -self.config.block_size :]
                )
                # Forward sans targets : on veut juste les logits.
                logits, _ = self(idx_cond)
                # Ne garder que la dernière position (c'est le next-token).
                logits = logits[:, -1, :] / temperature  # (B, V)
                # Filtrage top-k (optionnel) : -inf aux tokens hors top-k.
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                # Softmax -> probas -> sampling multinomial.
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)
                # Concaténer au contexte courant, puis livrer le token.
                idx = torch.cat([idx, next_id], dim=1)
                yield next_id
        finally:
            if was_training:
                self.train()

    @torch.no_grad()
    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Tensor:
        """Génération auto-régressive complète (wrapper sur generate_stream).

        Returns:
            (B, T_in + max_new_tokens) tokens.
        """
        for next_id in self.generate_stream(idx, max_new_tokens, temperature, top_k):
            idx = torch.cat([idx, next_id], dim=1)
        return idx


if __name__ == "__main__":
    # On valide MHSA, Block, puis GPT complet sur une petite config.
    cfg = GPTConfig(
        vocab_size=27, block_size=16, n_embd=64, n_head=4, n_layer=2,
        dropout=0.0,
    )
    B, T, C = 2, 8, cfg.n_embd
    x = torch.randn(B, T, C)

    # ----- MHSA -----
    attn = MultiHeadSelfAttention(cfg).eval()
    n = sum(p.numel() for p in attn.parameters())
    print(f"[MHSA]  params = {n}")
    with torch.no_grad():
        y1 = attn(x)
        x2 = x.clone(); x2[:, -1, :] = torch.randn(B, C)
        y2 = attn(x2)
    assert (y1[:, 0] - y2[:, 0]).abs().max() < 1e-6, "MHSA causalite cassee"
    print(f"[MHSA]  causalite OK")

    # ----- Block -----
    block = Block(cfg).eval()
    n = sum(p.numel() for p in block.parameters())
    print(f"[Block] params = {n}")
    with torch.no_grad():
        y1 = block(x)
        x2 = x.clone(); x2[:, -1, :] = torch.randn(B, C)
        y2 = block(x2)
    assert (y1[:, 0] - y2[:, 0]).abs().max() < 1e-6, "Block causalite cassee"
    print(f"[Block] causalite OK")

    # ----- GPT complet -----
    model = GPT(cfg).eval()
    n = sum(p.numel() for p in model.parameters())
    # Note : avec weight tying, lm_head.weight et token_embed.weight pointent
    # vers le MEME tenseur. parameters() le voit une seule fois.
    print(f"[GPT]   total params = {n:,}")

    # Faux batch
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))

    # Forward sans targets
    logits, loss = model(idx)
    assert logits.shape == (B, T, cfg.vocab_size)
    assert loss is None
    print(f"[GPT]   forward sans targets : logits {tuple(logits.shape)}, loss=None")

    # Forward avec targets
    logits, loss = model(idx, targets)
    assert logits.shape == (B, T, cfg.vocab_size)
    assert loss.dim() == 0
    expected = math.log(cfg.vocab_size)
    print(f"[GPT]   forward avec targets : loss = {loss.item():.4f}  (attendu ~ ln({cfg.vocab_size}) = {expected:.4f})")
    assert abs(loss.item() - expected) < 0.5, "Loss init trop loin de ln(V) : init suspecte"

    # Weight tying : verifier que c'est bien la meme memoire
    assert model.lm_head.weight.data_ptr() == model.token_embed.weight.data_ptr()
    print(f"[GPT]   weight tying confirme (meme adresse memoire)")

    # Generation
    start = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(start, max_new_tokens=10, top_k=5)
    print(f"[GPT]   generate : {tuple(start.shape)} -> {tuple(out.shape)}  ids = {out[0].tolist()}")
    assert out.shape == (1, 11)

    print("\nOK : MHSA + Block + GPT valides")
