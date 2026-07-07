"""serve.py - mini playground web pour le Mini-GPT (stdlib uniquement).

Usage (depuis la racine du projet) :
    python src/serve.py                # http://127.0.0.1:8000
    python src/serve.py --port 8080

Sert une page HTML (web/index.html) et une API de génération en streaming :
chaque caractère est envoyé au client dès qu'il est échantillonné, ce qui
donne l'effet « machine à écrire » des interfaces LLM modernes.

Aucune dépendance web (pas de Flask/FastAPI) : http.server suffit largement
pour un serveur local mono-utilisateur, et ça reste dans l'esprit du projet
(tout comprendre, zéro boîte noire).
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import torch

from sample import load_checkpoint

INDEX_HTML = Path(__file__).resolve().parent.parent / "web" / "index.html"


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def make_handler(model, tokenizer, meta: dict):
    """Fabrique la classe Handler avec modèle/tokenizer capturés en closure.

    BaseHTTPRequestHandler est instancié par requête : on ne peut pas lui
    passer d'arguments au __init__, d'où la closure.
    """

    info = {
        "n_params": meta["n_params"],
        "iter": meta["iter"],
        "val_loss": round(meta["val_loss"], 4),
        "vocab_size": tokenizer.vocab_size,
        "block_size": model.config.block_size,
        "n_layer": model.config.n_layer,
        "n_head": model.config.n_head,
        "n_embd": model.config.n_embd,
    }
    # Device réel du modèle : le tenseur du prompt doit être créé dessus,
    # sinon RuntimeError au premier forward sur une machine CUDA.
    device = next(model.parameters()).device

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, obj: dict, status: int = 200) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            # urlparse : ignorer la query string (ex: /?autorun=1).
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                body = INDEX_HTML.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/api/info":
                self._send_json(info)
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            if self.path != "/api/generate":
                self.send_error(404)
                return

            length = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send_json({"error": "JSON invalide"}, status=400)
                return
            if not isinstance(payload, dict):
                self._send_json({"error": "objet JSON attendu"}, status=400)
                return

            # Coercition défensive : un client curl/script peut envoyer
            # n'importe quoi ; on répond 400 plutôt que de tuer le thread.
            try:
                prompt = str(payload.get("prompt") or "\n")
                max_new_tokens = int(clamp(int(payload.get("max_new_tokens", 300)), 1, 2000))
                temperature = clamp(float(payload.get("temperature", 0.8)), 0.05, 3.0)
                top_k = payload.get("top_k") or None
                if top_k is not None:
                    top_k = int(clamp(int(top_k), 1, tokenizer.vocab_size))
            except (TypeError, ValueError) as e:
                self._send_json({"error": f"paramètre invalide : {e}"}, status=400)
                return

            # Encoder le prompt ; 400 explicite si un caractère est inconnu.
            try:
                idx_list = tokenizer.encode(prompt)
            except KeyError as e:
                self._send_json({"error": str(e)}, status=400)
                return
            idx = torch.tensor([idx_list], dtype=torch.long, device=device)

            # Réponse streamée : pas de Content-Length, le client lit
            # jusqu'à la fermeture de la connexion (HTTP/1.0).
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            try:
                for token in model.generate_stream(
                    idx, max_new_tokens, temperature=temperature, top_k=top_k
                ):
                    ch = tokenizer.decode([token.item()])
                    self.wfile.write(ch.encode("utf-8"))
                    self.wfile.flush()
            except (ConnectionAbortedError, BrokenPipeError, ConnectionResetError):
                # Le client a annulé (bouton Stop) : on arrête la génération,
                # inutile de brûler du CPU pour personne.
                pass

        def log_message(self, format: str, *args) -> None:
            print(f"[serve] {self.address_string()} - {format % args}")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini-GPT web playground.")
    parser.add_argument("--ckpt", default="checkpoints/ckpt.pt")
    parser.add_argument("--vocab", default="checkpoints/vocab.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, tokenizer, meta = load_checkpoint(args.ckpt, args.vocab, device)
    except FileNotFoundError as e:
        print(
            f"[serve] checkpoint introuvable : {e.filename}\n"
            f"[serve] entraînez d'abord un modèle : python src/train.py --preset cpu-small"
        )
        raise SystemExit(1)

    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(model, tokenizer, meta)
    )
    print(f"[serve] playground sur http://{args.host}:{args.port}  (Ctrl+C pour arrêter)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] arrêt.")


if __name__ == "__main__":
    main()
