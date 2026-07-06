"""Tests du CharTokenizer : round-trip, déterminisme, persistance, erreurs."""

import pytest

from tokenizer import CharTokenizer

SAMPLE = "Bonjour, GPT ! Apprends-moi quelque chose.\nÉté, çà et là…\n"


def test_roundtrip_is_identity():
    tok = CharTokenizer.from_text(SAMPLE)
    assert tok.decode(tok.encode(SAMPLE)) == SAMPLE


def test_vocab_is_sorted_and_deduplicated():
    tok = CharTokenizer.from_text("abba")
    assert tok.chars == ["a", "b"]
    assert tok.vocab_size == 2


def test_vocab_order_is_deterministic():
    # Deux constructions sur le même texte doivent donner les mêmes indices.
    tok1 = CharTokenizer.from_text(SAMPLE)
    tok2 = CharTokenizer.from_text(SAMPLE)
    assert tok1.stoi == tok2.stoi


def test_save_and_from_file_roundtrip(tmp_path):
    tok = CharTokenizer.from_text(SAMPLE)
    path = tmp_path / "vocab.json"
    tok.save(path)
    tok2 = CharTokenizer.from_file(path)
    assert tok2.chars == tok.chars
    assert tok2.decode(tok2.encode(SAMPLE)) == SAMPLE


def test_unknown_char_raises_explicit_keyerror():
    tok = CharTokenizer.from_text("abc")
    with pytest.raises(KeyError, match="absent du vocabulaire"):
        tok.encode("abz#")
