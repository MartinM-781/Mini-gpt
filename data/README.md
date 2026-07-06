# Corpus

`input.txt` — **tinyshakespeare** : ~1,1 Mo de pièces de Shakespeare concaténées
(domaine public), popularisé par [char-rnn](https://github.com/karpathy/char-rnn)
d'Andrej Karpathy comme corpus de démonstration pour les modèles de langage
au niveau caractère.

Source : <https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt>

Pour entraîner sur votre propre corpus, remplacez simplement `input.txt` par
n'importe quel fichier texte UTF-8 (idéalement ≥ 1 Mo), ou passez
`--data-path chemin/vers/corpus.txt` à `src/train.py`.
