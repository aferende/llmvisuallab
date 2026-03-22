"""
Tokenizer module - word-level tokenizer that builds vocabulary from user input.
All internal logic in English; UI text is handled by lang.py.
"""
import re
from typing import List, Tuple, Dict


class Tokenizer:
    """Word-level tokenizer with vocabulary building."""

    PAD = "<PAD>"
    UNK = "<UNK>"

    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}

    def fit(self, sentences: List[str]) -> None:
        """Build vocabulary from a list of sentences."""
        words = set()
        for sent in sentences:
            words.update(self._split(sent))

        # Special tokens first (fixed indices)
        all_tokens = [self.PAD, self.UNK] + sorted(words)
        self.vocab = {w: i for i, w in enumerate(all_tokens)}
        self.idx2word = {i: w for w, i in self.vocab.items()}

    def _split(self, text: str) -> List[str]:
        """Basic word-level split: lowercase, keep alphanumeric words."""
        return re.findall(r"\b\w+\b", text.lower())

    def tokenize(self, text: str) -> List[int]:
        """Convert a sentence to a list of token indices."""
        return [self.vocab.get(w, self.vocab[self.UNK]) for w in self._split(text)]

    def tokenize_with_words(self, text: str) -> List[Tuple[str, int]]:
        """Return (word, index) pairs for display purposes."""
        words = self._split(text)
        return [(w, self.vocab.get(w, self.vocab[self.UNK])) for w in words]

    def detokenize(self, indices: List[int]) -> str:
        """Convert a list of indices back to a human-readable string."""
        return " ".join(self.idx2word.get(i, self.UNK) for i in indices)

    def get_training_pairs(self, sentences: List[str]) -> List[Tuple[int, int]]:
        """
        Build (input_token, target_token) pairs for next-token prediction.
        For 'Il gatto mangia' → [(il, gatto), (gatto, mangia)]
        """
        pairs = []
        for sent in sentences:
            indices = self.tokenize(sent)
            for i in range(len(indices) - 1):
                pairs.append((indices[i], indices[i + 1]))
        return pairs

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
