"""
Tokenizer module - word-level and BPE subword tokenizer.
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


class SubwordTokenizer:
    """
    Byte-Pair Encoding (BPE) tokenizer trained from scratch on the input sentences.

    BPE is the algorithm used by GPT-2, RoBERTa, and many other real LLMs.
    It starts from individual characters and iteratively merges the most frequent
    adjacent pair until the target vocabulary size is reached.

    Interface is identical to Tokenizer so the rest of the app is unchanged.
    """

    PAD = "<PAD>"
    UNK = "<UNK>"
    _EOW = "</w>"   # end-of-word marker (internal BPE convention)
    _EOW_DISPLAY = "·"  # shown in the UI instead of </w>

    def __init__(self, target_vocab_size: int = 80):
        self.target_vocab_size = max(target_vocab_size, 10)
        self.vocab: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self.n_merges_done: int = 0

    # ------------------------------------------------------------------ #
    # Internal BPE helpers
    # ------------------------------------------------------------------ #

    def _word_to_chars(self, word: str) -> Tuple[str, ...]:
        """'hello' → ('h','e','l','l','o</w>')"""
        chars = list(word)
        if chars:
            chars[-1] = chars[-1] + self._EOW
        return tuple(chars)

    @staticmethod
    def _get_pairs(word: Tuple[str, ...]) -> List[Tuple[str, str]]:
        return [(word[i], word[i + 1]) for i in range(len(word) - 1)]

    @staticmethod
    def _merge_pair(word: Tuple[str, ...], pair: Tuple[str, str]) -> Tuple[str, ...]:
        out: List[str] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                out.append(word[i] + word[i + 1])
                i += 2
            else:
                out.append(word[i])
                i += 1
        return tuple(out)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def fit(self, sentences: List[str]) -> None:
        """Learn BPE merges from the corpus and build the vocabulary."""
        # Count word frequencies (case-insensitive, word-split)
        word_freq: Dict[str, int] = {}
        for sent in sentences:
            for word in re.findall(r"\b\w+\b", sent.lower()):
                word_freq[word] = word_freq.get(word, 0) + 1

        # Represent each word as a sequence of characters + </w>
        bpe_vocab: Dict[Tuple[str, ...], int] = {
            self._word_to_chars(w): freq for w, freq in word_freq.items()
        }

        # Collect initial character symbols — always kept in final vocab (like real BPE)
        initial_symbols: set = set()
        for word in bpe_vocab:
            initial_symbols.update(word)

        # How many merges can we run before hitting target vocab size?
        n_base = len(initial_symbols) + 2   # +2 for PAD and UNK
        n_merges = max(0, self.target_vocab_size - n_base)

        self.merges = []
        for _ in range(n_merges):
            pair_freq: Dict[Tuple[str, str], int] = {}
            for word, freq in bpe_vocab.items():
                for pair in self._get_pairs(word):
                    pair_freq[pair] = pair_freq.get(pair, 0) + freq

            if not pair_freq:
                break

            best = max(pair_freq, key=lambda p: pair_freq[p])
            self.merges.append(best)

            new_bpe: Dict[Tuple[str, ...], int] = {}
            for word, freq in bpe_vocab.items():
                merged = self._merge_pair(word, best)
                new_bpe[merged] = new_bpe.get(merged, 0) + freq
            bpe_vocab = new_bpe

        self.n_merges_done = len(self.merges)

        # Build final vocab: base characters + every merged token (incl. intermediates)
        final_syms: set = set(initial_symbols)   # base chars always retained
        for a, b in self.merges:                 # all merge results are valid tokens
            final_syms.add(a + b)
        for word in bpe_vocab:                   # tokens surviving to the end
            final_syms.update(word)

        sorted_syms = sorted(final_syms)
        token_list = [self.PAD, self.UNK] + sorted_syms
        self.vocab = {tok: i for i, tok in enumerate(token_list)}
        self.idx2word = {i: tok for tok, i in self.vocab.items()}

    def _encode_word(self, word: str) -> Tuple[str, ...]:
        """Apply learned BPE merges to a single word."""
        tokens = self._word_to_chars(word)
        for pair in self.merges:
            tokens = self._merge_pair(tokens, pair)
        return tokens

    def tokenize(self, text: str) -> List[int]:
        """Convert text to a list of subword token indices."""
        unk = self.vocab.get(self.UNK, 1)
        result = []
        for word in re.findall(r"\b\w+\b", text.lower()):
            for piece in self._encode_word(word):
                result.append(self.vocab.get(piece, unk))
        return result

    def tokenize_with_words(self, text: str) -> List[Tuple[str, int]]:
        """Return (display_piece, index) pairs — same interface as Tokenizer."""
        unk = self.vocab.get(self.UNK, 1)
        result = []
        for word in re.findall(r"\b\w+\b", text.lower()):
            for piece in self._encode_word(word):
                display = piece.replace(self._EOW, self._EOW_DISPLAY)
                result.append((display, self.vocab.get(piece, unk)))
        return result

    def detokenize(self, indices: List[int]) -> str:
        return " ".join(
            self.idx2word.get(i, self.UNK).replace(self._EOW, "") for i in indices
        )

    def get_training_pairs(self, sentences: List[str]) -> List[Tuple[int, int]]:
        pairs: set = set()
        for sent in sentences:
            tokens = self.tokenize(sent)
            for i in range(len(tokens) - 1):
                pairs.add((tokens[i], tokens[i + 1]))
        return list(pairs)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
