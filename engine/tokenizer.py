from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SimpleTokenizer:
    """Simple word-level tokenizer that keeps punctuation attached for visibility."""

    token_to_id: Dict[str, int] = field(default_factory=dict)
    id_to_token: Dict[int, str] = field(default_factory=dict)

    def fit(self, texts: List[str]) -> None:
        for text in texts:
            for token in self.tokenize(text):
                if token not in self.token_to_id:
                    idx = len(self.token_to_id)
                    self.token_to_id[token] = idx
                    self.id_to_token[idx] = token

    def tokenize(self, text: str) -> List[str]:
        return [tok.strip() for tok in text.split() if tok.strip()]

    def encode(self, text: str) -> List[int]:
        return [self.token_to_id[token] for token in self.tokenize(text) if token in self.token_to_id]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.id_to_token[idx] for idx in ids if idx in self.id_to_token]

    @property
    def vocab_size(self) -> int:
        return max(1, len(self.token_to_id))
