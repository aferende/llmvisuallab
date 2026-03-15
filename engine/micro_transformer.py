from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class TrainSnapshot:
    step: int
    loss: float
    weight_strength: float


class MicroTransformer:
    """Tiny educational model with embedding + one hidden layer + output projection."""

    def __init__(self, vocab_size: int, hidden_size: int = 8, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embeddings = rng.normal(0, 0.2, size=(vocab_size, hidden_size))
        self.w_hidden = rng.normal(0, 0.2, size=(hidden_size, hidden_size))
        self.b_hidden = np.zeros(hidden_size)
        self.w_out = rng.normal(0, 0.2, size=(hidden_size, vocab_size))
        self.b_out = np.zeros(vocab_size)

    def _forward(self, token_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if token_ids.size == 0:
            token_ids = np.array([0])
        embed = self.embeddings[token_ids].mean(axis=0)
        hidden_lin = embed @ self.w_hidden + self.b_hidden
        hidden = np.tanh(hidden_lin)
        logits = hidden @ self.w_out + self.b_out
        return embed, hidden, logits

    def train(self, pairs: List[Tuple[List[int], int]], steps: int = 20, lr: float = 0.05) -> List[TrainSnapshot]:
        if not pairs:
            return []

        snapshots: List[TrainSnapshot] = []
        for step in range(1, steps + 1):
            x_tokens, y = pairs[(step - 1) % len(pairs)]
            x_arr = np.array(x_tokens, dtype=int)
            embed, hidden, logits = self._forward(x_arr)

            exps = np.exp(logits - np.max(logits))
            probs = exps / np.sum(exps)
            loss = -np.log(max(probs[y], 1e-9))

            dlogits = probs
            dlogits[y] -= 1.0

            grad_w_out = np.outer(hidden, dlogits)
            grad_b_out = dlogits

            dhidden = self.w_out @ dlogits
            dtanh = (1 - hidden**2) * dhidden

            grad_w_hidden = np.outer(embed, dtanh)
            grad_b_hidden = dtanh
            dembed = self.w_hidden @ dtanh

            if x_arr.size > 0:
                emb_grad = dembed / x_arr.size
                for token_id in x_arr:
                    self.embeddings[token_id] -= lr * emb_grad

            self.w_out -= lr * grad_w_out
            self.b_out -= lr * grad_b_out
            self.w_hidden -= lr * grad_w_hidden
            self.b_hidden -= lr * grad_b_hidden

            weight_strength = float(np.linalg.norm(self.w_hidden) + np.linalg.norm(self.w_out))
            snapshots.append(TrainSnapshot(step=step, loss=float(loss), weight_strength=weight_strength))

        return snapshots

    def predict_next_token(self, token_ids: List[int]) -> Tuple[int, np.ndarray]:
        arr = np.array(token_ids, dtype=int)
        _, _, logits = self._forward(arr)
        probs = np.exp(logits - np.max(logits))
        probs = probs / probs.sum()
        pred = int(np.argmax(probs))
        return pred, probs
