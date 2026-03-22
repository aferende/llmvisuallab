"""
TinyLM - A minimal neural language model implemented from scratch with NumPy.

Architecture:
  Input token (one-hot) → Embedding lookup → Hidden layer (tanh) → Output (softmax)

This is intentionally simple for educational clarity.
"""
import numpy as np
from typing import Dict, Tuple


class TinyLM:
    """
    Tiny language model with:
      W_emb  : (vocab_size, embed_dim)    - embedding matrix
      W_h    : (embed_dim,  hidden_size)  - hidden layer weights
      b_h    : (hidden_size,)             - hidden bias
      W_out  : (hidden_size, vocab_size)  - output weights
      b_out  : (vocab_size,)              - output bias
    """

    def __init__(self, vocab_size: int, embed_dim: int = 8, hidden_size: int = 16):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier-style initialisation for stable early training."""
        rng = np.random.default_rng(42)

        def xavier(fan_in, fan_out):
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, (fan_in, fan_out))

        self.W_emb = xavier(self.vocab_size, self.embed_dim)
        self.W_h = xavier(self.embed_dim, self.hidden_size)
        self.b_h = np.zeros(self.hidden_size)
        self.W_out = xavier(self.hidden_size, self.vocab_size)
        self.b_out = np.zeros(self.vocab_size)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, token_idx: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Run a forward pass for a single input token.

        Returns
        -------
        probs       : (vocab_size,)  – probability distribution over next tokens
        activations : dict with intermediate values for visualization
        """
        # 1. Embedding lookup (equivalent to one_hot @ W_emb)
        emb = self.W_emb[token_idx].copy()  # (embed_dim,)

        # 2. Hidden layer with tanh activation
        h_pre = emb @ self.W_h + self.b_h   # (hidden_size,)
        h = np.tanh(h_pre)                   # (hidden_size,)

        # 3. Output layer → softmax
        logits = h @ self.W_out + self.b_out  # (vocab_size,)
        probs = self._softmax(logits)          # (vocab_size,)

        activations = {
            "input_onehot": self._one_hot(token_idx),  # (vocab_size,)
            "embedding": emb,                           # (embed_dim,)
            "hidden_pre": h_pre,                        # (hidden_size,) before tanh
            "hidden": h,                                # (hidden_size,) after tanh
            "logits": logits,                           # (vocab_size,) before softmax
            "output": probs,                            # (vocab_size,) probabilities
        }
        return probs, activations

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)  # numerical stability
        e = np.exp(x)
        return e / (e.sum() + 1e-12)

    def _one_hot(self, idx: int) -> np.ndarray:
        v = np.zeros(self.vocab_size)
        v[idx] = 1.0
        return v

    def get_embedding(self, token_idx: int) -> np.ndarray:
        return self.W_emb[token_idx].copy()

    def get_all_embeddings(self) -> np.ndarray:
        return self.W_emb.copy()

    def snapshot(self) -> Dict[str, np.ndarray]:
        """Return a shallow copy of all weights (for history logging)."""
        return {
            "W_emb": self.W_emb.copy(),
            "W_h": self.W_h.copy(),
            "b_h": self.b_h.copy(),
            "W_out": self.W_out.copy(),
            "b_out": self.b_out.copy(),
        }
