"""
Training module - manual backpropagation with SGD, implemented with NumPy.

No autograd frameworks used; all gradients are derived by hand so that
the educational value of seeing "how gradients flow" is preserved.
"""
import numpy as np
from typing import Generator, List, Tuple, Dict

from engine.model import TinyLM


def cross_entropy_loss(probs: np.ndarray, target_idx: int) -> float:
    """Standard cross-entropy loss for a single prediction."""
    return float(-np.log(probs[target_idx] + 1e-12))


def train_step(
    model: TinyLM,
    input_idx: int,
    target_idx: int,
    lr: float = 0.05,
) -> Tuple[float, Dict, Dict]:
    """
    Perform one forward + backward pass and update weights in-place.

    Returns
    -------
    loss        : scalar cross-entropy loss
    activations : intermediate values from forward pass (for visualization)
    gradients   : gradient magnitudes per layer (for visualization)
    """
    # ---- Forward ----
    probs, acts = model.forward(input_idx)
    loss = cross_entropy_loss(probs, target_idx)

    # ---- Backward ----
    # Gradient of cross-entropy + softmax combined: dL/d_logits = probs - onehot(target)
    d_logits = probs.copy()
    d_logits[target_idx] -= 1.0  # (vocab_size,)

    # Output layer
    d_W_out = np.outer(acts["hidden"], d_logits)          # (hidden_size, vocab_size)
    d_b_out = d_logits.copy()                              # (vocab_size,)

    # Hidden layer (tanh derivative: 1 - tanh²)
    d_h = d_logits @ model.W_out.T                         # (hidden_size,)
    d_h_pre = d_h * (1.0 - acts["hidden"] ** 2)           # (hidden_size,)
    d_W_h = np.outer(acts["embedding"], d_h_pre)           # (embed_dim, hidden_size)
    d_b_h = d_h_pre.copy()                                 # (hidden_size,)

    # Embedding layer
    d_emb = d_h_pre @ model.W_h.T                          # (embed_dim,)

    # ---- SGD weight update ----
    model.W_out -= lr * d_W_out
    model.b_out -= lr * d_b_out
    model.W_h -= lr * d_W_h
    model.b_h -= lr * d_b_h
    model.W_emb[input_idx] -= lr * d_emb

    gradients = {
        "W_out_norm": float(np.linalg.norm(d_W_out)),
        "W_h_norm": float(np.linalg.norm(d_W_h)),
        "emb_norm": float(np.linalg.norm(d_emb)),
    }

    return loss, acts, gradients


def train(
    model: TinyLM,
    pairs: List[Tuple[int, int]],
    n_steps: int,
    lr: float = 0.1,
) -> Generator[Dict, None, None]:
    """
    Training loop generator – yields a state dict at every step so the
    caller (Streamlit) can update the UI without blocking.

    Pairs are shuffled at the start of each epoch so the loss curve shows
    a clean downward trend rather than oscillating in a fixed cycle.

    Usage:
        for state in train(model, pairs, n_steps=100):
            update_ui(state)
    """
    if not pairs:
        return

    rng = np.random.default_rng(42)
    pairs_arr = list(pairs)
    epoch_order = list(range(len(pairs_arr)))

    for step in range(n_steps):
        # Reshuffle at the start of every new epoch
        if step % len(pairs_arr) == 0:
            epoch_order = rng.permutation(len(pairs_arr)).tolist()

        input_idx, target_idx = pairs_arr[epoch_order[step % len(pairs_arr)]]
        loss, acts, grads = train_step(model, input_idx, target_idx, lr)

        yield {
            "step": step,
            "total_steps": n_steps,
            "loss": loss,
            "input_idx": input_idx,
            "target_idx": target_idx,
            "activations": acts,
            "gradients": grads,
        }
