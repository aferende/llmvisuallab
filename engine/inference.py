"""
Inference module - next-token prediction, embedding extraction, cosine similarity
and semantic search, all implemented with NumPy.
"""
import numpy as np
from typing import List, Tuple, Dict

from engine.model import TinyLM
from engine.tokenizer import Tokenizer


# ------------------------------------------------------------------
# Next-token prediction
# ------------------------------------------------------------------

def predict_next_tokens(
    model: TinyLM,
    tokenizer: Tokenizer,
    text: str,
    top_k: int = 5,
) -> Tuple[List[Tuple[str, float]], Dict]:
    """
    Given a text, predict the most likely next tokens.

    Returns (top_k_predictions, activations_for_viz).
    Each prediction is (word, probability).
    """
    indices = tokenizer.tokenize(text)
    if not indices:
        return [], {}

    last_idx = indices[-1]
    probs, acts = model.forward(last_idx)

    top_idxs = np.argsort(probs)[::-1][:top_k]
    predictions = [(tokenizer.idx2word[i], float(probs[i])) for i in top_idxs]
    return predictions, acts


# ------------------------------------------------------------------
# Embeddings
# ------------------------------------------------------------------

def get_token_embeddings(
    model: TinyLM,
    tokenizer: Tokenizer,
) -> Tuple[np.ndarray, List[str]]:
    """
    Return embedding matrix and corresponding word labels (skip special tokens).
    Shape: (n_words, embed_dim)
    """
    special = {tokenizer.PAD, tokenizer.UNK}
    words = [w for w in tokenizer.vocab if w not in special]
    words.sort()
    indices = [tokenizer.vocab[w] for w in words]
    embeddings = np.array([model.get_embedding(i) for i in indices])
    return embeddings, words


def get_sentence_embedding(
    model: TinyLM,
    tokenizer: Tokenizer,
    text: str,
) -> np.ndarray:
    """Average embedding of all tokens in a sentence."""
    indices = tokenizer.tokenize(text)
    if not indices:
        return np.zeros(model.embed_dim)
    return np.mean([model.get_embedding(i) for i in indices], axis=0)


# ------------------------------------------------------------------
# Similarity
# ------------------------------------------------------------------

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity between two vectors. Returns a value in [-1, 1]."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def pairwise_cosine(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute the full pairwise cosine similarity matrix.
    Shape: (n, embed_dim) → (n, n)
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    normed = embeddings / norms
    return normed @ normed.T


def pca_reduce(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Reduce embedding dimensions via PCA (SVD). Pure NumPy – no sklearn needed.
    Returns (n_samples, n_components).
    """
    centered = embeddings - embeddings.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ Vt[:n_components].T


# ------------------------------------------------------------------
# Semantic search
# ------------------------------------------------------------------

def semantic_search(
    query: str,
    sentences: List[str],
    model: TinyLM,
    tokenizer: Tokenizer,
) -> List[Tuple[str, float]]:
    """
    Rank sentences by cosine similarity to the query embedding.
    Returns list of (sentence, similarity) sorted descending.
    """
    query_emb = get_sentence_embedding(model, tokenizer, query)
    results = []
    for sent in sentences:
        sent_emb = get_sentence_embedding(model, tokenizer, sent)
        sim = cosine_similarity(query_emb, sent_emb)
        results.append((sent, sim))
    return sorted(results, key=lambda x: x[1], reverse=True)
