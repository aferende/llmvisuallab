"""
Visualization module - all Plotly charts used in the app.

Every chart is built from real model values (weights, activations, embeddings).
Nothing here is decorative-only: colors, sizes, and positions reflect actual data.
"""
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional

from engine.model import TinyLM
from engine.inference import pca_reduce, pairwise_cosine


# ------------------------------------------------------------------
# Color helpers
# ------------------------------------------------------------------

def _rgba(r: int, g: int, b: int, a: float) -> str:
    return f"rgba({r},{g},{b},{a:.2f})"


def _activation_color(val: float, alpha: float = 0.9) -> str:
    """Map a value in [-1, 1] to a blue-white-red color."""
    t = (val + 1.0) / 2.0  # [0, 1]
    t = float(np.clip(t, 0, 1))
    if t < 0.5:
        s = t * 2
        return _rgba(int(50 + 150 * s), int(100 + 50 * s), 230, alpha)
    else:
        s = (t - 0.5) * 2
        return _rgba(230, int(150 - 100 * s), int(50 + 50 * (1 - s)), alpha)


def _prob_color(val: float, alpha: float = 0.9) -> str:
    """Map a probability in [0, 1] to gray-to-green."""
    t = float(np.clip(val * 5, 0, 1))
    return _rgba(int(40 + 40 * t), int(80 + 175 * t), int(60 + 40 * t), alpha)


# ------------------------------------------------------------------
# 3D Neural Network
# ------------------------------------------------------------------

def plot_3d_network(
    model: TinyLM,
    activations: Dict[str, np.ndarray],
    tokenizer,
    input_idx: int,
    target_idx: int,
    step: int,
    loss: float,
    lang_labels: Dict[str, str],
) -> go.Figure:
    """
    Build a real 3D visualisation of TinyLM for a given forward pass.

    Node colours and sizes are derived from actual activation values.
    Edge colours are derived from actual weight magnitudes.
    The active path (input → embedding → hidden → output) is highlighted.
    """
    vocab_size = model.vocab_size
    embed_dim = model.embed_dim
    hidden_size = model.hidden_size

    # Special token indices to exclude from visualisation (PAD, UNK)
    special_idxs = {
        tokenizer.vocab.get(tokenizer.PAD, -1),
        tokenizer.vocab.get(tokenizer.UNK, -2),
    }

    # Input layer: show real vocabulary tokens only (no PAD/UNK), cap at 14
    real_vocab_idxs = [i for i in range(vocab_size) if i not in special_idxs]
    max_vocab_display = min(len(real_vocab_idxs), 14)
    display_in_idxs = real_vocab_idxs[:max_vocab_display]

    # ---- Layer x positions ----
    lx = [0.0, 1.8, 3.6, 5.4]

    # ---- Node positions: y = uniform, z = sinusoidal arc for 3D depth ----
    def layer_positions(n: int, x: float, z_amp: float = 0.6) -> List[Tuple]:
        ys = np.linspace(-n / 2, n / 2, n)
        zs = np.sin(np.linspace(0, np.pi, n)) * z_amp
        return [(x, float(y), float(z)) for y, z in zip(ys, zs)]

    pos_in = layer_positions(max_vocab_display, lx[0], 0.5)
    pos_em = layer_positions(embed_dim, lx[1], 0.7)
    pos_hid = layer_positions(hidden_size, lx[2], 1.0)
    # Show top-8 output nodes by probability, excluding special tokens
    out_probs = activations["output"]
    all_sorted_idxs = np.argsort(out_probs)[::-1].tolist()
    top_out_idxs = [i for i in all_sorted_idxs if i not in special_idxs][:8]
    pos_out = layer_positions(len(top_out_idxs), lx[3], 0.5)

    fig = go.Figure()
    bg = "rgb(12,14,26)"

    # ================================================================
    # EDGES – drawn as one trace each with None separators
    # ================================================================

    def add_edge_trace(pairs, color, width, name=""):
        xs, ys, zs = [], [], []
        for p0, p1 in pairs:
            xs += [p0[0], p1[0], None]
            ys += [p0[1], p1[1], None]
            zs += [p0[2], p1[2], None]
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(color=color, width=width),
            hoverinfo="none",
            showlegend=False,
            name=name,
        ))

    # Background edges: Embedding → Hidden (all, low opacity)
    bg_pairs_eh = [(pos_em[i], pos_hid[j])
                   for i in range(embed_dim) for j in range(hidden_size)]
    add_edge_trace(bg_pairs_eh, "rgba(80,100,180,0.08)", 0.8)

    # Background edges: Hidden → Output (all, low opacity)
    bg_pairs_ho = [(pos_hid[i], pos_out[j])
                   for i in range(hidden_size) for j in range(len(top_out_idxs))]
    add_edge_trace(bg_pairs_ho, "rgba(80,100,180,0.08)", 0.8)

    # Active path: input_idx → its embedding connections (W_emb row)
    in_node = pos_in[min(input_idx, max_vocab_display - 1)]
    active_emb_pairs = [(in_node, pos_em[e]) for e in range(embed_dim)]
    add_edge_trace(active_emb_pairs, "rgba(0,200,255,0.55)", 2.0, "Input→Emb")

    # Active path: embedding → hidden (weighted by W_h magnitude)
    max_w = np.abs(model.W_h).max() + 1e-6
    active_eh_pairs = []
    for e in range(embed_dim):
        for h in range(hidden_size):
            w = abs(model.W_h[e, h]) / max_w
            if w > 0.3:
                active_eh_pairs.append((pos_em[e], pos_hid[h]))
    add_edge_trace(active_eh_pairs, "rgba(100,220,180,0.35)", 1.5, "Emb→Hid")

    # Active path: hidden → top output nodes
    max_wo = np.abs(model.W_out).max() + 1e-6
    active_ho_pairs = []
    for h in range(hidden_size):
        for oi, out_idx in enumerate(top_out_idxs):
            w = abs(model.W_out[h, out_idx]) / max_wo
            if w > 0.3:
                active_ho_pairs.append((pos_hid[h], pos_out[oi]))
    add_edge_trace(active_ho_pairs, "rgba(255,160,60,0.35)", 1.5, "Hid→Out")

    # ================================================================
    # NODES
    # ================================================================

    def scatter_nodes(positions, colors, sizes, labels, hover_texts, name):
        fig.add_trace(go.Scatter3d(
            x=[p[0] for p in positions],
            y=[p[1] for p in positions],
            z=[p[2] for p in positions],
            mode="markers+text",
            marker=dict(
                size=sizes,
                color=colors,
                line=dict(width=1.2, color="rgba(255,255,255,0.35)"),
            ),
            text=labels,
            textposition="middle right",
            textfont=dict(size=10, color="rgba(230,235,255,0.95)"),
            hovertext=hover_texts,
            hovertemplate="%{hovertext}<extra></extra>",
            name=name,
            showlegend=False,
        ))

    # ---- Input layer ----
    in_colors, in_sizes, in_labels, in_hover = [], [], [], []
    for i in display_in_idxs:
        is_active = (i == input_idx)
        word = tokenizer.idx2word.get(i, "?")
        in_colors.append("rgba(0,230,130,0.95)" if is_active else "rgba(70,90,160,0.55)")
        in_sizes.append(14 if is_active else 7)
        in_labels.append(f"▶ {word}" if is_active else word)
        in_hover.append(
            f"<b>{'▶ TOKEN ATTIVO' if is_active else 'Token'}: {word}</b><br>"
            f"Indice vocabolario: {i}<br>"
            f"{'← questo è il token in input al modello' if is_active else ''}"
        )
    scatter_nodes(pos_in, in_colors, in_sizes, in_labels, in_hover,
                  lang_labels.get("layer_input", "Input"))

    # ---- Embedding layer ----
    emb_vals = activations["embedding"]  # (embed_dim,)
    emb_colors = [_activation_color(float(v)) for v in emb_vals]
    emb_sizes = [8 + int(abs(float(v)) * 6) for v in emb_vals]
    emb_labels = [f"e{i}" for i in range(len(emb_vals))]
    emb_hover = [
        f"<b>Embedding dim {i}</b><br>"
        f"Valore: <b>{v:.4f}</b><br>"
        f"Questo numero rappresenta una caratteristica<br>della parola in input."
        for i, v in enumerate(emb_vals)
    ]
    scatter_nodes(pos_em, emb_colors, emb_sizes, emb_labels, emb_hover,
                  lang_labels.get("layer_emb", "Embedding"))

    # ---- Hidden layer ----
    hid_vals = activations["hidden"]  # (hidden_size,)
    hid_colors = [_activation_color(float(v)) for v in hid_vals]
    hid_sizes = [7 + int(abs(float(v)) * 7) for v in hid_vals]
    hid_labels = [f"h{i}" for i in range(len(hid_vals))]
    hid_hover = [
        f"<b>Neurone hidden {i}</b><br>"
        f"Attivazione (tanh): <b>{v:.4f}</b><br>"
        f"Range: -1 (inibito) → 0 (neutro) → +1 (attivo)<br>"
        f"{'🔴 fortemente attivo' if abs(v) > 0.6 else '⚪ attivazione moderata' if abs(v) > 0.3 else '🔵 quasi silente'}"
        for i, v in enumerate(hid_vals)
    ]
    scatter_nodes(pos_hid, hid_colors, hid_sizes, hid_labels, hid_hover,
                  lang_labels.get("layer_hid", "Hidden"))

    # ---- Output layer (top-8) ----
    out_colors, out_sizes, out_labels, out_hover = [], [], [], []
    for oi, out_idx in enumerate(top_out_idxs):
        p = float(out_probs[out_idx])
        is_target = (out_idx == target_idx)
        word = tokenizer.idx2word.get(out_idx, "?")
        out_colors.append("rgba(255,215,0,0.97)" if is_target else _prob_color(p))
        out_sizes.append(15 if is_target else 6 + int(p * 35))
        out_labels.append(f"★ {word}" if is_target else f"{word}")
        out_hover.append(
            f"<b>{'★ TARGET: ' if is_target else ''}{word}</b><br>"
            f"Probabilità: <b>{p:.4f}</b> ({int(p*100)}%)<br>"
            f"{'← questo è il token che il modello deve imparare a predire' if is_target else ''}"
        )
    scatter_nodes(pos_out, out_colors, out_sizes, out_labels, out_hover,
                  lang_labels.get("layer_out", "Output"))

    # ================================================================
    # Layer name annotations (above each layer)
    # ================================================================
    layer_names = [
        lang_labels.get("layer_input", "Input"),
        lang_labels.get("layer_emb", "Embedding"),
        lang_labels.get("layer_hid", "Hidden"),
        lang_labels.get("layer_out", "Output"),
    ]
    layer_sizes = [
        f"({max_vocab_display} token)",
        f"({embed_dim} dim)",
        f"({model.hidden_size} neuroni)",
        "(top-8 token)",
    ]
    for xi, name, sz, pos_list in zip(lx, layer_names, layer_sizes,
                                      [pos_in, pos_em, pos_hid, pos_out]):
        top_z = max(p[2] for p in pos_list) + 0.7
        # NOTE: Plotly 3D does NOT render HTML — plain text only
        fig.add_trace(go.Scatter3d(
            x=[xi], y=[0], z=[top_z],
            mode="text",
            text=[f"{name} {sz}"],
            textfont=dict(size=11, color="rgba(180,210,255,0.95)"),
            hoverinfo="none",
            showlegend=False,
        ))

    # ================================================================
    # Layout
    # ================================================================
    step_label = lang_labels.get("step_label", "Step")
    loss_label = lang_labels.get("loss_label", "Loss")

    fig.update_layout(
        title=dict(
            text=f"<b>{step_label} {step + 1}</b>  |  {loss_label}: <b>{loss:.4f}</b>  "
                 f"|  Input: <b>{tokenizer.idx2word.get(input_idx,'?')}</b>  →  "
                 f"Target: <b>{tokenizer.idx2word.get(target_idx,'?')}</b>",
            font=dict(color="rgba(200,220,255,0.95)", size=13),
            x=0.5,
        ),
        scene=dict(
            bgcolor=bg,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=dict(eye=dict(x=1.6, y=-1.2, z=0.8)),
        ),
        paper_bgcolor=bg,
        margin=dict(l=0, r=0, t=50, b=0),
        height=510,
    )

    return fig


# ------------------------------------------------------------------
# Loss curve
# ------------------------------------------------------------------

def plot_loss_curve(
    losses: List[float],
    lang_labels: Dict[str, str],
) -> go.Figure:
    """
    Loss-over-steps chart with:
    - Raw instantaneous loss (light, semi-transparent)
    - Rolling-average smoothed trend line (bold) — shows clear downward trend
    """
    fig = go.Figure()
    xs = list(range(1, len(losses) + 1))
    loss_label = lang_labels.get("loss_label", "Loss")
    step_label = lang_labels.get("step_label", "Step")

    # ── Raw loss — thin, semi-transparent background ───────────────────
    fig.add_trace(go.Scatter(
        y=losses, x=xs,
        mode="lines",
        line=dict(color="rgba(0,212,255,0.25)", width=1),
        name=f"{loss_label} (raw)",
        hovertemplate=f"{step_label} %{{x}}: %{{y:.4f}}<extra></extra>",
    ))

    # ── Smoothed trend line ─────────────────────────────────────────────
    if len(losses) >= 5:
        window = max(5, len(losses) // 10)
        kernel = np.ones(window) / window
        smoothed = np.convolve(losses, kernel, mode="valid")
        xs_sm = list(range(window, len(losses) + 1))
        fig.add_trace(go.Scatter(
            y=smoothed.tolist(), x=xs_sm,
            mode="lines",
            line=dict(color="#00d4ff", width=3),
            fill="tozeroy",
            fillcolor="rgba(0,212,255,0.07)",
            name=f"{loss_label} (trend)",
            hovertemplate=f"{step_label} %{{x}}: %{{y:.4f}}<extra></extra>",
        ))

    # ── Best-loss star marker ────────────────────────────────────────────
    if len(losses) > 1:
        best_idx = int(np.argmin(losses))
        fig.add_trace(go.Scatter(
            x=[xs[best_idx]], y=[losses[best_idx]],
            mode="markers+text",
            marker=dict(size=10, color="#00ff88", symbol="star"),
            text=["min"],
            textposition="top center",
            textfont=dict(size=9, color="#00ff88"),
            showlegend=False,
            hovertemplate=f"Min: {losses[best_idx]:.4f}<extra></extra>",
        ))

    fig.update_layout(
        xaxis_title=step_label,
        yaxis_title=loss_label,
        paper_bgcolor="rgb(12,14,26)",
        plot_bgcolor="rgb(18,20,36)",
        font=dict(color="rgba(200,220,255,0.85)", size=11),
        legend=dict(
            orientation="h", yanchor="top", y=1.0, xanchor="right", x=1.0,
            font=dict(size=9), bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=50, r=20, t=30, b=40),
        height=260,
    )
    fig.update_xaxes(gridcolor="rgba(100,120,200,0.15)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(100,120,200,0.15)", zeroline=False)
    return fig


# ------------------------------------------------------------------
# 2D Embedding scatter (PCA)
# ------------------------------------------------------------------

def plot_embeddings_2d(
    embeddings: np.ndarray,
    labels: List[str],
    query_label: Optional[str],
    query_emb: Optional[np.ndarray],
    lang_labels: Dict[str, str],
) -> go.Figure:
    """2D scatter of word embeddings reduced to 2D via PCA."""
    all_embs = embeddings
    all_labels = list(labels)

    if query_emb is not None and query_label:
        all_embs = np.vstack([embeddings, query_emb.reshape(1, -1)])
        all_labels = list(labels) + [f"❓ {query_label}"]

    if all_embs.shape[1] > 2:
        coords = pca_reduce(all_embs, 2)
    elif all_embs.shape[1] == 2:
        coords = all_embs
    else:
        coords = np.hstack([all_embs, np.zeros((len(all_embs), 2 - all_embs.shape[1]))])

    n_vocab = len(labels)
    colors_vocab = ["rgba(100,180,255,0.85)"] * n_vocab
    sizes_vocab = [11] * n_vocab
    colors_query = ["rgba(255,200,0,0.95)"] if query_emb is not None else []
    sizes_query = [17] if query_emb is not None else []

    all_colors = colors_vocab + colors_query
    all_sizes = sizes_vocab + sizes_query

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        mode="markers+text",
        text=all_labels,
        textposition="top center",
        textfont=dict(size=11, color="rgba(220,235,255,0.95)"),
        marker=dict(
            color=all_colors,
            size=all_sizes,
            line=dict(width=0.8, color="rgba(255,255,255,0.35)"),
        ),
        hovertemplate="<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text=lang_labels.get("emb_2d_title", "Token Embeddings (PCA 2D)"),
            font=dict(color="rgba(200,220,255,0.9)", size=12),
            x=0.5,
        ),
        paper_bgcolor="rgb(12,14,26)",
        plot_bgcolor="rgb(18,20,36)",
        font=dict(color="rgba(200,220,255,0.85)"),
        xaxis=dict(title="PC1", gridcolor="rgba(100,120,200,0.15)", zeroline=False),
        yaxis=dict(title="PC2", gridcolor="rgba(100,120,200,0.15)", zeroline=False),
        margin=dict(l=30, r=20, t=40, b=30),
        height=420,
    )
    return fig


# ------------------------------------------------------------------
# 3D Embedding scatter (PCA)
# ------------------------------------------------------------------

def plot_embeddings_3d(
    embeddings: np.ndarray,
    labels: List[str],
    query_label: Optional[str],
    query_emb: Optional[np.ndarray],
    lang_labels: Dict[str, str],
) -> go.Figure:
    """
    3D vector space: each word is drawn as an arrow from the origin (0,0,0)
    to its PCA-reduced 3D position. Cartesian axes X/Y/Z are clearly visible.
    """
    bg = "rgb(12, 14, 26)"
    all_embs = embeddings
    all_labels = list(labels)

    if query_emb is not None and query_label:
        all_embs = np.vstack([embeddings, query_emb.reshape(1, -1)])
        all_labels = list(labels) + [f"? {query_label}"]

    if all_embs.shape[1] > 3:
        coords = pca_reduce(all_embs, 3)
    elif all_embs.shape[1] == 3:
        coords = all_embs.copy()
    else:
        pad = np.zeros((len(all_embs), 3 - all_embs.shape[1]))
        coords = np.hstack([all_embs, pad])

    n_vocab = len(labels)

    fig = go.Figure()

    # ── Cartesian axes (X=red, Y=green, Z=blue) ───────────────────────
    axis_range = float(np.abs(coords).max()) * 1.2 + 0.1
    for direction, color, axis_lbl in [
        ([1, 0, 0], "rgba(255,80,80,0.8)", "PC1"),
        ([0, 1, 0], "rgba(80,220,80,0.8)", "PC2"),
        ([0, 0, 1], "rgba(80,130,255,0.8)", "PC3"),
    ]:
        end = [d * axis_range for d in direction]
        fig.add_trace(go.Scatter3d(
            x=[0, end[0]], y=[0, end[1]], z=[0, end[2]],
            mode="lines+text",
            line=dict(color=color, width=3),
            text=["", axis_lbl],
            textfont=dict(size=10, color=color),
            hoverinfo="none",
            showlegend=False,
        ))
        # Negative axis stub (lighter)
        neg = [-d * axis_range * 0.4 for d in direction]
        fig.add_trace(go.Scatter3d(
            x=[0, neg[0]], y=[0, neg[1]], z=[0, neg[2]],
            mode="lines",
            line=dict(color=color.replace("0.8", "0.2"), width=1),
            hoverinfo="none",
            showlegend=False,
        ))

    # ── Origin marker ──────────────────────────────────────────────────
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers+text",
        marker=dict(size=6, color="white", symbol="circle"),
        text=["O"],
        textfont=dict(size=9, color="rgba(200,200,200,0.6)"),
        hoverinfo="none",
        showlegend=False,
    ))

    # ── Vector lines from origin ───────────────────────────────────────
    for i, (coord, label) in enumerate(zip(coords, all_labels)):
        is_query = (i >= n_vocab)
        lc = "rgba(255,200,0,0.85)" if is_query else "rgba(80,140,255,0.50)"
        lw = 3 if is_query else 1.5
        fig.add_trace(go.Scatter3d(
            x=[0, coord[0]], y=[0, coord[1]], z=[0, coord[2]],
            mode="lines",
            line=dict(color=lc, width=lw),
            hoverinfo="skip",
            showlegend=False,
        ))

    # ── Tip markers + word labels ──────────────────────────────────────
    tip_colors = ["rgba(100,180,255,0.9)"] * n_vocab
    tip_sizes = [7] * n_vocab
    if query_emb is not None:
        tip_colors.append("rgba(255,200,0,0.95)")
        tip_sizes.append(13)

    fig.add_trace(go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode="markers+text",
        text=all_labels,
        textposition="top center",
        textfont=dict(size=10, color="rgba(220,235,255,0.95)"),
        marker=dict(
            color=tip_colors,
            size=tip_sizes,
            line=dict(width=0.8, color="rgba(255,255,255,0.3)"),
        ),
        hovertemplate="<b>%{text}</b><br>(%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>",
        showlegend=False,
    ))

    fig.update_layout(
        title=dict(
            text=lang_labels.get("emb_3d_title", "Token Embeddings (PCA 3D)"),
            font=dict(color="rgba(200,220,255,0.9)", size=12),
            x=0.5,
        ),
        scene=dict(
            bgcolor=bg,
            xaxis=dict(
                title="PC1", showgrid=True,
                gridcolor="rgba(255,80,80,0.12)",
                zeroline=True, zerolinecolor="rgba(255,80,80,0.3)",
                showbackground=False,
                tickfont=dict(color="rgba(200,200,200,0.5)", size=8),
            ),
            yaxis=dict(
                title="PC2", showgrid=True,
                gridcolor="rgba(80,220,80,0.12)",
                zeroline=True, zerolinecolor="rgba(80,220,80,0.3)",
                showbackground=False,
                tickfont=dict(color="rgba(200,200,200,0.5)", size=8),
            ),
            zaxis=dict(
                title="PC3", showgrid=True,
                gridcolor="rgba(80,130,255,0.12)",
                zeroline=True, zerolinecolor="rgba(80,130,255,0.3)",
                showbackground=False,
                tickfont=dict(color="rgba(200,200,200,0.5)", size=8),
            ),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.6, y=1.4, z=0.9)),
        ),
        paper_bgcolor=bg,
        margin=dict(l=0, r=0, t=40, b=0),
        height=500,
    )
    return fig


# ------------------------------------------------------------------
# 3D Cosine Similarity on Unit Sphere  ← NEW
# ------------------------------------------------------------------

def plot_cosine_similarity_3d(
    embeddings: np.ndarray,
    labels: List[str],
    word1_idx: int,
    word2_idx: int,
    word1_label: str,
    word2_label: str,
    similarity: float,
    lang_labels: Dict[str, str],
) -> go.Figure:
    """
    3D unit-sphere visualization showing ALL word embedding vectors.

    - Semi-transparent sphere as reference surface
    - Every word = one vector arrow from origin to sphere surface
    - Selected word1 → bright green  (#00e887)
    - Selected word2 → bright red    (#ff4d6d)
    - All other words → soft blue
    - Golden dashed arc showing the angle between the two selected vectors
    - Word labels at every vector tip
    - Cosine value and angle in title
    """
    bg = "rgb(10, 12, 24)"

    # ── PCA to 3D then normalise to unit sphere ──────────────────────
    if embeddings.shape[1] > 3:
        coords3d = pca_reduce(embeddings, 3)
    elif embeddings.shape[1] == 3:
        coords3d = embeddings.copy()
    else:
        pad = np.zeros((len(embeddings), 3 - embeddings.shape[1]))
        coords3d = np.hstack([embeddings, pad])

    norms = np.linalg.norm(coords3d, axis=1, keepdims=True) + 1e-10
    unit = coords3d / norms          # all vectors now sit on the unit sphere

    fig = go.Figure()

    # ── Sphere surface ────────────────────────────────────────────────
    phi = np.linspace(0, 2 * np.pi, 50)
    theta = np.linspace(0, np.pi, 50)
    xs = np.outer(np.cos(phi), np.sin(theta))
    ys = np.outer(np.sin(phi), np.sin(theta))
    zs = np.outer(np.ones_like(phi), np.cos(theta))

    fig.add_trace(go.Surface(
        x=xs, y=ys, z=zs,
        opacity=0.07,
        colorscale=[[0, "rgb(20,40,110)"], [1, "rgb(50,90,200)"]],
        showscale=False,
        hoverinfo="skip",
        name="sphere",
    ))

    # ── Guide circles (equator + 2 meridians) ────────────────────────
    circ = np.linspace(0, 2 * np.pi, 120)
    circle_color = "rgba(80,110,200,0.18)"
    for cx, cy, cz in [
        (np.cos(circ), np.sin(circ), np.zeros_like(circ)),       # equator
        (np.cos(circ), np.zeros_like(circ), np.sin(circ)),        # meridian YZ
        (np.zeros_like(circ), np.cos(circ), np.sin(circ)),        # meridian XZ
    ]:
        fig.add_trace(go.Scatter3d(
            x=cx, y=cy, z=cz,
            mode="lines",
            line=dict(color=circle_color, width=1),
            hoverinfo="none",
            showlegend=False,
        ))

    # ── Axis stubs ────────────────────────────────────────────────────
    for axis in ["x", "y", "z"]:
        ax = [1.15 if a == axis else 0 for a in "xyz"]
        fig.add_trace(go.Scatter3d(
            x=[0, ax[0]], y=[0, ax[1]], z=[0, ax[2]],
            mode="lines+text",
            line=dict(color="rgba(150,170,220,0.25)", width=1),
            text=["", axis.upper()],
            textfont=dict(size=9, color="rgba(150,170,220,0.5)"),
            hoverinfo="none",
            showlegend=False,
        ))

    # ── All word vectors (lines + tips) ──────────────────────────────
    # Draw lines first (so labels are on top)
    for i, coord in enumerate(unit):
        is_w1 = (i == word1_idx)
        is_w2 = (i == word2_idx)

        if is_w1:
            lc, lw = "#00e887", 5
        elif is_w2:
            lc, lw = "#ff4d6d", 5
        else:
            lc, lw = "rgba(80,120,220,0.40)", 1.5

        fig.add_trace(go.Scatter3d(
            x=[0, coord[0]], y=[0, coord[1]], z=[0, coord[2]],
            mode="lines",
            line=dict(color=lc, width=lw),
            hoverinfo="skip",
            showlegend=False,
        ))

    # ── Word tip markers + labels ─────────────────────────────────────
    pt_colors, pt_sizes = [], []
    for i in range(len(labels)):
        if i == word1_idx:
            pt_colors.append("#00e887")
            pt_sizes.append(11)
        elif i == word2_idx:
            pt_colors.append("#ff4d6d")
            pt_sizes.append(11)
        else:
            pt_colors.append("rgba(120,160,255,0.80)")
            pt_sizes.append(6)

    fig.add_trace(go.Scatter3d(
        x=unit[:, 0], y=unit[:, 1], z=unit[:, 2],
        mode="markers+text",
        text=labels,
        textposition="top center",
        textfont=dict(size=10, color="rgba(220,235,255,0.95)"),
        marker=dict(
            size=pt_sizes,
            color=pt_colors,
            line=dict(width=0.6, color="rgba(255,255,255,0.3)"),
        ),
        hovertemplate="<b>%{text}</b><extra></extra>",
        showlegend=False,
    ))

    # ── Arc between the two selected vectors ─────────────────────────
    v1 = unit[word1_idx]
    v2 = unit[word2_idx]
    arc_t = np.linspace(0, 1, 80)
    arc_raw = np.array([(1 - t) * v1 + t * v2 for t in arc_t])
    arc_n = np.linalg.norm(arc_raw, axis=1, keepdims=True) + 1e-10
    arc_pts = arc_raw / arc_n  # keep on sphere

    fig.add_trace(go.Scatter3d(
        x=arc_pts[:, 0], y=arc_pts[:, 1], z=arc_pts[:, 2],
        mode="lines",
        line=dict(color="rgba(255,210,0,0.92)", width=4),
        hoverinfo="none",
        showlegend=False,
        name="angle-arc",
    ))

    # ── Angle label at arc midpoint ───────────────────────────────────
    mid = arc_pts[len(arc_pts) // 2] * 1.28     # slightly outside sphere
    angle_deg = float(np.degrees(np.arccos(np.clip(similarity, -1.0, 1.0))))

    fig.add_trace(go.Scatter3d(
        x=[mid[0]], y=[mid[1]], z=[mid[2]],
        mode="text",
        text=[f"θ = {angle_deg:.1f}°"],
        textfont=dict(size=14, color="rgba(255,215,0,1.0)"),
        hoverinfo="none",
        showlegend=False,
    ))

    # ── Shaded "pizza slice" between the two vectors ──────────────────
    n_slice = 40
    slice_t = np.linspace(0, 1, n_slice)
    # Fan from origin through arc
    fan_x, fan_y, fan_z = [0], [0], [0]
    for t in slice_t:
        p = (1 - t) * v1 + t * v2
        p = p / (np.linalg.norm(p) + 1e-10)
        fan_x.append(p[0]); fan_y.append(p[1]); fan_z.append(p[2])
    fan_x.append(0); fan_y.append(0); fan_z.append(0)

    fig.add_trace(go.Scatter3d(
        x=fan_x, y=fan_y, z=fan_z,
        mode="lines",
        line=dict(color="rgba(255,215,0,0.12)", width=1),
        hoverinfo="none",
        showlegend=False,
    ))

    # ── Origin dot ────────────────────────────────────────────────────
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers",
        marker=dict(size=5, color="white", symbol="circle"),
        hoverinfo="none",
        showlegend=False,
    ))

    # ── Layout ────────────────────────────────────────────────────────
    sim_label = lang_labels.get("similarity_label", "Cosine similarity")
    fig.update_layout(
        title=dict(
            text=(
                f"<b>cos({word1_label}, {word2_label}) = {similarity:.4f}</b>"
                f"   |   θ = {angle_deg:.1f}°"
                f"   |   <span style='color:#00e887'>{word1_label}</span>"
                f"  vs  <span style='color:#ff4d6d'>{word2_label}</span>"
            ),
            font=dict(color="rgba(200,220,255,0.95)", size=13),
            x=0.5,
        ),
        scene=dict(
            bgcolor=bg,
            xaxis=dict(visible=False, range=[-1.5, 1.5]),
            yaxis=dict(visible=False, range=[-1.5, 1.5]),
            zaxis=dict(visible=False, range=[-1.5, 1.5]),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.5, y=1.3, z=0.9)),
        ),
        paper_bgcolor=bg,
        margin=dict(l=0, r=0, t=55, b=0),
        height=540,
    )

    return fig


# ------------------------------------------------------------------
# Similarity heatmap
# ------------------------------------------------------------------

def plot_similarity_heatmap(
    sim_matrix: np.ndarray,
    labels: List[str],
    lang_labels: Dict[str, str],
) -> go.Figure:
    """Heatmap of pairwise cosine similarities between word embeddings."""
    fig = go.Figure(data=go.Heatmap(
        z=sim_matrix,
        x=labels,
        y=labels,
        colorscale="RdBu",
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(sim_matrix, 2),
        texttemplate="%{text}",
        textfont=dict(size=9),
        hovertemplate="%{y} – %{x}: %{z:.3f}<extra></extra>",
        colorbar=dict(
            title=dict(
                text="Cosine",
                side="right",
                font=dict(color="rgba(200,220,255,0.8)", size=10),
            ),
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["-1 opposite", "-0.5", "0 orthogonal", "+0.5", "+1 identical"],
            tickfont=dict(color="rgba(200,220,255,0.75)", size=9),
            outlinewidth=0,
            thickness=14,
        ),
    ))
    fig.update_layout(
        title=dict(
            text=lang_labels.get("heatmap_title", "Pairwise Cosine Similarity"),
            font=dict(color="rgba(200,220,255,0.9)", size=12),
            x=0.5,
        ),
        paper_bgcolor="rgb(12,14,26)",
        plot_bgcolor="rgb(18,20,36)",
        font=dict(color="rgba(200,220,255,0.85)", size=10),
        margin=dict(l=60, r=100, t=50, b=60),
        height=420,
    )
    return fig


# ------------------------------------------------------------------
# Inference animation
# ------------------------------------------------------------------

def plot_inference_animation(
    model: TinyLM,
    tokenizer,
    input_idx: int,
    activations: Dict[str, np.ndarray],
    predictions: List[Tuple[str, float]],
    lang_labels: Dict[str, str],
) -> go.Figure:
    """
    6-frame Plotly animation: shows how one token flows through TinyLM.

    Frame 0 — Network at rest
    Frame 1 — Input token activated (green)
    Frame 2 — Signal propagates to Embedding layer
    Frame 3 — Signal propagates to Hidden layer
    Frame 4 — Signal propagates to Output layer (probabilities appear)
    Frame 5 — Winner path highlighted in green end-to-end

    Trace layout (7 traces):
      0  background edges       – static gray lines
      1  active positive edges  – orange, updated per frame
      2  active negative edges  – steel-blue, updated per frame
      3  winner path edges      – green, only in frame 5
      4  all nodes              – colors/sizes updated per frame
      5  output probability labels – text, visible in frames 4-5
      6  input token label      – static, shows the query token
    """
    BG   = "rgb(12,14,26)"
    GRAY = "rgba(80,90,140,0.20)"
    ORANGE = "rgba(255,140,30,{a})"
    STEEL  = "rgba(60,130,230,{a})"
    GREEN  = "rgba(0,230,120,{a})"

    V = model.vocab_size
    E = model.embed_dim
    H = model.hidden_size
    top_k = min(len(predictions), 8)
    max_in = min(V, 14)

    # ---- activations & weights ----------------------------------------
    emb_vals = activations["embedding"]   # [E]
    hid_vals = activations["hidden"]      # [H]

    W_emb = model.W_emb   # [V, E]
    W_h   = model.W_h     # [E, H]
    W_out = model.W_out   # [H, V]

    top_words = [w for w, _ in predictions[:top_k]]
    top_probs = [p for _, p in predictions[:top_k]]
    top_out_idx = [
        min(tokenizer.vocab.get(w, 1), V - 1)
        for w in top_words
    ]

    # ---- 2-D node positions ------------------------------------------
    def col_y(n: int) -> List[float]:
        return list(np.linspace(0.95, 0.05, n)) if n > 1 else [0.5]

    in_y  = col_y(max_in);  in_x  = [0.00] * max_in
    em_y  = col_y(E);       em_x  = [1.00] * E
    hd_y  = col_y(H);       hd_x  = [2.00] * H
    ot_y  = col_y(top_k);   ot_x  = [3.00] * top_k

    # active input position
    act_in = min(input_idx, max_in - 1)

    # ---- color helpers -----------------------------------------------
    def _act_color(v: float, alpha: float = 0.9) -> str:
        t = float(np.clip((v + 1.0) / 2.0, 0, 1))
        if t < 0.5:
            s = t * 2
            r, g, b = int(50 + 150 * s), int(100 + 50 * s), 230
        else:
            s = (t - 0.5) * 2
            r, g, b = 230, int(150 - 100 * s), int(50 + 50 * (1 - s))
        return f"rgba({r},{g},{b},{alpha:.2f})"

    def _prob_color(p: float) -> str:
        t = float(np.clip(p * 4, 0, 1))
        return f"rgba({int(40+80*t)},{int(80+175*t)},{int(60+80*t)},0.92)"

    GRAY_NODE = "rgba(80,90,140,0.45)"
    GREEN_IN  = "rgba(0,230,130,0.97)"
    GREEN_WIN = "rgba(30,255,120,1.0)"

    # ---- edge builders -----------------------------------------------
    def _segs(src_xy, tgt_xy, wmat):
        """
        Returns (pos_xs, pos_ys, neg_xs, neg_ys) split by weight sign.
        wmat shape: [n_src, n_tgt]
        """
        pos_x, pos_y, neg_x, neg_y = [], [], [], []
        if wmat.size == 0:
            return pos_x, pos_y, neg_x, neg_y
        mx = float(np.max(np.abs(wmat))) or 1.0
        threshold = mx * 0.08          # skip near-zero edges for clarity
        for i, (x0, y0) in enumerate(src_xy):
            for j, (x1, y1) in enumerate(tgt_xy):
                w = float(wmat[i, j])
                if abs(w) < threshold:
                    continue
                if w >= 0:
                    pos_x += [x0, x1, None]; pos_y += [y0, y1, None]
                else:
                    neg_x += [x0, x1, None]; neg_y += [y0, y1, None]
        return pos_x, pos_y, neg_x, neg_y

    def _winner_segs():
        wx, wy = [], []
        # in → em (active input only)
        for j in range(E):
            wx += [in_x[act_in], em_x[j], None]
            wy += [in_y[act_in], em_y[j], None]
        # em → hid
        for i in range(E):
            for j in range(H):
                wx += [em_x[i], hd_x[j], None]
                wy += [em_y[i], hd_y[j], None]
        # hid → winner (index 0 in top-k)
        for i in range(H):
            wx += [hd_x[i], ot_x[0], None]
            wy += [hd_y[i], ot_y[0], None]
        return wx, wy

    # ---- background edges (all connections) --------------------------
    bg_x, bg_y = [], []
    # in → em (all)
    for i in range(max_in):
        for j in range(E):
            bg_x += [in_x[i], em_x[j], None]
            bg_y += [in_y[i], em_y[j], None]
    # em → hid
    for i in range(E):
        for j in range(H):
            bg_x += [em_x[i], hd_x[j], None]
            bg_y += [em_y[i], hd_y[j], None]
    # hid → out top-k
    for i in range(H):
        for j in range(top_k):
            bg_x += [hd_x[i], ot_x[j], None]
            bg_y += [hd_y[i], ot_y[j], None]

    # ---- node labels -------------------------------------------------
    in_labels = [tokenizer.idx2word.get(i, "?") for i in range(max_in)]
    em_labels = [f"e{j}" for j in range(E)]
    hd_labels = [f"h{j}" for j in range(H)]
    ot_labels_full = [
        f"{w}<br>{p * 100:.1f}%" for w, p in zip(top_words, top_probs)
    ]
    ot_labels_empty = [""] * top_k

    # ---- per-frame state -----------------------------------------
    # returns (pos_xs, pos_ys, neg_xs, neg_ys,
    #          win_xs, win_ys, node_colors, node_sizes,
    #          node_text, out_label_text)
    def _frame_state(f: int):
        # --- default: everything gray, no active edges
        nc = [GRAY_NODE] * (max_in + E + H + top_k)
        ns = [7]         * (max_in + E + H + top_k)
        px, py, nx, ny = [], [], [], []
        wx, wy = [], []
        out_txt = ot_labels_empty

        if f >= 1:   # input node active
            nc[act_in] = GREEN_IN
            ns[act_in] = 16

        if f >= 2:   # embedding activated
            w_in_em = W_emb[min(input_idx, V - 1), :].reshape(1, E)
            px, py, nx, ny = _segs(
                [(in_x[act_in], in_y[act_in])], list(zip(em_x, em_y)), w_in_em
            )
            for j in range(E):
                nc[max_in + j] = _act_color(float(emb_vals[j]))
                ns[max_in + j] = 8 + int(abs(float(emb_vals[j])) * 9)

        if f >= 3:   # hidden activated
            # contribution = diag(emb_vals) @ W_h  →  [E, H]
            wmat_eh = np.outer(emb_vals, np.ones(H)) * W_h
            px, py, nx, ny = _segs(
                list(zip(em_x, em_y)), list(zip(hd_x, hd_y)), wmat_eh
            )
            for j in range(H):
                nc[max_in + E + j] = _act_color(float(hid_vals[j]))
                ns[max_in + E + j] = 7 + int(abs(float(hid_vals[j])) * 11)

        if f >= 4:   # output activated
            # contributions: [H, top_k]
            wmat_ho = np.column_stack([
                hid_vals * W_out[:, oi] for oi in top_out_idx
            ])
            px, py, nx, ny = _segs(
                list(zip(hd_x, hd_y)), list(zip(ot_x, ot_y)), wmat_ho
            )
            for j in range(top_k):
                nc[max_in + E + H + j] = _prob_color(top_probs[j])
                ns[max_in + E + H + j] = 9 + int(top_probs[j] * 35)
            out_txt = ot_labels_full

        if f == 5:   # winner path green
            wx, wy = _winner_segs()
            nc[max_in + E + H] = GREEN_WIN
            ns[max_in + E + H] = 22
            out_txt = ot_labels_full

        # flatten node positions and labels
        all_x = in_x + em_x + hd_x + ot_x
        all_y = in_y + em_y + hd_y + ot_y
        all_lbl = in_labels + em_labels + hd_labels + out_txt

        return px, py, nx, ny, wx, wy, nc, ns, all_x, all_y, all_lbl

    # ---- build initial (frame 0) state -------------------------------
    i0 = _frame_state(0)
    px0, py0, nx0, ny0, wx0, wy0, nc0, ns0, allx0, ally0, lbl0 = i0

    def _empty_trace(color="rgba(0,0,0,0)"):
        return go.Scatter(x=[], y=[], mode="lines",
                          line=dict(color=color, width=0),
                          hoverinfo="none", showlegend=False)

    # ---- build Plotly figure with 7 traces ---------------------------
    fig = go.Figure(data=[
        # 0: background edges (static)
        go.Scatter(
            x=bg_x, y=bg_y, mode="lines",
            line=dict(color=GRAY, width=0.5),
            hoverinfo="none", showlegend=False,
        ),
        # 1: active positive edges
        _empty_trace(ORANGE.format(a=0.0)),
        # 2: active negative edges
        _empty_trace(STEEL.format(a=0.0)),
        # 3: winner path edges
        _empty_trace(GREEN.format(a=0.0)),
        # 4: all nodes
        go.Scatter(
            x=allx0, y=ally0,
            mode="markers+text",
            marker=dict(color=nc0, size=ns0,
                        line=dict(color="rgba(200,220,255,0.3)", width=0.5)),
            text=lbl0,
            textposition="middle right",
            textfont=dict(size=9, color="rgba(200,220,255,0.85)"),
            hoverinfo="text",
            hovertext=lbl0,
            showlegend=False,
        ),
        # 5: output label text only (separate trace for visibility toggle)
        go.Scatter(
            x=ot_x, y=ot_y, mode="text",
            text=ot_labels_empty,
            textposition="middle right",
            textfont=dict(size=11, color="rgba(240,240,100,0.95)"),
            showlegend=False, hoverinfo="none",
        ),
        # 6: input token label (static)
        go.Scatter(
            x=[in_x[act_in]], y=[in_y[act_in]],
            mode="text",
            text=[f"▶ {tokenizer.idx2word.get(input_idx, '?')}"],
            textposition="middle left",
            textfont=dict(size=12, color=GREEN_IN),
            showlegend=False, hoverinfo="none",
        ),
    ])

    # ---- animation frames --------------------------------------------
    frame_names = [
        "0 — Network",
        "1 — Input →",
        "2 — → Embedding",
        "3 — → Hidden",
        "4 — → Output",
        "5 — 🏆 Winner",
    ]

    frames = []
    for f in range(6):
        px, py, nx, ny, wx, wy, nc, ns, ax, ay, lbl = _frame_state(f)
        _, _, _, _, _, _, _, _, _, _, out_lbl = _frame_state(f)
        # Extract only the output portion of lbl for trace 5
        out_lbl_only = lbl[max_in + E + H:]

        frame = go.Frame(
            name=frame_names[f],
            traces=[1, 2, 3, 4, 5],
            data=[
                # trace 1: positive edges
                go.Scatter(x=px, y=py, mode="lines",
                           line=dict(color=ORANGE.format(a=0.75), width=1.8),
                           hoverinfo="none", showlegend=False),
                # trace 2: negative edges
                go.Scatter(x=nx, y=ny, mode="lines",
                           line=dict(color=STEEL.format(a=0.65), width=1.8),
                           hoverinfo="none", showlegend=False),
                # trace 3: winner edges
                go.Scatter(x=wx, y=wy, mode="lines",
                           line=dict(color=GREEN.format(a=0.90), width=2.5),
                           hoverinfo="none", showlegend=False),
                # trace 4: nodes
                go.Scatter(
                    x=ax, y=ay,
                    mode="markers+text",
                    marker=dict(color=nc, size=ns,
                                line=dict(color="rgba(200,220,255,0.3)", width=0.5)),
                    text=lbl,
                    textposition="middle right",
                    textfont=dict(size=9, color="rgba(200,220,255,0.85)"),
                    hoverinfo="text", hovertext=lbl, showlegend=False,
                ),
                # trace 5: output labels
                go.Scatter(
                    x=ot_x, y=ot_y, mode="text",
                    text=out_lbl_only,
                    textposition="middle right",
                    textfont=dict(size=11, color="rgba(240,240,100,0.95)"),
                    showlegend=False, hoverinfo="none",
                ),
            ]
        )
        frames.append(frame)

    fig.frames = frames

    # ---- layer annotations (static) ----------------------------------
    for lx, lbl in [(0.0, "Input"), (1.0, "Embedding"), (2.0, "Hidden"), (3.0, "Output")]:
        fig.add_annotation(
            x=lx, y=1.03, xref="x", yref="paper",
            text=f"<b>{lbl}</b>",
            showarrow=False,
            font=dict(size=11, color="rgba(167,139,250,0.9)"),
        )

    # Probability legend annotation
    fig.add_annotation(
        x=3.35, y=0.5, xref="x", yref="paper",
        text=(
            "<b>% = probability</b><br>"
            "<span style='color:rgba(255,140,30,0.9)'>─ positive weight</span><br>"
            "<span style='color:rgba(60,130,230,0.9)'>─ negative weight</span><br>"
            "<span style='color:rgba(0,230,120,0.9)'>─ winner path</span>"
        ),
        showarrow=False, align="left",
        font=dict(size=9, color="rgba(200,220,255,0.75)"),
    )

    # ---- animation controls ------------------------------------------
    slider_steps = [
        dict(
            args=[[fn], dict(frame=dict(duration=0, redraw=True), mode="immediate",
                             transition=dict(duration=0))],
            label=fn.split("—")[-1].strip(),
            method="animate",
        )
        for fn in frame_names
    ]

    fig.update_layout(
        title=dict(
            text=(
                f"<b>{lang_labels.get('inference_anim_title', 'Inference Animation')}</b>"
                f" — input: <b>{tokenizer.idx2word.get(input_idx, '?')}</b>"
                f" → top: <b>{top_words[0] if top_words else '?'}</b>"
                f" ({top_probs[0]*100:.1f}%)"
            ),
            font=dict(color="rgba(200,220,255,0.95)", size=13),
            x=0.5,
        ),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color="rgba(200,220,255,0.85)", size=10),
        xaxis=dict(
            range=[-0.35, 3.85],
            showgrid=False, zeroline=False, showticklabels=False,
        ),
        yaxis=dict(
            range=[-0.05, 1.1],
            showgrid=False, zeroline=False, showticklabels=False,
            scaleanchor="x", scaleratio=3.5,
        ),
        height=520,
        margin=dict(l=20, r=160, t=60, b=80),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=0,
            x=0.5,
            xanchor="center",
            yanchor="top",
            buttons=[
                dict(
                    label="▶ Play",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=1300, redraw=True),
                        fromcurrent=True,
                        transition=dict(duration=400, easing="cubic-in-out"),
                    )],
                ),
                dict(
                    label="⏸ Pause",
                    method="animate",
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate",
                        transition=dict(duration=0),
                    )],
                ),
            ],
            font=dict(color="white"),
            bgcolor="rgba(100,80,200,0.7)",
            bordercolor="rgba(167,139,250,0.5)",
        )],
        sliders=[dict(
            active=0,
            steps=slider_steps,
            x=0.0, len=1.0,
            y=-0.04,
            currentvalue=dict(
                prefix="Step: ",
                font=dict(color="rgba(200,220,255,0.85)", size=10),
                visible=True,
                xanchor="center",
            ),
            tickcolor="rgba(200,220,255,0.4)",
            font=dict(color="rgba(200,220,255,0.6)", size=9),
            bgcolor="rgba(30,30,60,0.6)",
            bordercolor="rgba(100,80,200,0.4)",
            pad=dict(b=10, t=5),
        )],
    )

    return fig
