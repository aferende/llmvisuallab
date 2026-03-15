from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import plotly.graph_objects as go


def build_network_figure(
    input_tokens: List[str], hidden_size: int, output_tokens: List[str], step: int, weight_strength: float
) -> go.Figure:
    max_tokens = 6
    input_nodes = input_tokens[:max_tokens] or ["<empty>"]
    output_nodes = output_tokens[:max_tokens] or ["<vocab>"]
    hidden_nodes = [f"H{i+1}" for i in range(min(hidden_size, 8))]

    x_coords, y_coords, labels, colors = [], [], [], []

    for idx, token in enumerate(input_nodes):
        x_coords.append(0)
        y_coords.append(idx)
        labels.append(token)
        colors.append("#00BFFF")

    for idx, node in enumerate(hidden_nodes):
        x_coords.append(1)
        y_coords.append(idx)
        labels.append(node)
        colors.append("#FFD166")

    for idx, token in enumerate(output_nodes):
        x_coords.append(2)
        y_coords.append(idx)
        labels.append(token)
        colors.append("#90EE90")

    fig = go.Figure()

    flow_shift = (step % 10) / 10
    thickness = 1.5 + min(weight_strength / 4.0, 6.0)

    for i in range(len(input_nodes)):
        for h in range(len(hidden_nodes)):
            fig.add_shape(
                type="line",
                x0=0.05 + flow_shift * 0.02,
                y0=i,
                x1=0.95,
                y1=h,
                line=dict(color=f"rgba(0,191,255,{0.15 + 0.05 * (step % 5)})", width=thickness),
            )

    for h in range(len(hidden_nodes)):
        for o in range(len(output_nodes)):
            fig.add_shape(
                type="line",
                x0=1.05,
                y0=h,
                x1=1.95 - flow_shift * 0.02,
                y1=o,
                line=dict(color=f"rgba(255,99,71,{0.12 + 0.06 * ((step + 2) % 5)})", width=thickness),
            )

    fig.add_trace(
        go.Scatter(
            x=x_coords,
            y=y_coords,
            mode="markers+text",
            text=labels,
            textposition="middle center",
            marker=dict(size=36, color=colors, line=dict(color="#222", width=1)),
            hoverinfo="skip",
        )
    )

    for i, token in enumerate(input_nodes):
        token_x = 0.25 + flow_shift * 1.5
        fig.add_annotation(
            x=token_x,
            y=i,
            text=f"{token}",
            showarrow=False,
            font=dict(size=10, color="#e8f0fe"),
            bgcolor="rgba(37,38,43,0.8)",
        )

    fig.update_layout(
        title="Training Flow: Input → Hidden → Output",
        plot_bgcolor="#111827",
        paper_bgcolor="#111827",
        font=dict(color="white"),
        xaxis=dict(visible=False, range=[-0.2, 2.2]),
        yaxis=dict(visible=False, range=[-0.8, max(len(input_nodes), len(hidden_nodes), len(output_nodes)) + 0.5]),
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def _project_2d(vectors: np.ndarray) -> np.ndarray:
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    principal = eigvecs[:, order[:2]]
    return centered @ principal


def build_embedding_projection(embeddings: np.ndarray, tokens: List[str], use_3d: bool = False) -> go.Figure:
    if embeddings.shape[0] < 2:
        embeddings = np.vstack([embeddings, embeddings + 1e-3])
        tokens = tokens + ["(pad)"]

    points_2d = _project_2d(embeddings)

    if use_3d:
        z = embeddings[:, 0]
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=points_2d[:, 0],
                    y=points_2d[:, 1],
                    z=z,
                    mode="markers+text",
                    text=tokens,
                    marker=dict(size=6, color=z, colorscale="Viridis"),
                )
            ]
        )
        fig.update_layout(height=480, scene=dict(bgcolor="#0f172a"), paper_bgcolor="#0f172a", font=dict(color="white"))
        return fig

    fig = go.Figure(
        data=[
            go.Scatter(
                x=points_2d[:, 0],
                y=points_2d[:, 1],
                mode="markers+text",
                text=tokens,
                textposition="top center",
                marker=dict(size=13, color=points_2d[:, 0], colorscale="Plasma", line=dict(color="#111", width=1)),
            )
        ]
    )
    fig.update_layout(
        title="Token Embedding Space",
        height=430,
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font=dict(color="white"),
        xaxis=dict(zeroline=False),
        yaxis=dict(zeroline=False),
    )
    return fig


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def build_similarity_figure(vectors: Dict[str, np.ndarray], query_token: str) -> go.Figure:
    fig = go.Figure()
    origin = np.array([0.0, 0.0])
    q = vectors[query_token][:2]

    for token, vec in vectors.items():
        point = vec[:2]
        color = "#f97316" if token == query_token else "#22c55e"
        fig.add_trace(
            go.Scatter(
                x=[origin[0], point[0]],
                y=[origin[1], point[1]],
                mode="lines+markers+text",
                text=["", token],
                textposition="top center",
                line=dict(width=3, color=color),
                marker=dict(size=[2, 10], color=color),
                showlegend=False,
            )
        )

        if token != query_token:
            sim = cosine_similarity(q, point)
            fig.add_annotation(x=point[0], y=point[1], text=f"cos={sim:.2f}", showarrow=False, yshift=16, font=dict(color="white", size=10))

    fig.update_layout(
        title="Cosine Similarity (Angle Between Vectors)",
        height=420,
        plot_bgcolor="#020617",
        paper_bgcolor="#020617",
        font=dict(color="white"),
        xaxis=dict(zeroline=True, gridcolor="#334155"),
        yaxis=dict(zeroline=True, gridcolor="#334155"),
    )
    return fig
