from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st

from engine import (
    MicroTransformer,
    SimpleTokenizer,
    build_embedding_projection,
    build_network_figure,
    build_similarity_figure,
    cosine_similarity,
)

st.set_page_config(page_title="LLM Visual Lab", page_icon="🧠", layout="wide")

with open("assets/styles.css", "r", encoding="utf-8") as css_file:
    st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)

TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "English": {
        "sidebar_lang": "Language",
        "title": "LLM Visual Lab",
        "intro": "A beginner-friendly playground to visualize how small language models train, reason, and predict.",
        "section_setup": "1) Setup & Dataset",
        "dataset_help": "Add one sentence per line. Mix languages if you want.",
        "dataset_label": "Dataset sentences",
        "dataset_default": "The cat eats fish\nThe dog runs fast\nAI learns patterns\nIl cane corre veloce",
        "tokens_title": "Visible Tokenization",
        "section_train": "2) Interactive Training (CPU-only)",
        "brain_size": "Brain Size (hidden neurons)",
        "train_steps": "Training Steps",
        "run_training": "Run Training Animation",
        "info_training": "💡 Tip: Bigger brain size can represent richer patterns but takes more training.",
        "console_title": "Linux-style Training Console",
        "section_infer": "3) Inference & Semantic Search",
        "prompt_label": "Type a prompt",
        "prompt_default": "The cat",
        "run_infer": "Run Next Token Prediction",
        "embed_mode": "Embedding view",
        "embed_2d": "2D",
        "embed_3d": "3D",
        "cos_title": "Cosine Similarity Math",
        "formula": "cos(θ) = (a · b) / (||a|| ||b||)",
        "prediction": "Predicted next token",
        "prob_table": "Top token probabilities",
        "info_infer": "💡 Tip: Higher cosine similarity means vectors point in a similar semantic direction.",
    },
    "Italiano": {
        "sidebar_lang": "Lingua",
        "title": "LLM Visual Lab",
        "intro": "Un laboratorio visivo per principianti per capire come i piccoli modelli linguistici si addestrano e predicono.",
        "section_setup": "1) Configurazione e Dataset",
        "dataset_help": "Inserisci una frase per riga. Puoi mescolare lingue diverse.",
        "dataset_label": "Frasi del dataset",
        "dataset_default": "The cat eats fish\nThe dog runs fast\nAI learns patterns\nIl cane corre veloce",
        "tokens_title": "Tokenizzazione Visibile",
        "section_train": "2) Training Interattivo (solo CPU)",
        "brain_size": "Dimensione cervello (neuroni nascosti)",
        "train_steps": "Passi di training",
        "run_training": "Avvia Animazione Training",
        "info_training": "💡 Suggerimento: una rete più grande cattura pattern più ricchi ma richiede più training.",
        "console_title": "Console Linux-style del Training",
        "section_infer": "3) Inferenza e Ricerca Semantica",
        "prompt_label": "Scrivi un prompt",
        "prompt_default": "The cat",
        "run_infer": "Esegui Predizione Prossimo Token",
        "embed_mode": "Vista embedding",
        "embed_2d": "2D",
        "embed_3d": "3D",
        "cos_title": "Matematica della Similarità Coseno",
        "formula": "cos(θ) = (a · b) / (||a|| ||b||)",
        "prediction": "Prossimo token predetto",
        "prob_table": "Probabilità token più alte",
        "info_infer": "💡 Suggerimento: una similarità coseno alta indica vettori semanticamente vicini.",
    },
}

language = st.sidebar.selectbox("Language / Lingua", ["English", "Italiano"])
T = TRANSLATIONS[language]

st.title(T["title"])
st.markdown(f"<div class='intro-box'>{T['intro']}</div>", unsafe_allow_html=True)

st.header(T["section_setup"])
dataset_text = st.text_area(T["dataset_label"], value=T["dataset_default"], height=140, help=T["dataset_help"])

dataset_lines = [line.strip() for line in dataset_text.splitlines() if line.strip()]

if not dataset_lines:
    st.warning("Please provide at least one training sentence.")
    st.stop()

tokenizer = SimpleTokenizer()
tokenizer.fit(dataset_lines)

st.subheader(T["tokens_title"])
token_columns = st.columns(min(4, len(dataset_lines)))
for idx, sentence in enumerate(dataset_lines[:4]):
    with token_columns[idx % len(token_columns)]:
        st.markdown(f"**{sentence}**")
        st.code(tokenizer.tokenize(sentence))

pairs: List[Tuple[List[int], int]] = []
for sentence in dataset_lines:
    ids = tokenizer.encode(sentence)
    if len(ids) > 1:
        for i in range(1, len(ids)):
            pairs.append((ids[:i], ids[i]))

st.header(T["section_train"])
st.info(T["info_training"])
col_a, col_b = st.columns(2)
with col_a:
    hidden_size = st.slider(T["brain_size"], min_value=4, max_value=32, value=12, step=2)
with col_b:
    train_steps = st.slider(T["train_steps"], min_value=5, max_value=80, value=24, step=1)

model = MicroTransformer(vocab_size=tokenizer.vocab_size, hidden_size=hidden_size)

network_placeholder = st.empty()
console_placeholder = st.empty()

if st.button(T["run_training"], type="primary"):
    snapshots = model.train(pairs, steps=train_steps, lr=0.06)
    logs = ["[INFO] Tokenizing dataset...", f"[INFO] Vocabulary size: {tokenizer.vocab_size}"]

    token_names = [tokenizer.id_to_token[i] for i in range(min(tokenizer.vocab_size, 6))]
    input_tokens = tokenizer.tokenize(dataset_lines[0])

    for snap in snapshots:
        fig = build_network_figure(
            input_tokens=input_tokens,
            hidden_size=hidden_size,
            output_tokens=token_names,
            step=snap.step,
            weight_strength=snap.weight_strength,
        )
        network_placeholder.plotly_chart(fig, use_container_width=True)

        logs.append(f"[TRAIN] Epoch {snap.step:02d}: Loss {snap.loss:.4f} | WeightNorm {snap.weight_strength:.3f}")
        console_html = "<div class='terminal-box'><div class='terminal-header'>" + T["console_title"] + "</div><pre>"
        console_html += "\n".join(logs[-12:]) + "</pre></div>"
        console_placeholder.markdown(console_html, unsafe_allow_html=True)
        time.sleep(0.12)

    st.success("Training simulation completed.")

st.header(T["section_infer"])
st.info(T["info_infer"])

prompt = st.text_input(T["prompt_label"], value=T["prompt_default"])
embed_mode = st.radio(T["embed_mode"], [T["embed_2d"], T["embed_3d"]], horizontal=True)

if st.button(T["run_infer"]):
    # quick training warmup for inference stability
    model.train(pairs, steps=max(12, train_steps // 2), lr=0.05)

    encoded_prompt = tokenizer.encode(prompt)
    pred_id, probs = model.predict_next_token(encoded_prompt)
    pred_token = tokenizer.id_to_token.get(pred_id, "?")

    st.markdown(f"### {T['prediction']}: `{pred_token}`")

    top_idx = np.argsort(probs)[::-1][: min(8, tokenizer.vocab_size)]
    st.subheader(T["prob_table"])
    st.dataframe(
        {
            "token": [tokenizer.id_to_token[i] for i in top_idx],
            "probability": [float(probs[i]) for i in top_idx],
        },
        use_container_width=True,
        hide_index=True,
    )

    vocab_tokens = [tokenizer.id_to_token[i] for i in range(tokenizer.vocab_size)]
    fig_embed = build_embedding_projection(model.embeddings, vocab_tokens, use_3d=(embed_mode == T["embed_3d"]))
    st.plotly_chart(fig_embed, use_container_width=True)

    st.subheader(T["cos_title"])
    st.caption(T["formula"])

    query_token = pred_token if pred_token in vocab_tokens else vocab_tokens[0]
    vec_map = {tok: model.embeddings[tokenizer.token_to_id[tok]] for tok in vocab_tokens[: min(7, len(vocab_tokens))]}
    if query_token not in vec_map:
        query_token = next(iter(vec_map))

    sim_fig = build_similarity_figure(vec_map, query_token=query_token)
    st.plotly_chart(sim_fig, use_container_width=True)

    q_vec = vec_map[query_token]
    sims = []
    for tok, vec in vec_map.items():
        if tok == query_token:
            continue
        sims.append((tok, cosine_similarity(q_vec, vec)))
    sims.sort(key=lambda x: x[1], reverse=True)

    if sims:
        st.write("Top semantic neighbors:")
        for tok, score in sims[:3]:
            st.write(f"- **{tok}**: {score:.3f}")
