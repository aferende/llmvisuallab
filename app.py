"""
LLM Visual Lab — Main Streamlit Application.

Run with:
    streamlit run app.py
"""
import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from engine.tokenizer import Tokenizer
from engine.model import TinyLM
from engine.training import train
from engine.inference import (
    predict_next_tokens,
    get_token_embeddings,
    get_sentence_embedding,
    cosine_similarity,
    pairwise_cosine,
    semantic_search,
    pca_reduce,
)
from engine.visualization import (
    plot_3d_network,
    plot_loss_curve,
    plot_embeddings_2d,
    plot_embeddings_3d,
    plot_cosine_similarity_3d,
    plot_similarity_heatmap,
)
from lang import TRANSLATIONS, t as _t

# ======================================================================
# Page configuration
# ======================================================================

st.set_page_config(
    page_title="LLM Visual Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================================
# CSS
# ======================================================================

def load_css() -> None:
    try:
        with open("assets/styles.css", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass  # CSS is cosmetic; app still works without it


load_css()


# ======================================================================
# Session state initialisation
# ======================================================================

def init_state() -> None:
    defaults = {
        "lang": "it",
        "tokenizer": None,
        "model": None,
        "trained": False,
        "sentences": [],
        "pairs": [],
        "loss_history": [],
        "log_lines": [],
        "dataset_ready": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()
LANG = st.session_state.lang


def T(key: str, **kwargs) -> str:
    """Shorthand translator using current language."""
    return _t(key, LANG, **kwargs)


# ======================================================================
# Helper: styled HTML blocks
# ======================================================================

def tip(text: str) -> None:
    st.markdown(f'<div class="tip-box">{text}</div>', unsafe_allow_html=True)


def section_header(text: str) -> None:
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def render_terminal(lines: list) -> None:
    """Render log lines in a Linux-style terminal widget."""
    content = "\n".join(lines[-60:])  # last 60 lines
    html = (
        '<div class="terminal-header">'
        '<span class="terminal-dot" style="background:#ff5f56"></span>'
        '<span class="terminal-dot" style="background:#ffbd2e"></span>'
        '<span class="terminal-dot" style="background:#27c93f"></span>'
        '&nbsp; llm-visual-lab — training console'
        '</div>'
        f'<div class="terminal-container">{content}</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def prediction_bars(predictions: list) -> None:
    """Render next-token predictions as horizontal progress bars."""
    html = ""
    for word, prob in predictions:
        bar_pct = int(prob * 100)
        bar_width = max(1, int(prob * 300))
        html += (
            f'<div class="pred-bar-container">'
            f'<span class="pred-word">{word}</span>'
            f'<div class="pred-bar-bg">'
            f'<div class="pred-bar-fill" style="width:{bar_width}px"></div>'
            f'</div>'
            f'<span class="pred-prob">{bar_pct}%</span>'
            f'</div>'
        )
    st.markdown(html, unsafe_allow_html=True)


def search_results_html(results: list) -> None:
    """Render semantic search results as ranked cards."""
    html = ""
    for rank, (sent, sim) in enumerate(results, 1):
        bar_w = int(max(0, sim) * 100)
        html += (
            f'<div class="search-result">'
            f'<div class="rank-badge">{rank}</div>'
            f'<div class="sentence-text">'
            f'{sent}'
            f'<div style="margin-top:5px;height:4px;background:rgba(40,50,100,0.5);border-radius:2px;">'
            f'<div style="width:{bar_w}%;height:100%;background:linear-gradient(90deg,#3a86ff,#8338ec);border-radius:2px;"></div>'
            f'</div>'
            f'</div>'
            f'<span class="sim-score">{sim:.4f}</span>'
            f'</div>'
        )
    st.markdown(html, unsafe_allow_html=True)


# ======================================================================
# SIDEBAR
# ======================================================================

with st.sidebar:
    # Large icon + brand name
    st.markdown(
        '<div style="text-align:center;padding:18px 0 8px 0;">'
        '<span style="font-size:4rem;line-height:1.1;">🧠</span><br>'
        '<span style="font-size:1.25rem;font-weight:800;color:#a78bfa;'
        'letter-spacing:0.03em;">LLM Visual Lab</span><br>'
        '<span style="font-size:0.72rem;color:rgba(150,160,200,0.7);">'
        'Educational AI Laboratory</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # Language selector
    lang_options = {"🇮🇹 Italiano": "it", "🇬🇧 English": "en"}
    selected_lang_name = st.selectbox(
        T("sidebar_lang"),
        list(lang_options.keys()),
        index=0 if st.session_state.lang == "it" else 1,
    )
    new_lang = lang_options[selected_lang_name]
    if new_lang != st.session_state.lang:
        st.session_state.lang = new_lang
        st.rerun()

    LANG = st.session_state.lang

    def T(key: str, **kwargs) -> str:
        return _t(key, LANG, **kwargs)

    st.divider()

    # Training hyperparameters
    st.markdown(f"### ⚙️ {'Iperparametri' if LANG == 'it' else 'Hyperparameters'}")
    hidden_size = st.slider(T("sidebar_brain_size"), min_value=4, max_value=64,
                            value=16, step=4)
    n_steps = st.slider(T("sidebar_steps"), min_value=10, max_value=300,
                        value=80, step=10)

    st.divider()

    st.markdown(T("sidebar_about"))
    st.caption(T("sidebar_about_text"))


# ======================================================================
# HERO HEADER
# ======================================================================

st.markdown(
    f'<div class="hero-header">'
    f'<h1>{T("app_title")}</h1>'
    f'<p>{T("app_subtitle")}</p>'
    f'</div>',
    unsafe_allow_html=True,
)

# Intro concept cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f'<div class="concept-card"><h4>🤖 {T("intro_llm_title")}</h4>'
        f'{T("intro_llm_text")}</div>',
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f'<div class="concept-card"><h4>🏋️ {T("intro_training_title")}</h4>'
        f'{T("intro_training_text")}</div>',
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f'<div class="concept-card"><h4>🔮 {T("intro_inference_title")}</h4>'
        f'{T("intro_inference_text")}</div>',
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f'<div class="concept-card"><h4>📐 {T("intro_embedding_title")}</h4>'
        f'{T("intro_embedding_text")}</div>',
        unsafe_allow_html=True,
    )

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)


# ======================================================================
# SECTION 1 — DATASET & TOKENISATION
# ======================================================================

section_header(T("sec1_title"))
tip(T("sec1_intro"))

raw_input = st.text_area(
    T("sec1_input_label"),
    value=T("sec1_default_sentences"),
    height=160,
    key="raw_sentences",
)

btn_tokenize = st.button(T("sec1_btn_tokenize"), key="btn_tok")

if btn_tokenize or st.session_state.dataset_ready:
    sentences = [s.strip() for s in raw_input.strip().splitlines() if s.strip()]

    if len(sentences) < 2:
        st.warning("Inserisci almeno 2 frasi. / Please enter at least 2 sentences.")
    else:
        tokenizer = Tokenizer()
        tokenizer.fit(sentences)
        pairs = tokenizer.get_training_pairs(sentences)

        st.session_state.tokenizer = tokenizer
        st.session_state.sentences = sentences
        st.session_state.pairs = pairs
        st.session_state.dataset_ready = True
        # Reset model if sentences changed
        if btn_tokenize:
            st.session_state.trained = False
            st.session_state.model = None
            st.session_state.loss_history = []
            st.session_state.log_lines = []

        # ---- Vocabulary ----
        st.markdown(f"#### {T('sec1_vocab_title')}")
        tip(T("sec1_vocab_info"))

        # Show vocabulary table
        special = {tokenizer.PAD, tokenizer.UNK}
        vocab_words = sorted(
            [(w, i) for w, i in tokenizer.vocab.items() if w not in special],
            key=lambda x: x[1],
        )

        # Render as token badges
        badges = "".join(
            f'<span class="token-badge">{w} <span class="idx">#{i}</span></span>'
            for w, i in vocab_words
        )
        st.markdown(
            f'<div style="line-height:2.2;">{badges}</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"📊 Vocab size: **{tokenizer.vocab_size}** tokens "
                   f"(including {tokenizer.PAD} and {tokenizer.UNK})")

        # ---- Training pairs ----
        st.markdown(f"#### {T('sec1_pairs_title')}")
        tip(T("sec1_pairs_info"))
        pair_cols = st.columns(min(4, len(pairs)))
        for i, (inp, tgt) in enumerate(pairs[:16]):
            col = pair_cols[i % len(pair_cols)]
            col.markdown(
                f"`{tokenizer.idx2word[inp]}` → `{tokenizer.idx2word[tgt]}`"
            )
        if len(pairs) > 16:
            st.caption(f"... e altri {len(pairs) - 16} coppie")

        # ---- Per-sentence tokenisation ----
        with st.expander(T("sec1_tokenization_title"), expanded=False):
            for sent in sentences:
                tok_pairs = tokenizer.tokenize_with_words(sent)
                badges_s = "".join(
                    f'<span class="token-badge">{w} <span class="idx">#{i}</span></span>'
                    for w, i in tok_pairs
                )
                st.markdown(
                    f"**{sent}** → {badges_s}",
                    unsafe_allow_html=True,
                )

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)


# ======================================================================
# SECTION 2 — INTERACTIVE TRAINING
# ======================================================================

section_header(T("sec2_title"))
tip(T("sec2_intro"))

if not st.session_state.dataset_ready:
    st.info("⚠️ Completa prima la Sezione 1. / Complete Section 1 first.")
else:
    tokenizer = st.session_state.tokenizer
    pairs = st.session_state.pairs

    # Architecture info box
    with st.expander(T("sec2_arch_title"), expanded=False):
        st.markdown(
            f'<div class="arch-box">'
            f'Input  (one-hot)  → [{tokenizer.vocab_size}]\n'
            f'Embedding         → [8]\n'
            f'Hidden (tanh)     → [{hidden_size}]\n'
            f'Output (softmax)  → [{tokenizer.vocab_size}]\n'
            f'\n'
            f'Objective : next-token prediction\n'
            f'Loss      : cross-entropy\n'
            f'Optimizer : SGD  lr=0.05\n'
            f'Steps     : {n_steps}\n'
            f'Pairs     : {len(pairs)}'
            f'</div>',
            unsafe_allow_html=True,
        )

    col_btn1, col_btn2 = st.columns([1, 5])
    btn_train = col_btn1.button(T("sec2_btn_train"), key="btn_train")
    btn_reset = col_btn2.button(T("sec2_btn_reset"), key="btn_reset")

    if btn_reset:
        st.session_state.trained = False
        st.session_state.model = None
        st.session_state.loss_history = []
        st.session_state.log_lines = []
        st.rerun()

    # ---- TRAINING LOOP ----
    if btn_train:
        st.session_state.trained = False
        st.session_state.loss_history = []
        st.session_state.log_lines = []

        model = TinyLM(
            vocab_size=tokenizer.vocab_size,
            embed_dim=8,
            hidden_size=hidden_size,
        )

        lang_labels = TRANSLATIONS[LANG]
        logs = []

        def add_log(msg: str) -> None:
            logs.append(msg)
            st.session_state.log_lines = list(logs)

        add_log(T("log_init"))
        add_log(T("log_tokenize"))
        add_log(T("log_vocab", n=tokenizer.vocab_size))
        add_log(T("log_pairs", n=len(pairs)))
        add_log(T("log_start", steps=n_steps))

        # Explanation boxes shown before training starts
        st.markdown(f"#### {T('sec2_net_title')}")
        with st.expander(T("net_legend_title"), expanded=True):
            tip(T("net_legend_text"))

        progress_bar = st.progress(0)
        net_placeholder = st.empty()

        st.markdown(f"#### {T('sec2_loss_title')}")
        tip(T("loss_deep_explain"))
        loss_placeholder = st.empty()

        st.markdown(f"#### {T('sec2_log_title')}")
        log_placeholder = st.empty()

        losses = []
        UPDATE_VIZ_EVERY = max(1, n_steps // 40)  # update viz up to 40 times

        for state in train(model, pairs, n_steps, lr=0.05):
            step = state["step"]
            loss = state["loss"]
            losses.append(loss)

            # Log
            add_log(T("log_step", step=step + 1, total=n_steps, loss=loss))
            if step % 5 == 0:
                g = state["gradients"]
                add_log(T("log_weights", wh=g["W_h_norm"], wo=g["W_out_norm"]))

            # Update visuals every N steps (performance)
            if step % UPDATE_VIZ_EVERY == 0 or step == n_steps - 1:
                fig_net = plot_3d_network(
                    model=model,
                    activations=state["activations"],
                    tokenizer=tokenizer,
                    input_idx=state["input_idx"],
                    target_idx=state["target_idx"],
                    step=step,
                    loss=loss,
                    lang_labels=lang_labels,
                )
                net_placeholder.plotly_chart(
                    fig_net, width='stretch', key=f"net_{step}"
                )

                fig_loss = plot_loss_curve(losses, lang_labels)
                loss_placeholder.plotly_chart(
                    fig_loss, width='stretch', key=f"loss_{step}"
                )

            log_placeholder.empty()
            with log_placeholder.container():
                render_terminal(logs)

            progress_bar.progress((step + 1) / n_steps)

        add_log(T("log_done", loss=losses[-1]))
        log_placeholder.empty()
        with log_placeholder.container():
            render_terminal(logs)

        st.session_state.model = model
        st.session_state.loss_history = losses
        st.session_state.trained = True
        st.success(T("sec2_done"))

        # Weights explanation after training
        tip(T("weights_explain"))

    # ---- Show results if already trained ----
    elif st.session_state.trained and st.session_state.model is not None:
        model = st.session_state.model
        losses = st.session_state.loss_history
        lang_labels = TRANSLATIONS[LANG]

        st.success(T("sec2_done"))

        # ---- Metrics ----
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric(
            "📉 " + T("loss_label"),
            f"{losses[-1]:.4f}",
            delta=f"{losses[-1] - losses[0]:.4f}",
            delta_color="inverse",
            help=T("loss_deep_explain"),
        )
        mc2.metric("🔢 Vocab", f"{tokenizer.vocab_size}")
        mc3.metric("🧠 " + T("sidebar_brain_size"), f"{hidden_size}")

        # ---- 3D Network ----
        st.markdown(f"#### {T('sec2_net_title')}")
        with st.expander(T("net_legend_title"), expanded=False):
            tip(T("net_legend_text"))
        tip(T("sec2_net_info"))

        if pairs:
            inp_idx, tgt_idx = pairs[-1]
            _, last_acts = model.forward(inp_idx)
            fig_net = plot_3d_network(
                model=model,
                activations=last_acts,
                tokenizer=tokenizer,
                input_idx=inp_idx,
                target_idx=tgt_idx,
                step=len(losses) - 1,
                loss=losses[-1],
                lang_labels=lang_labels,
            )
            st.plotly_chart(fig_net, width='stretch')

        # ---- Loss curve ----
        st.markdown(f"#### {T('sec2_loss_title')}")
        tip(T("loss_deep_explain"))
        if losses:
            fig_loss = plot_loss_curve(losses, lang_labels)
            st.plotly_chart(fig_loss, width='stretch')

        # ---- Weights explanation ----
        tip(T("weights_explain"))

        # ---- Console replay ----
        if st.session_state.log_lines:
            st.markdown(f"#### {T('sec2_log_title')}")
            render_terminal(st.session_state.log_lines)

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)


# ======================================================================
# SECTION 3 — INFERENCE & SEMANTIC SEARCH
# ======================================================================

section_header(T("sec3_title"))

if not st.session_state.trained or st.session_state.model is None:
    st.info(T("sec3_info_train_first"))
else:
    tip(T("sec3_intro"))

    model = st.session_state.model
    tokenizer = st.session_state.tokenizer
    sentences = st.session_state.sentences
    lang_labels = TRANSLATIONS[LANG]

    # ----------------------------------------------------------------
    # 3A: Next-token prediction
    # ----------------------------------------------------------------
    st.markdown(f"### 🎯 {T('sec3_next_token_title')}")
    tip(T("sec3_next_token_info"))

    inf_col1, inf_col2 = st.columns([2, 3])
    with inf_col1:
        query_text = st.text_input(
            T("sec3_query_label"),
            value="il gatto",
            placeholder=T("sec3_query_placeholder"),
            key="infer_query",
        )
        btn_infer = st.button(T("sec3_btn_infer"), key="btn_infer")

    if btn_infer and query_text.strip():
        predictions, acts = predict_next_tokens(
            model, tokenizer, query_text.strip(), top_k=6
        )
        with inf_col2:
            st.markdown(f"**Input:** `{query_text}`")
            prediction_bars(predictions)

        # Log
        if predictions:
            best_word, best_prob = predictions[0]
            tok_idx = tokenizer.tokenize(query_text.strip())[-1] if tokenizer.tokenize(query_text.strip()) else 0
            st.session_state.log_lines.append(
                T("log_infer", token=query_text.split()[-1], idx=tok_idx)
            )
            st.session_state.log_lines.append(
                T("log_predict", word=best_word, prob=best_prob)
            )

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ----------------------------------------------------------------
    # 3B: Embeddings visualisation
    # ----------------------------------------------------------------
    st.markdown(f"### 🌐 {T('sec3_emb_title')}")
    tip(T("sec3_emb_info"))
    tip(T("dataset_tip_semantic"))

    embeddings, labels = get_token_embeddings(model, tokenizer)

    # Optional query overlay
    query_emb_for_viz = None
    query_label_for_viz = None
    if query_text.strip():
        idx_list = tokenizer.tokenize(query_text.strip())
        if idx_list:
            query_emb_for_viz = get_sentence_embedding(model, tokenizer, query_text.strip())
            query_label_for_viz = query_text.strip()

    tab_2d, tab_3d = st.tabs(["2D (PCA)", "3D (PCA)"])
    with tab_2d:
        fig_2d = plot_embeddings_2d(
            embeddings, labels,
            query_label_for_viz, query_emb_for_viz,
            lang_labels,
        )
        st.plotly_chart(fig_2d, width='stretch')
    with tab_3d:
        fig_3d = plot_embeddings_3d(
            embeddings, labels,
            query_label_for_viz, query_emb_for_viz,
            lang_labels,
        )
        st.plotly_chart(fig_3d, width='stretch')

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ----------------------------------------------------------------
    # 3C: Cosine similarity between two words
    # ----------------------------------------------------------------
    st.markdown(f"### 📐 {T('sec3_cos_title')}")
    tip(T("sec3_cos_info"))

    vocab_words = sorted(
        [w for w in tokenizer.vocab if w not in {tokenizer.PAD, tokenizer.UNK}]
    )

    cos_col1, cos_col2, cos_col3 = st.columns([2, 2, 3])
    with cos_col1:
        word1 = st.selectbox(T("sec3_word1_label"), vocab_words,
                             index=0, key="cos_w1")
    with cos_col2:
        default_w2 = min(1, len(vocab_words) - 1)
        word2 = st.selectbox(T("sec3_word2_label"), vocab_words,
                             index=default_w2, key="cos_w2")

    if word1 and word2:
        v1 = model.get_embedding(tokenizer.vocab[word1])
        v2 = model.get_embedding(tokenizer.vocab[word2])
        sim = cosine_similarity(v1, v2)

        with cos_col3:
            st.metric(
                f"cos({word1}, {word2})",
                f"{sim:.4f}",
                help="Range: -1 (opposti) → 0 (ortogonali) → 1 (identici)",
            )

        # Sphere info with word names interpolated
        sphere_info = T("cosine_sphere_info").replace("{w1}", word1).replace("{w2}", word2)
        tip(sphere_info)

        # Build index within the filtered labels list
        w1_sphere_idx = labels.index(word1) if word1 in labels else 0
        w2_sphere_idx = labels.index(word2) if word2 in labels else min(1, len(labels) - 1)

        fig_cos = plot_cosine_similarity_3d(
            embeddings, labels,
            w1_sphere_idx, w2_sphere_idx,
            word1, word2, sim, lang_labels,
        )
        st.plotly_chart(fig_cos, width='stretch')

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ----------------------------------------------------------------
    # 3D: Semantic search
    # ----------------------------------------------------------------
    st.markdown(f"### 🔍 {T('sec3_search_title')}")
    tip(T("sec3_search_info"))

    srch_col1, srch_col2 = st.columns([3, 1])
    with srch_col1:
        search_query = st.text_input(
            "Query:", value=query_text,
            key="search_query_input",
        )
    with srch_col2:
        btn_search = st.button(T("sec3_search_btn"), key="btn_search")

    if btn_search and search_query.strip():
        results = semantic_search(search_query.strip(), sentences, model, tokenizer)
        search_results_html(results)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ----------------------------------------------------------------
    # 3E: Pairwise similarity heatmap
    # ----------------------------------------------------------------
    st.markdown(f"### 🗺️ {T('sec3_heatmap_title')}")
    tip(T("sec3_heatmap_info"))

    # Use sentence-level embeddings
    sent_embs = np.array(
        [get_sentence_embedding(model, tokenizer, s) for s in sentences]
    )
    sent_labels = [s[:30] + "…" if len(s) > 30 else s for s in sentences]
    sim_mat = pairwise_cosine(sent_embs)

    fig_heat = plot_similarity_heatmap(sim_mat, sent_labels, lang_labels)
    st.plotly_chart(fig_heat, width='stretch')


# ======================================================================
# Footer
# ======================================================================

st.markdown(
    '<div style="text-align:center;margin-top:40px;color:rgba(120,140,180,0.5);'
    'font-size:0.8rem;padding-bottom:20px;">'
    "LLM Visual Lab — Progetto educativo open-source | "
    "Powered by NumPy + Streamlit + Plotly"
    "</div>",
    unsafe_allow_html=True,
)
