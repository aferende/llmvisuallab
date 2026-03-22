# 🧠 LLM Visual Lab

> An open-source educational laboratory that opens the **"black box"** of Artificial Intelligence.

**🚀 Try it live → [llmvisuallab.streamlit.app](https://llmvisuallab.streamlit.app)**

---

## Project Goal

LLM Visual Lab makes the inner workings of Large Language Models (LLMs) **visible and understandable** — even without any technical background.

The app guides you through five interconnected concepts:

1. **Tokenisation** — how raw text is split into numbered units that the model can process
2. **Training** — how the neural network adjusts its internal numbers (weights) by trying to predict the next word and correcting its mistakes
3. **3D Neural Network** — a live 3D view of the network computing forward and backward passes on your own sentences, with colours and sizes derived from real activation values
4. **Embeddings** — how each word becomes a vector of numbers in a high-dimensional space, with semantically similar words clustering together
5. **Cosine Similarity** — how the angle between two vectors measures semantic similarity, visualised on an interactive 3D sphere

Everything runs on a tiny NumPy model trained on **your own custom sentences** — no external API, no GPU, no black box.

---

## Target Audience

- **Non-technical users** who want to understand what AI is really doing inside
- **Students and educators** approaching machine learning for the first time
- **Beginners** curious about how transformers, embeddings and language models work
- **AI developers** who want to build a deeper intuition for the mechanics behind LLMs — tokenisation, backpropagation, embedding spaces and cosine similarity — without the abstraction layers of a full framework

---

## Application Sections

### 📝 Section 1 — Dataset & Tokenisation
Enter your own sentences (Italian, English or mixed, up to 100). The app shows you:
- How each word becomes a **numbered token**
- The complete **vocabulary** built from your sentences
- All **training pairs** (input token → next token) the model will learn from
- A clear note explaining that real models use *subword* tokens (word fragments), while this lab uses whole words for educational clarity

### ⚡ Section 2 — Interactive Training
Click **Start Training** to watch the model learn in real time:
- **3D Neural Network** — 4 visible layers (Input → Embedding → Hidden → Output). Node colours reflect actual activation values (🔵 blue = inhibited, ⚪ grey = neutral, 🔴 red = excited). The active input token is highlighted in green, the target token in gold.
- **Live Loss Curve** — raw instantaneous loss (light background) plus a smoothed rolling-average trend line that shows the model genuinely improving over time
- **Linux-style terminal console** — step-by-step logs with loss values and gradient norms at each training step
- **Adjustable hyperparameters** — brain size (hidden neurons 4–64) and number of training steps (10–300)

### 🔮 Section 3 — Inference & Semantic Search
After training, explore what the model has learned:
- **Next-token prediction** — type any word or phrase; the model shows the top most likely continuations with probability bars
- **Embedding map 2D** — all word vectors projected to 2 dimensions via PCA; semantically related words cluster visibly together
- **Embedding map 3D** — the same vectors drawn as arrows from the origin (0,0,0) with visible Cartesian axes PC1/PC2/PC3 — exactly like a vector space diagram
- **Cosine Similarity sphere** — a 3D unit sphere with **all** word vectors as arrows; two selected words are highlighted in 🟢 green and 🔴 red; a golden arc shows the angle θ between them with the cosine value
- **Semantic search** — rank all training sentences by similarity to your query embedding
- **Pairwise heatmap** — full cosine similarity matrix between all sentences

---

## Features at a Glance

| Feature | Description |
|---|---|
| 📝 Custom dataset | Train on your own sentences (up to 100) |
| 🔬 Tokenisation view | Word-level tokens with vocabulary and index badges |
| 🕸️ Real-time 3D network | Live network coloured by actual activation values |
| ⚡ Interactive training | Adjustable brain size and steps; shuffled-epoch SGD |
| 📉 Loss curve | Raw + smoothed trend — genuine learning visible |
| 🖥️ Training console | Linux terminal with step-by-step gradient logs |
| 🌐 Embeddings 2D/3D | PCA projection with Cartesian axes and vector arrows |
| 🌍 Cosine sphere | 3D unit sphere with all word vectors and angle arc |
| 🔍 Semantic search | Ranked results by embedding similarity |
| 🌍 Multi-language UI | Full Italian / English (Italian default) |

---

## Local Installation Guide (WSL / Ubuntu)

### Prerequisites
- Python 3.9 or higher
- WSL (Windows Subsystem for Linux) or native Linux / macOS

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/aferende/llmvisuallab.git
cd llmvisuallab

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### Quick syntax check

```bash
python3 -m py_compile app.py engine/tokenizer.py engine/model.py \
        engine/training.py engine/inference.py engine/visualization.py
echo "All files OK"
```

---

## Project Structure

```
llmvisuallab/
├── app.py                  # Main Streamlit application
├── lang.py                 # Translations — Italian (default) / English
├── requirements.txt        # streamlit · numpy · plotly
├── README.md
│
├── engine/
│   ├── __init__.py
│   ├── tokenizer.py        # Word-level tokeniser + vocabulary builder
│   ├── model.py            # TinyLM — NumPy neural language model
│   ├── training.py         # Manual backprop + shuffled-epoch SGD
│   ├── inference.py        # Next-token prediction, PCA, cosine similarity
│   └── visualization.py   # All Plotly charts
│
├── assets/
│   ├── styles.css          # Dark UI theme
│   └── images/             # Local image assets
│
└── .streamlit/
    └── config.toml         # Streamlit Cloud configuration
```

---

## Model Architecture

```
Input (one-hot)  →  [vocab_size]
Embedding        →  [8]            W_emb  (vocab_size × 8)
Hidden (tanh)    →  [hidden_size]  W_h    (8 × hidden_size)
Output (softmax) →  [vocab_size]   W_out  (hidden_size × vocab_size)

Objective  : next-token prediction (bigram)
Loss       : cross-entropy
Optimiser  : SGD with lr = 0.1, pairs shuffled each epoch
Backprop   : hand-coded with NumPy (no autograd)
```

---

## Notes & Limitations

- **Educational project only** — not a production-grade language model.
- **Token granularity**: this lab uses **one word = one token** for clarity. Real models (GPT, Claude, etc.) use subword tokens (BPE / SentencePiece) so that rare words and multiple languages fit in a fixed vocabulary.
- All computations are CPU-only, pure NumPy — no GPU, no PyTorch, no TensorFlow.
- Use 6–15 training sentences for the best visual results. With very short datasets the model has little to learn.
- Cosine similarity clusters become clearly visible after sufficient training steps (≥ 100 recommended).

---

## Dependencies

```
streamlit  ≥ 1.28
numpy      ≥ 1.24
plotly     ≥ 5.17
```

---

*LLM Visual Lab — built for curiosity, not production.*
*Live demo: [llmvisuallab.streamlit.app](https://llmvisuallab.streamlit.app)*

---

*Built with [Claude Code](https://claude.ai/claude-code) — Anthropic's AI coding assistant.*
