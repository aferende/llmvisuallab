# 🧠 LLM Visual Lab

> An open-source educational laboratory that opens the "black box" of Artificial Intelligence.

---

## Project Goal

LLM Visual Lab makes the inner workings of Large Language Models (LLMs) visible and understandable.
The app walks you through tokenisation, training, neural network activation, inference, embeddings and
cosine similarity — all using a tiny model trained live on **your own sentences**, with no external
APIs or cloud dependencies.

---

## Target Audience

- **Non-technical users** who want to understand what AI is really doing
- **Students** approaching machine learning for the first time
- **Beginners** curious about transformers, embeddings and language models

---

## Features

| Feature | Description |
|---|---|
| 📝 Custom dataset input | Train on your own sentences (Italian, English, or mixed) |
| 🔬 Tokenisation visualisation | See every word become a numbered token |
| 🕸️ Real-time 3D neural network | Live 3D view of the network trained on your data |
| ⚡ Interactive training | Adjustable brain size and training steps; loss curve updated live |
| 🖥️ Training console | Linux-style terminal showing every training step |
| 🌐 Embeddings map | 2D and 3D PCA projection of learned word vectors |
| 📐 Cosine similarity | Vector diagram showing how similar any two words are |
| 🔍 Semantic search | Find the most semantically similar sentence to your query |
| 🗺️ Similarity heatmap | Pairwise cosine similarity between all training sentences |
| 🌍 Multi-language UI | Full Italian / English interface (Italian default) |

---

## Local Installation Guide (WSL / Ubuntu)

### Prerequisites

- Python 3.9 or higher already installed
- WSL (Windows Subsystem for Linux) or native Linux / macOS

### Steps

```bash
# 1. Clone or copy the project folder
cd LLMVisualLab

# 2. Create a virtual environment
python3 -m venv .venv

# 3. Activate it
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the app
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### Quick syntax check (optional)

```bash
python3 -m py_compile app.py
python3 -m py_compile engine/tokenizer.py
python3 -m py_compile engine/model.py
python3 -m py_compile engine/training.py
python3 -m py_compile engine/inference.py
python3 -m py_compile engine/visualization.py
echo "All files OK"
```

---

## Project Structure

```
LLMVisualLab/
├── app.py                  # Main Streamlit application
├── lang.py                 # Translations (Italian / English)
├── requirements.txt        # Python dependencies
├── README.md
│
├── engine/
│   ├── __init__.py
│   ├── tokenizer.py        # Word-level tokeniser + vocabulary builder
│   ├── model.py            # TinyLM – NumPy neural language model
│   ├── training.py         # Manual backpropagation + SGD training loop
│   ├── inference.py        # Next-token prediction, embeddings, cosine similarity
│   └── visualization.py   # All Plotly charts (3D network, embeddings, heatmap…)
│
└── assets/
    ├── styles.css          # Custom dark UI styles
    └── images/             # Local image assets
```

---

## Model Architecture

```
Input (one-hot)  →  [vocab_size]
Embedding        →  [8]           W_emb  (vocab_size × 8)
Hidden (tanh)    →  [hidden_size]  W_h   (8 × hidden_size)
Output (softmax) →  [vocab_size]  W_out  (hidden_size × vocab_size)

Objective : next-token prediction (bigram)
Loss      : cross-entropy
Optimiser : SGD with lr = 0.05
Backprop  : hand-coded with NumPy (no autograd)
```

---

## Notes & Limitations

- **Educational project only** — not a production-grade language model.
- The model is a minimal bigram neural LM, not a real transformer. It is intentionally
  simplified so that every calculation can be visualised and understood.
- All computations are CPU-only and run in pure NumPy — no GPU, no PyTorch, no TensorFlow.
- Cosine similarity results reflect the *learned* embeddings from your small training set,
  which may appear random until the model has trained enough steps.
- With very small datasets (< 3 sentences) the model has little to learn from.
  Use at least 4–6 sentences for meaningful results.

---

## Dependencies

```
streamlit  ≥ 1.28
numpy      ≥ 1.24
plotly     ≥ 5.17
```

No other external packages required.

---

*LLM Visual Lab — built for curiosity, not production.*
