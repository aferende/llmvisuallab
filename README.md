# LLM Visual Lab

LLM Visual Lab is an educational, single-page Streamlit application that makes core language-model concepts visible for beginners.

## Project Goal
The goal is to open the AI "black box" by visualizing:
- tokenization,
- training signals in a tiny neural model,
- next-token prediction,
- semantic relationships through vector spaces and cosine similarity.

## Target Audience
- Non-technical users curious about AI.
- Students and educators who need an intuitive explanation of LLM fundamentals.

## Features
- **Interactive dataset setup** with custom sentences.
- **Visible tokenization** at word level.
- **CPU-only training simulation** with:
  - network graph (Input → Hidden → Output),
  - animated token flow,
  - weight-strength visual updates over steps,
  - adjustable hyperparameters (**Brain Size**, **Training Steps**).
- **Inference playground** with:
  - prompt input,
  - next-token prediction,
  - probability table,
  - 2D/3D embedding visualization,
  - cosine similarity angle-based vector plot.
- **Linux-style terminal emulator panel** for real-time logs.
- **Multi-language UI** (English / Italiano) via sidebar toggle.

## Repository Structure
```
.
├── app.py
├── engine/
│   ├── __init__.py
│   ├── micro_transformer.py
│   ├── tokenizer.py
│   └── visualization.py
├── assets/
│   └── styles.css
├── requirements.txt
└── README.md
```

## Installation Guide

### Windows
1. `git clone <repo_url>`
2. `python -m venv venv`
3. `.\venv\Scripts\activate`
4. `pip install -r requirements.txt`
5. `streamlit run app.py`

### Linux (Ubuntu)
1. `git clone <repo_url>`
2. `python3 -m venv venv`
3. `source venv/bin/activate`
4. `pip3 install -r requirements.txt`
5. `streamlit run app.py`

## Run Offline or Deploy Online
- **Offline/local**: run directly with Streamlit.
- **Online hosting**: compatible with services like Streamlit Community Cloud.

## Notes
This project intentionally uses a tiny NumPy model (no external LLM APIs) to keep every learning step transparent and inspectable.
