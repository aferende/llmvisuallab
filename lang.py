"""
Translations dictionary.
Default language: English.
All UI strings live here; code identifiers remain in English.
"""

TRANSLATIONS = {
    # ================================================================
    # ENGLISH (default)
    # ================================================================
    "en": {
        # ---- Sidebar ----
        "sidebar_lang": "Language",
        "sidebar_brain_size": "Brain size (hidden neurons)",
        "sidebar_steps": "Training steps",
        "sidebar_hyperparams": "⚙️ Hyperparameters",
        "sidebar_about": "ℹ️ About",
        "sidebar_about_text": (
            "**LLM Visual Lab** is an open-source educational lab "
            "that visually and interactively shows how a language model works."
        ),
        "sidebar_github": "⭐ GitHub Repository",
        "sidebar_credits_title": "🛠️ Built with",
        "sidebar_credits_text": (
            "- [Streamlit](https://streamlit.io) — web app framework\n"
            "- [NumPy](https://numpy.org) — neural network math\n"
            "- [Plotly](https://plotly.com) — interactive charts\n"
            "- [Claude Code](https://claude.ai/claude-code) — AI-assisted development"
        ),

        # ---- Header ----
        "app_title": "LLM Visual Lab",
        "app_subtitle": "Explore how Artificial Intelligence works",
        "intro_llm_title": "What is an LLM?",
        "intro_llm_text": (
            "A <b>Large Language Model (LLM)</b> is an AI system that has learned "
            "to understand and generate text by reading enormous amounts of text. "
            "In this lab you'll observe – on a small scale – the same mechanisms "
            "used by ChatGPT, Gemini and Claude."
        ),
        "intro_training_title": "What is Training?",
        "intro_training_text": (
            "<b>Training</b> is the phase where the model learns. "
            "We show the model many sentences and it tries to guess which word "
            "comes next. Each time it fails, it adjusts its \"internal weights\" "
            "(the numbers in the neural network) to make fewer mistakes next time."
        ),
        "intro_inference_title": "What is Inference?",
        "intro_inference_text": (
            "Once training is done, we can <b>use</b> the model: "
            "we give it the beginning of a sentence and it suggests how to continue it. "
            "This is called <b>inference</b>."
        ),
        "intro_embedding_title": "What are Embeddings?",
        "intro_embedding_text": (
            "Each word is transformed into a <b>vector of numbers</b> (embedding). "
            "Words with similar meanings end up close together in vector space. "
            "<b>Cosine similarity</b> measures how much two vectors \"point in the same "
            "direction\": the closer to 1, the more similar the words are."
        ),

        # ---- Section 1: Dataset ----
        "sec1_title": "📝 Section 1 — Dataset & Tokenisation",
        "sec1_intro": (
            "Enter the sentences you want to use to train the model. "
            "One sentence per line. You can use Italian, English or both! "
            "(max 100 sentences)"
        ),
        "token_reality_note": (
            "⚠️ <b>Educational note on tokens:</b> In real models (GPT, Claude, etc.) "
            "a token is <b>not a whole word</b> but a <b>word fragment</b> (subword). "
            "For example <i>unbelievable</i> might become 3 tokens: "
            "<code>un</code> + <code>believ</code> + <code>able</code>. "
            "This lets models handle rare words, neologisms and multiple languages "
            "with a fixed vocabulary. "
            "In this lab we use <b>one word = one token</b> for educational simplicity, "
            "so every step is easy to follow."
        ),
        "sec1_input_label": "Training sentences (one per line):",
        "sec1_default_sentences": (
            "il gatto mangia il pesce\n"
            "il cane mangia la carne\n"
            "il gatto beve il latte\n"
            "il cane beve l acqua\n"
            "the cat eats fish\n"
            "the dog eats meat\n"
            "il sole splende forte\n"
            "la luna brilla alta\n"
            "il vento soffia freddo"
        ),
        "sec1_btn_tokenize": "🔍 Analyse Dataset",
        "sec1_vocab_title": "📚 Built vocabulary",
        "sec1_vocab_info": (
            "Every unique word in the text becomes a <b>token</b>. "
            "Each token is assigned a unique number (index)."
        ),
        "sec1_pairs_title": "🔗 Training pairs (input → target)",
        "sec1_pairs_info": (
            "The model learns to predict the next word. "
            "These are the pairs it will train on."
        ),
        "sec1_tokenization_title": "🔬 Tokenisation per sentence",

        # ---- Section 2: Training ----
        "sec2_title": "⚡ Section 2 — Interactive Training",
        "sec2_intro": (
            "Click the button to start training. The 3D neural network "
            "will show in real time how weights and activations change "
            "at each step, based on your data."
        ),
        "sec2_btn_train": "🚀 Start Training",
        "sec2_btn_reset": "🔄 Reset",
        "sec2_training_label": "Training in progress...",
        "sec2_net_title": "🕸️ Real-time 3D Neural Network",
        "sec2_net_info": (
            "<b>Node colours:</b> 🔵 negative activation → ⚪ zero → 🔴 positive activation. "
            "<b>Green node:</b> input token | <b>Yellow ★ node:</b> target token. "
            "Bright edges show the active path through the network."
        ),
        "sec2_loss_title": "📉 Loss Curve",
        "sec2_loss_info": (
            "<b>Loss</b> measures how much the model is mistaken. "
            "It should decrease over time: that means the model is learning!"
        ),
        "sec2_log_title": "🖥️ Training Console",
        "sec2_done": "✅ Training complete!",
        "sec2_arch_title": "🏗️ Model architecture",
        "sec2_warn_first": "⚠️ Complete Section 1 first.",

        # ---- Section 3: Inference ----
        "sec3_title": "🔮 Section 3 — Inference & Semantic Search",
        "sec3_info_train_first": "⚠️ Please complete training in Section 2 first.",
        "sec3_intro": (
            "Enter a word or partial sentence. The model will suggest how "
            "to continue it, and you can explore the embeddings in vector space."
        ),
        "sec3_query_label": "Enter text for inference:",
        "sec3_query_placeholder": "e.g. il gatto",
        "sec3_btn_infer": "🔮 Run Inference",
        "sec3_next_token_title": "🎯 Predicted next tokens",
        "sec3_next_token_info": (
            "These are the tokens the model considers most likely "
            "as a continuation of your input."
        ),
        "sec3_emb_title": "🌐 Embedding Map",
        "sec3_emb_info": (
            "Each word is a point in vector space. "
            "Semantically similar words tend to be <b>close</b> to each other."
        ),
        "sec3_cos_title": "📐 Cosine Similarity",
        "sec3_cos_info": (
            "Select two words to see how \"close\" they are "
            "in the model's understanding."
        ),
        "sec3_word1_label": "First word:",
        "sec3_word2_label": "Second word:",
        "sec3_search_title": "🔍 Semantic Search",
        "sec3_search_info": (
            "The model compares your query with all dataset sentences "
            "and returns the most semantically similar ones, ranked by similarity."
        ),
        "sec3_search_btn": "🔍 Search",
        "sec3_heatmap_title": "🗺️ Similarity Map",
        "sec3_heatmap_info": (
            "Each cell shows how similar two sentences are according to the model. "
            "<b style='color:#e05252'>Red</b> = very similar (cos ≈ 1) | "
            "<b style='color:#aaaaaa'>White/Grey</b> = unrelated (cos ≈ 0) | "
            "<b style='color:#5278e0'>Blue</b> = opposite meaning (cos ≈ -1)"
        ),
        "not_trained_msg": "Complete training before using this section.",

        # ---- Viz labels ----
        "layer_input": "Input",
        "layer_emb": "Embedding",
        "layer_hid": "Hidden",
        "layer_out": "Output",
        "step_label": "Step",
        "loss_label": "Loss",
        "similarity_label": "Cosine similarity",
        "emb_2d_title": "Word Embeddings (PCA 2D)",
        "emb_3d_title": "Word Embeddings (PCA 3D)",
        "heatmap_title": "Cosine Similarity between Sentences",
        "probability": "Probability",
        "word": "Word",
        "sentence": "Sentence",
        "similarity": "Similarity",
        "rank": "Rank",

        # ---- Detailed explanations ----
        "net_legend_title": "🔑 Neural Network Legend",
        "net_legend_text": (
            "<b>How to read the 3D network:</b><br><br>"
            "🟢 <b>Large green node</b> = <b>input</b> token — the word the model is analysing<br>"
            "⭐ <b>Yellow node</b> = <b>target</b> token — the word the model must learn to predict<br>"
            "🔵 <b>Blue node</b> = <b>negative</b> activation (neuron is inhibited)<br>"
            "🔴 <b>Red node</b> = <b>positive</b> activation (neuron is excited)<br>"
            "⚪ <b>Grey/white node</b> = activation near <b>zero</b> (neutral neuron)<br>"
            "✨ <b>Bright arrows</b> = active signal path through the network<br><br>"
            "<b>What changes during training:</b> neurons change colour and size "
            "because their internal values (weights) are updated at every step."
        ),
        "loss_deep_explain": (
            "📉 <b>What is LOSS?</b><br><br>"
            "Loss is a number that measures <b>how much the model is wrong</b>. "
            "It is calculated by comparing the model's prediction with the correct answer.<br><br>"
            "• <b>High loss</b> (e.g. 2.5) → the model is very wrong, almost guessing<br>"
            "• <b>Low loss</b> (e.g. 0.3) → the model has learned well<br>"
            "• <b>Loss = 0</b> → the model is perfect (impossible in practice)<br><br>"
            "The graph should show a <b>descending curve</b>: that means the model is learning!"
        ),
        "weights_explain": (
            "⚖️ <b>What are WEIGHTS?</b><br><br>"
            "Weights are <b>numbers</b> that live in the connections between neurons. "
            "They are the model's 'memory': they encode everything it has learned.<br><br>"
            "• <b>Large positive weight</b> → that connection amplifies the signal<br>"
            "• <b>Large negative weight</b> → that connection inhibits the signal<br>"
            "• <b>Weight near zero</b> → weak connection, almost ignored<br><br>"
            "At every training step, all weights are adjusted by a small amount "
            "(the <i>gradient</i>) to reduce the loss."
        ),
        "cosine_sphere_title": "🌐 Vector Sphere — 3D Cosine Similarity",
        "cosine_sphere_info": (
            "Each word is a <b>vector</b> — an arrow in space. "
            "In this sphere, all vectors start from the centre and point to the surface.<br><br>"
            "🟢 = <b>{w1}</b> &nbsp;|&nbsp; 🔴 = <b>{w2}</b> &nbsp;|&nbsp; 🔵 = other words<br><br>"
            "The <b>golden arc</b> shows the <b>angle θ</b> between the two selected vectors. "
            "The <b>cosine similarity</b> is the cosine of that angle:<br>"
            "• θ = 0° → cos = <b>1.0</b> → <b>identical</b> words<br>"
            "• θ = 90° → cos = <b>0.0</b> → <b>orthogonal</b> (unrelated) words<br>"
            "• θ = 180° → cos = <b>-1.0</b> → <b>opposite</b> words"
        ),
        "dataset_tip_semantic": (
            "💡 <b>Why do some words cluster together?</b> The model learns that words "
            "appearing in similar contexts (e.g. 'cat' and 'dog' both appear after 'the' "
            "and before 'eats') end up with similar vectors. "
            "Words from different topics (e.g. 'fish' vs 'sun') end up far apart."
        ),

        # ---- Log messages ----
        "log_init": "[INFO] Initialising model...",
        "log_tokenize": "[INFO] Tokenising dataset...",
        "log_vocab": "[INFO] Vocabulary: {n} words",
        "log_pairs": "[INFO] Training pairs: {n}",
        "log_start": "[INFO] Starting training for {steps} steps...",
        "log_step": "[TRAIN] Step {step}/{total}: Loss = {loss:.4f}",
        "log_weights": "[TRAIN] Updating weights... |∇W_h|={wh:.3f} |∇W_out|={wo:.3f}",
        "log_done": "[INFO] Training complete! Final loss: {loss:.4f}",
        "log_infer": "[INFER] Input token: '{token}' (idx={idx})",
        "log_predict": "[INFER] Top prediction: '{word}' (prob={prob:.3f})",
    },

    # ================================================================
    # ITALIANO
    # ================================================================
    "it": {
        # ---- Sidebar ----
        "sidebar_lang": "Lingua / Language",
        "sidebar_brain_size": "Dimensione cervello (neuroni hidden)",
        "sidebar_steps": "Step di training",
        "sidebar_hyperparams": "⚙️ Iperparametri",
        "sidebar_about": "ℹ️ Informazioni",
        "sidebar_about_text": (
            "**LLM Visual Lab** è un laboratorio educativo open-source "
            "che mostra in modo visuale e interattivo come funziona "
            "un modello linguistico."
        ),
        "sidebar_github": "⭐ Repository GitHub",
        "sidebar_credits_title": "🛠️ Costruito con",
        "sidebar_credits_text": (
            "- [Streamlit](https://streamlit.io) — framework web app\n"
            "- [NumPy](https://numpy.org) — calcoli della rete neurale\n"
            "- [Plotly](https://plotly.com) — grafici interattivi\n"
            "- [Claude Code](https://claude.ai/claude-code) — sviluppo assistito da AI"
        ),

        # ---- Header ----
        "app_title": "LLM Visual Lab",
        "app_subtitle": "Scopri come funziona una Intelligenza Artificiale",
        "intro_llm_title": "Cos'è un LLM?",
        "intro_llm_text": (
            "Un <b>Large Language Model (LLM)</b> è un sistema di Intelligenza Artificiale "
            "che ha imparato a capire e generare testo leggendo enormi quantità di testi. "
            "In questo laboratorio potrai osservare – su scala ridotta – gli stessi "
            "meccanismi usati da ChatGPT, Gemini e Claude."
        ),
        "intro_training_title": "Cos'è il Training?",
        "intro_training_text": (
            "Il <b>training</b> è la fase in cui il modello impara. "
            "Mostriamo al modello molte frasi e lui cerca di indovinare "
            "quale parola viene dopo. Ogni volta che sbaglia, aggiusta i suoi "
            "\"pesi interni\" (i numeri nella rete neurale) per sbagliare meno "
            "la volta successiva."
        ),
        "intro_inference_title": "Cos'è l'Inferenza?",
        "intro_inference_text": (
            "Quando il training è finito, possiamo <b>usare</b> il modello: "
            "gli diamo l'inizio di una frase e lui suggerisce come continuarla. "
            "Questo si chiama <b>inferenza</b>."
        ),
        "intro_embedding_title": "Cosa sono gli Embedding?",
        "intro_embedding_text": (
            "Ogni parola viene trasformata in un <b>vettore di numeri</b> (embedding). "
            "Parole con significato simile finiscono vicine nello spazio vettoriale. "
            "La <b>similarità coseno</b> misura quanto due vettori \"puntano "
            "nella stessa direzione\": più è vicina a 1, più le parole sono simili."
        ),

        # ---- Section 1: Dataset ----
        "sec1_title": "📝 Sezione 1 — Dataset e Tokenizzazione",
        "sec1_intro": (
            "Inserisci le frasi che vuoi usare per addestrare il modello. "
            "Una frase per riga. Puoi usare italiano, inglese o entrambi! "
            "(massimo 100 frasi)"
        ),
        "token_reality_note": (
            "⚠️ <b>Nota didattica sui token:</b> Nei modelli reali (GPT, Claude, ecc.) "
            "un token <b>non è una parola intera</b>, ma un <b>frammento di parola</b> "
            "(subword). Ad esempio la parola <i>incomprensibile</i> potrebbe diventare "
            "3 token: <code>in</code> + <code>comprens</code> + <code>ibile</code>. "
            "Questo consente di gestire parole rare, neologismi e lingue diverse con un "
            "vocabolario limitato. "
            "In questo laboratorio usiamo invece <b>una parola = un token</b> "
            "per semplicità didattica, così è più facile seguire ogni passaggio."
        ),
        "sec1_input_label": "Frasi di training (una per riga):",
        "sec1_default_sentences": (
            "il gatto mangia il pesce\n"
            "il cane mangia la carne\n"
            "il gatto beve il latte\n"
            "il cane beve l acqua\n"
            "the cat eats fish\n"
            "the dog eats meat\n"
            "il sole splende forte\n"
            "la luna brilla alta\n"
            "il vento soffia freddo"
        ),
        "sec1_btn_tokenize": "🔍 Analizza Dataset",
        "sec1_vocab_title": "📚 Vocabolario costruito",
        "sec1_vocab_info": (
            "Ogni parola unica nel testo diventa un <b>token</b>. "
            "Al token viene assegnato un numero (indice) univoco."
        ),
        "sec1_pairs_title": "🔗 Coppie di training (input → target)",
        "sec1_pairs_info": (
            "Il modello impara a predire la parola successiva. "
            "Queste sono le coppie su cui si allenerà."
        ),
        "sec1_tokenization_title": "🔬 Tokenizzazione per frase",

        # ---- Section 2: Training ----
        "sec2_title": "⚡ Sezione 2 — Training Interattivo",
        "sec2_intro": (
            "Clicca il pulsante per avviare il training. La rete neurale "
            "3D mostrerà in tempo reale come cambiano i pesi e le attivazioni "
            "ad ogni step, basandosi sui tuoi dati."
        ),
        "sec2_btn_train": "🚀 Avvia Training",
        "sec2_btn_reset": "🔄 Reset",
        "sec2_training_label": "Training in corso...",
        "sec2_net_title": "🕸️ Rete Neurale 3D — in tempo reale",
        "sec2_net_info": (
            "<b>Colori dei nodi:</b> 🔵 attivazione negativa → ⚪ zero → 🔴 attivazione positiva. "
            "<b>Nodo verde:</b> token in input | <b>Nodo giallo ★:</b> token target. "
            "I bordi illuminati mostrano il percorso attivo nella rete."
        ),
        "sec2_loss_title": "📉 Curva di Loss",
        "sec2_loss_info": (
            "La <b>loss</b> misura quanto il modello sbaglia. "
            "Deve scendere nel tempo: significa che il modello sta imparando!"
        ),
        "sec2_log_title": "🖥️ Console di Training",
        "sec2_done": "✅ Training completato!",
        "sec2_arch_title": "🏗️ Architettura del modello",
        "sec2_warn_first": "⚠️ Completa prima la Sezione 1.",

        # ---- Section 3: Inference ----
        "sec3_title": "🔮 Sezione 3 — Inferenza e Ricerca Semantica",
        "sec3_info_train_first": "⚠️ Completa prima il training nella Sezione 2.",
        "sec3_intro": (
            "Inserisci una parola o una frase parziale. Il modello suggerirà "
            "come continuarla, e potrai esplorare gli embedding nello spazio vettoriale."
        ),
        "sec3_query_label": "Inserisci un testo per l'inferenza:",
        "sec3_query_placeholder": "es. il gatto",
        "sec3_btn_infer": "🔮 Inferenza",
        "sec3_next_token_title": "🎯 Token successivi predetti",
        "sec3_next_token_info": (
            "Questi sono i token che il modello ritiene più probabili "
            "come continuazione del tuo input."
        ),
        "sec3_emb_title": "🌐 Mappa degli Embedding",
        "sec3_emb_info": (
            "Ogni parola è un punto nello spazio vettoriale. "
            "Le parole semanticamente simili tendono a essere <b>vicine</b> tra loro."
        ),
        "sec3_cos_title": "📐 Similarità Coseno",
        "sec3_cos_info": (
            "Seleziona due parole per vedere quanto sono \"vicine\" "
            "nella mente del modello."
        ),
        "sec3_word1_label": "Prima parola:",
        "sec3_word2_label": "Seconda parola:",
        "sec3_search_title": "🔍 Ricerca Semantica",
        "sec3_search_info": (
            "Il modello confronta la tua ricerca con tutte le frasi del dataset "
            "e restituisce quelle semanticamente più simili, ordinate per similarità."
        ),
        "sec3_search_btn": "🔍 Cerca",
        "sec3_heatmap_title": "🗺️ Mappa di Similarità",
        "sec3_heatmap_info": (
            "Ogni cella mostra quanto sono simili due frasi secondo il modello. "
            "<b style='color:#e05252'>Rosso</b> = molto simili (cos ≈ 1) | "
            "<b style='color:#aaaaaa'>Bianco/Grigio</b> = non correlate (cos ≈ 0) | "
            "<b style='color:#5278e0'>Blu</b> = significato opposto (cos ≈ -1)"
        ),
        "not_trained_msg": "Completa il training prima di usare questa sezione.",

        # ---- Viz labels ----
        "layer_input": "Input",
        "layer_emb": "Embedding",
        "layer_hid": "Hidden",
        "layer_out": "Output",
        "step_label": "Step",
        "loss_label": "Loss",
        "similarity_label": "Similarità coseno",
        "emb_2d_title": "Embedding delle Parole (PCA 2D)",
        "emb_3d_title": "Embedding delle Parole (PCA 3D)",
        "heatmap_title": "Similarità Coseno tra Frasi",
        "probability": "Probabilità",
        "word": "Parola",
        "sentence": "Frase",
        "similarity": "Similarità",
        "rank": "Posizione",

        # ---- Detailed explanations ----
        "net_legend_title": "🔑 Legenda della rete neurale",
        "net_legend_text": (
            "<b>Come leggere la rete 3D:</b><br><br>"
            "🟢 <b>Nodo verde grande</b> = token in <b>input</b> — la parola che il modello sta analizzando<br>"
            "⭐ <b>Nodo giallo</b> = token <b>target</b> — la parola che il modello deve imparare a predire<br>"
            "🔵 <b>Nodo blu</b> = attivazione <b>negativa</b> (il neurone è inibito)<br>"
            "🔴 <b>Nodo rosso</b> = attivazione <b>positiva</b> (il neurone è eccitato)<br>"
            "⚪ <b>Nodo grigio/bianco</b> = attivazione vicina a <b>zero</b> (neurone neutro)<br>"
            "✨ <b>Frecce luminose</b> = percorso attivo del segnale nella rete<br><br>"
            "<b>Cosa cambia durante il training:</b> i neuroni cambiano colore e dimensione "
            "perché i loro valori interni (pesi) vengono aggiornati ad ogni step."
        ),
        "loss_deep_explain": (
            "📉 <b>Cos'è la LOSS (perdita)?</b><br><br>"
            "La loss è un numero che misura <b>quanto il modello sbaglia</b>. "
            "Si calcola confrontando la predizione del modello con la risposta corretta.<br><br>"
            "• <b>Loss alta</b> (es. 2.5) → il modello sbaglia molto, quasi indovina a caso<br>"
            "• <b>Loss bassa</b> (es. 0.3) → il modello ha imparato bene<br>"
            "• <b>Loss = 0</b> → il modello è perfetto (impossibile in pratica)<br><br>"
            "Il grafico deve mostrare una curva <b>discendente</b>: significa che il modello sta imparando!"
        ),
        "weights_explain": (
            "⚖️ <b>Cosa sono i PESI?</b><br><br>"
            "I pesi sono <b>numeri</b> che vivono nelle connessioni tra i neuroni. "
            "Sono la 'memoria' del modello: codificano tutto ciò che ha imparato.<br><br>"
            "• <b>Peso alto positivo</b> → quella connessione amplifica il segnale<br>"
            "• <b>Peso alto negativo</b> → quella connessione inibisce il segnale<br>"
            "• <b>Peso vicino a zero</b> → connessione debole, quasi ignorata<br><br>"
            "Ad ogni step di training, tutti i pesi vengono aggiustati di una piccola "
            "quantità (il <i>gradiente</i>) per ridurre la loss."
        ),
        "cosine_sphere_title": "🌐 Sfera dei Vettori — Similarità Coseno 3D",
        "cosine_sphere_info": (
            "Ogni parola è un <b>vettore</b> — una freccia nello spazio. "
            "In questa sfera, tutti i vettori partono dal centro e puntano verso la superficie.<br><br>"
            "🟢 = <b>{w1}</b> &nbsp;|&nbsp; 🔴 = <b>{w2}</b> &nbsp;|&nbsp; 🔵 = altre parole<br><br>"
            "L'<b>arco dorato</b> mostra l'<b>angolo θ</b> tra i due vettori selezionati. "
            "La <b>similarità coseno</b> è il coseno di quell'angolo:<br>"
            "• θ = 0° → cos = <b>1.0</b> → parole <b>identiche</b><br>"
            "• θ = 90° → cos = <b>0.0</b> → parole <b>ortogonali</b> (non correlate)<br>"
            "• θ = 180° → cos = <b>-1.0</b> → parole <b>opposte</b>"
        ),
        "dataset_tip_semantic": (
            "💡 <b>Perché alcune parole si raggruppano?</b> Il modello apprende che le parole "
            "che appaiono in contesti simili (es. 'gatto' e 'cane' appaiono entrambe dopo 'il' "
            "e prima di 'mangia') hanno vettori simili. "
            "Parole di argomenti diversi (es. 'pesce' vs 'sole') finiscono lontane."
        ),

        # ---- Log messages ----
        "log_init": "[INFO] Inizializzazione modello...",
        "log_tokenize": "[INFO] Tokenizzazione dataset...",
        "log_vocab": "[INFO] Vocabolario: {n} parole",
        "log_pairs": "[INFO] Coppie di training: {n}",
        "log_start": "[INFO] Avvio training per {steps} step...",
        "log_step": "[TRAIN] Step {step}/{total}: Loss = {loss:.4f}",
        "log_weights": "[TRAIN] Aggiornamento pesi... |∇W_h|={wh:.3f} |∇W_out|={wo:.3f}",
        "log_done": "[INFO] Training completato! Loss finale: {loss:.4f}",
        "log_infer": "[INFER] Token input: '{token}' (idx={idx})",
        "log_predict": "[INFER] Top predizione: '{word}' (prob={prob:.3f})",
    },

    # ================================================================
    # FRANÇAIS
    # ================================================================
    "fr": {
        "sidebar_lang": "Langue / Language",
        "sidebar_brain_size": "Taille du cerveau (neurones cachés)",
        "sidebar_steps": "Étapes d'entraînement",
        "sidebar_hyperparams": "⚙️ Hyperparamètres",
        "sidebar_about": "ℹ️ À propos",
        "sidebar_about_text": (
            "**LLM Visual Lab** est un laboratoire éducatif open-source "
            "qui montre visuellement et interactivement le fonctionnement "
            "d'un modèle de langage."
        ),
        "sidebar_github": "⭐ Dépôt GitHub",
        "sidebar_credits_title": "🛠️ Construit avec",
        "sidebar_credits_text": (
            "- [Streamlit](https://streamlit.io) — framework web\n"
            "- [NumPy](https://numpy.org) — calculs du réseau neuronal\n"
            "- [Plotly](https://plotly.com) — graphiques interactifs\n"
            "- [Claude Code](https://claude.ai/claude-code) — développement assisté par IA"
        ),
        "app_title": "LLM Visual Lab",
        "app_subtitle": "Découvrez le fonctionnement de l'Intelligence Artificielle",
        "intro_llm_title": "Qu'est-ce qu'un LLM ?",
        "intro_llm_text": (
            "Un <b>Grand Modèle de Langage (LLM)</b> est un système d'IA qui a appris "
            "à comprendre et générer du texte en lisant d'énormes quantités de textes. "
            "Dans ce laboratoire, vous observerez – à petite échelle – les mêmes "
            "mécanismes utilisés par ChatGPT, Gemini et Claude."
        ),
        "intro_training_title": "Qu'est-ce que l'Entraînement ?",
        "intro_training_text": (
            "L'<b>entraînement</b> est la phase où le modèle apprend. "
            "On lui montre de nombreuses phrases et il essaie de deviner "
            "quel mot vient ensuite. Chaque erreur lui permet d'ajuster ses "
            "\"poids internes\" pour mieux prédire la prochaine fois."
        ),
        "intro_inference_title": "Qu'est-ce que l'Inférence ?",
        "intro_inference_text": (
            "Une fois l'entraînement terminé, on peut <b>utiliser</b> le modèle : "
            "on lui donne le début d'une phrase et il suggère comment la continuer. "
            "C'est ce qu'on appelle l'<b>inférence</b>."
        ),
        "intro_embedding_title": "Que sont les Embeddings ?",
        "intro_embedding_text": (
            "Chaque mot est transformé en un <b>vecteur de nombres</b> (embedding). "
            "Les mots au sens similaire se retrouvent proches dans l'espace vectoriel. "
            "La <b>similarité cosinus</b> mesure à quel point deux vecteurs \"pointent "
            "dans la même direction\" : plus elle est proche de 1, plus les mots sont similaires."
        ),
        "sec1_title": "📝 Section 1 — Jeu de données & Tokenisation",
        "sec1_intro": (
            "Entrez les phrases que vous souhaitez utiliser pour entraîner le modèle. "
            "Une phrase par ligne. Vous pouvez utiliser l'italien, l'anglais ou les deux ! "
            "(max 100 phrases)"
        ),
        "token_reality_note": (
            "⚠️ <b>Note pédagogique sur les tokens :</b> Dans les vrais modèles (GPT, Claude, etc.) "
            "un token n'est <b>pas un mot entier</b> mais un <b>fragment de mot</b> (sous-mot). "
            "Par exemple <i>incompréhensible</i> peut devenir 3 tokens : "
            "<code>in</code> + <code>compréhen</code> + <code>sible</code>. "
            "Dans ce laboratoire, nous utilisons <b>un mot = un token</b> pour simplifier."
        ),
        "sec1_input_label": "Phrases d'entraînement (une par ligne) :",
        "sec1_default_sentences": (
            "il gatto mangia il pesce\n"
            "il cane mangia la carne\n"
            "il gatto beve il latte\n"
            "il cane beve l acqua\n"
            "the cat eats fish\n"
            "the dog eats meat\n"
            "il sole splende forte\n"
            "la luna brilla alta\n"
            "il vento soffia freddo"
        ),
        "sec1_btn_tokenize": "🔍 Analyser le jeu de données",
        "sec1_vocab_title": "📚 Vocabulaire construit",
        "sec1_vocab_info": "Chaque mot unique devient un <b>token</b> avec un index unique.",
        "sec1_pairs_title": "🔗 Paires d'entraînement (entrée → cible)",
        "sec1_pairs_info": "Le modèle apprend à prédire le mot suivant. Voici les paires.",
        "sec1_tokenization_title": "🔬 Tokenisation par phrase",
        "sec2_title": "⚡ Section 2 — Entraînement Interactif",
        "sec2_intro": "Cliquez pour démarrer l'entraînement. Le réseau 3D se met à jour en temps réel.",
        "sec2_btn_train": "🚀 Démarrer l'entraînement",
        "sec2_btn_reset": "🔄 Réinitialiser",
        "sec2_training_label": "Entraînement en cours...",
        "sec2_net_title": "🕸️ Réseau Neuronal 3D en temps réel",
        "sec2_net_info": (
            "<b>Couleurs des nœuds :</b> 🔵 activation négative → ⚪ zéro → 🔴 activation positive. "
            "<b>Nœud vert :</b> token d'entrée | <b>Nœud jaune ★ :</b> token cible."
        ),
        "sec2_loss_title": "📉 Courbe de Loss",
        "sec2_loss_info": "La <b>loss</b> mesure l'erreur du modèle. Elle doit diminuer.",
        "sec2_log_title": "🖥️ Console d'entraînement",
        "sec2_done": "✅ Entraînement terminé !",
        "sec2_arch_title": "🏗️ Architecture du modèle",
        "sec2_warn_first": "⚠️ Complétez d'abord la Section 1.",
        "sec3_title": "🔮 Section 3 — Inférence & Recherche Sémantique",
        "sec3_info_train_first": "⚠️ Complétez d'abord l'entraînement en Section 2.",
        "sec3_intro": "Entrez un mot ou une phrase partielle. Le modèle suggérera la suite.",
        "sec3_query_label": "Texte pour l'inférence :",
        "sec3_query_placeholder": "ex. il gatto",
        "sec3_btn_infer": "🔮 Lancer l'inférence",
        "sec3_next_token_title": "🎯 Tokens suivants prédits",
        "sec3_next_token_info": "Les tokens que le modèle considère comme les plus probables.",
        "sec3_emb_title": "🌐 Carte des Embeddings",
        "sec3_emb_info": "Chaque mot est un point dans l'espace vectoriel. Les mots similaires sont <b>proches</b>.",
        "sec3_cos_title": "📐 Similarité Cosinus",
        "sec3_cos_info": "Sélectionnez deux mots pour voir leur proximité sémantique.",
        "sec3_word1_label": "Premier mot :",
        "sec3_word2_label": "Deuxième mot :",
        "sec3_search_title": "🔍 Recherche Sémantique",
        "sec3_search_info": "Le modèle compare votre requête avec toutes les phrases et retourne les plus similaires.",
        "sec3_search_btn": "🔍 Rechercher",
        "sec3_heatmap_title": "🗺️ Carte de Similarité",
        "sec3_heatmap_info": (
            "Chaque cellule montre la similarité entre deux phrases. "
            "<b style='color:#e05252'>Rouge</b> = très similaires | "
            "<b style='color:#aaaaaa'>Blanc</b> = non corrélées | "
            "<b style='color:#5278e0'>Bleu</b> = sens opposé"
        ),
        "not_trained_msg": "Terminez l'entraînement avant d'utiliser cette section.",
        "layer_input": "Entrée", "layer_emb": "Embedding", "layer_hid": "Caché", "layer_out": "Sortie",
        "step_label": "Étape", "loss_label": "Loss", "similarity_label": "Similarité cosinus",
        "emb_2d_title": "Embeddings des Mots (PCA 2D)", "emb_3d_title": "Embeddings des Mots (PCA 3D)",
        "heatmap_title": "Similarité Cosinus entre Phrases",
        "probability": "Probabilité", "word": "Mot", "sentence": "Phrase",
        "similarity": "Similarité", "rank": "Rang",
        "net_legend_title": "🔑 Légende du réseau neuronal",
        "net_legend_text": (
            "<b>Comment lire le réseau 3D :</b><br><br>"
            "🟢 <b>Nœud vert</b> = token d'<b>entrée</b><br>"
            "⭐ <b>Nœud jaune</b> = token <b>cible</b><br>"
            "🔵 <b>Nœud bleu</b> = activation <b>négative</b><br>"
            "🔴 <b>Nœud rouge</b> = activation <b>positive</b><br>"
            "⚪ <b>Nœud gris</b> = activation proche de <b>zéro</b>"
        ),
        "loss_deep_explain": (
            "📉 <b>Qu'est-ce que la LOSS ?</b><br><br>"
            "La loss mesure <b>l'erreur du modèle</b>. "
            "• <b>Loss élevée</b> → le modèle se trompe beaucoup<br>"
            "• <b>Loss faible</b> → le modèle a bien appris<br>"
            "Le graphique doit montrer une courbe <b>descendante</b>."
        ),
        "weights_explain": (
            "⚖️ <b>Que sont les POIDS ?</b><br><br>"
            "Les poids sont des <b>nombres</b> dans les connexions entre neurones. "
            "Ils encodent tout ce que le modèle a appris."
        ),
        "cosine_sphere_title": "🌐 Sphère Vectorielle — Similarité Cosinus 3D",
        "cosine_sphere_info": (
            "Chaque mot est un <b>vecteur</b> — une flèche dans l'espace.<br><br>"
            "🟢 = <b>{w1}</b> &nbsp;|&nbsp; 🔴 = <b>{w2}</b> &nbsp;|&nbsp; 🔵 = autres mots<br><br>"
            "L'<b>arc doré</b> montre l'<b>angle θ</b> entre les deux vecteurs.<br>"
            "• θ = 0° → cos = <b>1.0</b> → mots <b>identiques</b><br>"
            "• θ = 90° → cos = <b>0.0</b> → mots <b>non corrélés</b><br>"
            "• θ = 180° → cos = <b>-1.0</b> → mots <b>opposés</b>"
        ),
        "dataset_tip_semantic": (
            "💡 <b>Pourquoi certains mots se regroupent-ils ?</b> Le modèle apprend que les mots "
            "apparaissant dans des contextes similaires ont des vecteurs similaires."
        ),
        "log_init": "[INFO] Initialisation du modèle...",
        "log_tokenize": "[INFO] Tokenisation du jeu de données...",
        "log_vocab": "[INFO] Vocabulaire : {n} mots",
        "log_pairs": "[INFO] Paires d'entraînement : {n}",
        "log_start": "[INFO] Démarrage de l'entraînement pour {steps} étapes...",
        "log_step": "[TRAIN] Étape {step}/{total} : Loss = {loss:.4f}",
        "log_weights": "[TRAIN] Mise à jour des poids... |∇W_h|={wh:.3f} |∇W_out|={wo:.3f}",
        "log_done": "[INFO] Entraînement terminé ! Loss finale : {loss:.4f}",
        "log_infer": "[INFER] Token d'entrée : '{token}' (idx={idx})",
        "log_predict": "[INFER] Meilleure prédiction : '{word}' (prob={prob:.3f})",
    },

    # ================================================================
    # DEUTSCH
    # ================================================================
    "de": {
        "sidebar_lang": "Sprache / Language",
        "sidebar_brain_size": "Gehirngröße (versteckte Neuronen)",
        "sidebar_steps": "Trainingsschritte",
        "sidebar_hyperparams": "⚙️ Hyperparameter",
        "sidebar_about": "ℹ️ Über",
        "sidebar_about_text": (
            "**LLM Visual Lab** ist ein Open-Source-Bildungslabor, "
            "das visuell und interaktiv zeigt, wie ein Sprachmodell funktioniert."
        ),
        "sidebar_github": "⭐ GitHub-Repository",
        "sidebar_credits_title": "🛠️ Erstellt mit",
        "sidebar_credits_text": (
            "- [Streamlit](https://streamlit.io) — Web-App-Framework\n"
            "- [NumPy](https://numpy.org) — neuronale Netzwerkberechnungen\n"
            "- [Plotly](https://plotly.com) — interaktive Diagramme\n"
            "- [Claude Code](https://claude.ai/claude-code) — KI-unterstützte Entwicklung"
        ),
        "app_title": "LLM Visual Lab",
        "app_subtitle": "Entdecken Sie, wie Künstliche Intelligenz funktioniert",
        "intro_llm_title": "Was ist ein LLM?",
        "intro_llm_text": (
            "Ein <b>Großes Sprachmodell (LLM)</b> ist ein KI-System, das gelernt hat, "
            "Text zu verstehen und zu generieren, indem es riesige Textmengen gelesen hat. "
            "In diesem Labor beobachten Sie – in kleinem Maßstab – dieselben "
            "Mechanismen wie ChatGPT, Gemini und Claude."
        ),
        "intro_training_title": "Was ist Training?",
        "intro_training_text": (
            "<b>Training</b> ist die Phase, in der das Modell lernt. "
            "Wir zeigen dem Modell viele Sätze und es versucht, das nächste Wort vorherzusagen. "
            "Bei jedem Fehler passt es seine \"internen Gewichte\" an."
        ),
        "intro_inference_title": "Was ist Inferenz?",
        "intro_inference_text": (
            "Nach dem Training können wir das Modell <b>nutzen</b>: "
            "Wir geben ihm den Anfang eines Satzes, und es schlägt vor, wie es weitergeht. "
            "Das nennt man <b>Inferenz</b>."
        ),
        "intro_embedding_title": "Was sind Embeddings?",
        "intro_embedding_text": (
            "Jedes Wort wird in einen <b>Zahlenvektor</b> (Embedding) umgewandelt. "
            "Wörter mit ähnlicher Bedeutung landen nahe beieinander im Vektorraum. "
            "Die <b>Kosinus-Ähnlichkeit</b> misst, wie sehr zwei Vektoren \"in dieselbe Richtung zeigen\"."
        ),
        "sec1_title": "📝 Abschnitt 1 — Datensatz & Tokenisierung",
        "sec1_intro": (
            "Geben Sie die Sätze ein, die Sie zum Training verwenden möchten. "
            "Ein Satz pro Zeile. (max. 100 Sätze)"
        ),
        "token_reality_note": (
            "⚠️ <b>Didaktischer Hinweis zu Tokens:</b> In echten Modellen (GPT, Claude usw.) "
            "ist ein Token <b>kein ganzes Wort</b>, sondern ein <b>Wortfragment</b> (Subword). "
            "In diesem Labor verwenden wir <b>ein Wort = ein Token</b> zur Vereinfachung."
        ),
        "sec1_input_label": "Trainingssätze (einer pro Zeile):",
        "sec1_default_sentences": (
            "il gatto mangia il pesce\n"
            "il cane mangia la carne\n"
            "il gatto beve il latte\n"
            "il cane beve l acqua\n"
            "the cat eats fish\n"
            "the dog eats meat\n"
            "il sole splende forte\n"
            "la luna brilla alta\n"
            "il vento soffia freddo"
        ),
        "sec1_btn_tokenize": "🔍 Datensatz analysieren",
        "sec1_vocab_title": "📚 Erstelltes Vokabular",
        "sec1_vocab_info": "Jedes eindeutige Wort wird zu einem <b>Token</b> mit einem eindeutigen Index.",
        "sec1_pairs_title": "🔗 Trainingspaare (Eingabe → Ziel)",
        "sec1_pairs_info": "Das Modell lernt, das nächste Wort vorherzusagen.",
        "sec1_tokenization_title": "🔬 Tokenisierung pro Satz",
        "sec2_title": "⚡ Abschnitt 2 — Interaktives Training",
        "sec2_intro": "Klicken Sie, um das Training zu starten. Das 3D-Netz aktualisiert sich in Echtzeit.",
        "sec2_btn_train": "🚀 Training starten",
        "sec2_btn_reset": "🔄 Zurücksetzen",
        "sec2_training_label": "Training läuft...",
        "sec2_net_title": "🕸️ 3D Neuronales Netz in Echtzeit",
        "sec2_net_info": (
            "<b>Knotenfarben:</b> 🔵 negative Aktivierung → ⚪ null → 🔴 positive Aktivierung. "
            "<b>Grüner Knoten:</b> Eingabe-Token | <b>Gelber ★-Knoten:</b> Ziel-Token."
        ),
        "sec2_loss_title": "📉 Verlustkurve",
        "sec2_loss_info": "Der <b>Verlust</b> misst den Fehler des Modells. Er sollte sinken.",
        "sec2_log_title": "🖥️ Trainingskonsole",
        "sec2_done": "✅ Training abgeschlossen!",
        "sec2_arch_title": "🏗️ Modellarchitektur",
        "sec2_warn_first": "⚠️ Bitte zuerst Abschnitt 1 abschließen.",
        "sec3_title": "🔮 Abschnitt 3 — Inferenz & Semantische Suche",
        "sec3_info_train_first": "⚠️ Bitte zuerst das Training in Abschnitt 2 abschließen.",
        "sec3_intro": "Geben Sie ein Wort oder einen Teilsatz ein. Das Modell schlägt eine Fortsetzung vor.",
        "sec3_query_label": "Text für die Inferenz:",
        "sec3_query_placeholder": "z.B. il gatto",
        "sec3_btn_infer": "🔮 Inferenz starten",
        "sec3_next_token_title": "🎯 Vorhergesagte nächste Tokens",
        "sec3_next_token_info": "Die Tokens, die das Modell als wahrscheinlichste Fortsetzung betrachtet.",
        "sec3_emb_title": "🌐 Embedding-Karte",
        "sec3_emb_info": "Jedes Wort ist ein Punkt im Vektorraum. Ähnliche Wörter sind <b>nahe beieinander</b>.",
        "sec3_cos_title": "📐 Kosinus-Ähnlichkeit",
        "sec3_cos_info": "Wählen Sie zwei Wörter aus, um ihre semantische Nähe zu sehen.",
        "sec3_word1_label": "Erstes Wort:",
        "sec3_word2_label": "Zweites Wort:",
        "sec3_search_title": "🔍 Semantische Suche",
        "sec3_search_info": "Das Modell vergleicht Ihre Anfrage mit allen Sätzen und gibt die ähnlichsten zurück.",
        "sec3_search_btn": "🔍 Suchen",
        "sec3_heatmap_title": "🗺️ Ähnlichkeitskarte",
        "sec3_heatmap_info": (
            "Jede Zelle zeigt die Ähnlichkeit zwischen zwei Sätzen. "
            "<b style='color:#e05252'>Rot</b> = sehr ähnlich | "
            "<b style='color:#aaaaaa'>Weiß</b> = unkorreliert | "
            "<b style='color:#5278e0'>Blau</b> = entgegengesetzte Bedeutung"
        ),
        "not_trained_msg": "Schließen Sie das Training ab, bevor Sie diesen Abschnitt nutzen.",
        "layer_input": "Eingabe", "layer_emb": "Embedding", "layer_hid": "Versteckt", "layer_out": "Ausgabe",
        "step_label": "Schritt", "loss_label": "Verlust", "similarity_label": "Kosinus-Ähnlichkeit",
        "emb_2d_title": "Wort-Embeddings (PCA 2D)", "emb_3d_title": "Wort-Embeddings (PCA 3D)",
        "heatmap_title": "Kosinus-Ähnlichkeit zwischen Sätzen",
        "probability": "Wahrscheinlichkeit", "word": "Wort", "sentence": "Satz",
        "similarity": "Ähnlichkeit", "rank": "Rang",
        "net_legend_title": "🔑 Legende des neuronalen Netzes",
        "net_legend_text": (
            "<b>So lesen Sie das 3D-Netz:</b><br><br>"
            "🟢 <b>Grüner Knoten</b> = <b>Eingabe</b>-Token<br>"
            "⭐ <b>Gelber Knoten</b> = <b>Ziel</b>-Token<br>"
            "🔵 <b>Blauer Knoten</b> = <b>negative</b> Aktivierung<br>"
            "🔴 <b>Roter Knoten</b> = <b>positive</b> Aktivierung<br>"
            "⚪ <b>Grauer Knoten</b> = Aktivierung nahe <b>null</b>"
        ),
        "loss_deep_explain": (
            "📉 <b>Was ist VERLUST (Loss)?</b><br><br>"
            "Der Verlust misst <b>wie sehr das Modell falsch liegt</b>.<br>"
            "• <b>Hoher Verlust</b> → Modell macht viele Fehler<br>"
            "• <b>Niedriger Verlust</b> → Modell hat gut gelernt<br>"
            "Der Graph sollte eine <b>absteigende Kurve</b> zeigen."
        ),
        "weights_explain": (
            "⚖️ <b>Was sind GEWICHTE?</b><br><br>"
            "Gewichte sind <b>Zahlen</b> in den Verbindungen zwischen Neuronen. "
            "Sie kodieren alles, was das Modell gelernt hat."
        ),
        "cosine_sphere_title": "🌐 Vektorsphäre — 3D Kosinus-Ähnlichkeit",
        "cosine_sphere_info": (
            "Jedes Wort ist ein <b>Vektor</b> — ein Pfeil im Raum.<br><br>"
            "🟢 = <b>{w1}</b> &nbsp;|&nbsp; 🔴 = <b>{w2}</b> &nbsp;|&nbsp; 🔵 = andere Wörter<br><br>"
            "Der <b>goldene Bogen</b> zeigt den <b>Winkel θ</b> zwischen den zwei Vektoren.<br>"
            "• θ = 0° → cos = <b>1.0</b> → <b>identische</b> Wörter<br>"
            "• θ = 90° → cos = <b>0.0</b> → <b>unkorrelierte</b> Wörter<br>"
            "• θ = 180° → cos = <b>-1.0</b> → <b>gegensätzliche</b> Wörter"
        ),
        "dataset_tip_semantic": (
            "💡 <b>Warum gruppieren sich manche Wörter?</b> Das Modell lernt, dass Wörter "
            "in ähnlichen Kontexten ähnliche Vektoren haben."
        ),
        "log_init": "[INFO] Modell wird initialisiert...",
        "log_tokenize": "[INFO] Datensatz wird tokenisiert...",
        "log_vocab": "[INFO] Vokabular: {n} Wörter",
        "log_pairs": "[INFO] Trainingspaare: {n}",
        "log_start": "[INFO] Training für {steps} Schritte wird gestartet...",
        "log_step": "[TRAIN] Schritt {step}/{total}: Verlust = {loss:.4f}",
        "log_weights": "[TRAIN] Gewichte werden aktualisiert... |∇W_h|={wh:.3f} |∇W_out|={wo:.3f}",
        "log_done": "[INFO] Training abgeschlossen! Endverlust: {loss:.4f}",
        "log_infer": "[INFER] Eingabe-Token: '{token}' (idx={idx})",
        "log_predict": "[INFER] Beste Vorhersage: '{word}' (prob={prob:.3f})",
    },

    # ================================================================
    # ESPAÑOL
    # ================================================================
    "es": {
        "sidebar_lang": "Idioma / Language",
        "sidebar_brain_size": "Tamaño del cerebro (neuronas ocultas)",
        "sidebar_steps": "Pasos de entrenamiento",
        "sidebar_hyperparams": "⚙️ Hiperparámetros",
        "sidebar_about": "ℹ️ Acerca de",
        "sidebar_about_text": (
            "**LLM Visual Lab** es un laboratorio educativo open-source "
            "que muestra visual e interactivamente cómo funciona un modelo de lenguaje."
        ),
        "sidebar_github": "⭐ Repositorio GitHub",
        "sidebar_credits_title": "🛠️ Construido con",
        "sidebar_credits_text": (
            "- [Streamlit](https://streamlit.io) — framework web\n"
            "- [NumPy](https://numpy.org) — cálculos de red neuronal\n"
            "- [Plotly](https://plotly.com) — gráficos interactivos\n"
            "- [Claude Code](https://claude.ai/claude-code) — desarrollo asistido por IA"
        ),
        "app_title": "LLM Visual Lab",
        "app_subtitle": "Descubre cómo funciona la Inteligencia Artificial",
        "intro_llm_title": "¿Qué es un LLM?",
        "intro_llm_text": (
            "Un <b>Gran Modelo de Lenguaje (LLM)</b> es un sistema de IA que ha aprendido "
            "a comprender y generar texto leyendo enormes cantidades de texto. "
            "En este laboratorio observarás – a pequeña escala – los mismos mecanismos "
            "utilizados por ChatGPT, Gemini y Claude."
        ),
        "intro_training_title": "¿Qué es el Entrenamiento?",
        "intro_training_text": (
            "El <b>entrenamiento</b> es la fase donde el modelo aprende. "
            "Le mostramos muchas frases e intenta adivinar qué palabra viene después. "
            "Cada vez que falla, ajusta sus \"pesos internos\" para cometer menos errores."
        ),
        "intro_inference_title": "¿Qué es la Inferencia?",
        "intro_inference_text": (
            "Cuando termina el entrenamiento, podemos <b>usar</b> el modelo: "
            "le damos el inicio de una frase y sugiere cómo continuarla. "
            "Esto se llama <b>inferencia</b>."
        ),
        "intro_embedding_title": "¿Qué son los Embeddings?",
        "intro_embedding_text": (
            "Cada palabra se transforma en un <b>vector de números</b> (embedding). "
            "Las palabras con significado similar terminan cerca en el espacio vectorial. "
            "La <b>similitud coseno</b> mide cuánto apuntan dos vectores \"en la misma dirección\"."
        ),
        "sec1_title": "📝 Sección 1 — Dataset y Tokenización",
        "sec1_intro": (
            "Introduce las frases que quieres usar para entrenar el modelo. "
            "Una frase por línea. (máx. 100 frases)"
        ),
        "token_reality_note": (
            "⚠️ <b>Nota didáctica sobre tokens:</b> En modelos reales (GPT, Claude, etc.) "
            "un token <b>no es una palabra completa</b> sino un <b>fragmento de palabra</b> (subword). "
            "En este laboratorio usamos <b>una palabra = un token</b> para simplificar."
        ),
        "sec1_input_label": "Frases de entrenamiento (una por línea):",
        "sec1_default_sentences": (
            "il gatto mangia il pesce\n"
            "il cane mangia la carne\n"
            "il gatto beve il latte\n"
            "il cane beve l acqua\n"
            "the cat eats fish\n"
            "the dog eats meat\n"
            "il sole splende forte\n"
            "la luna brilla alta\n"
            "il vento soffia freddo"
        ),
        "sec1_btn_tokenize": "🔍 Analizar Dataset",
        "sec1_vocab_title": "📚 Vocabulario construido",
        "sec1_vocab_info": "Cada palabra única se convierte en un <b>token</b> con un índice único.",
        "sec1_pairs_title": "🔗 Pares de entrenamiento (entrada → objetivo)",
        "sec1_pairs_info": "El modelo aprende a predecir la siguiente palabra.",
        "sec1_tokenization_title": "🔬 Tokenización por frase",
        "sec2_title": "⚡ Sección 2 — Entrenamiento Interactivo",
        "sec2_intro": "Haz clic para iniciar el entrenamiento. La red 3D se actualiza en tiempo real.",
        "sec2_btn_train": "🚀 Iniciar Entrenamiento",
        "sec2_btn_reset": "🔄 Reiniciar",
        "sec2_training_label": "Entrenando...",
        "sec2_net_title": "🕸️ Red Neuronal 3D en tiempo real",
        "sec2_net_info": (
            "<b>Colores de nodos:</b> 🔵 activación negativa → ⚪ cero → 🔴 activación positiva. "
            "<b>Nodo verde:</b> token de entrada | <b>Nodo amarillo ★:</b> token objetivo."
        ),
        "sec2_loss_title": "📉 Curva de Loss",
        "sec2_loss_info": "La <b>loss</b> mide el error del modelo. Debe disminuir con el tiempo.",
        "sec2_log_title": "🖥️ Consola de Entrenamiento",
        "sec2_done": "✅ ¡Entrenamiento completado!",
        "sec2_arch_title": "🏗️ Arquitectura del modelo",
        "sec2_warn_first": "⚠️ Completa primero la Sección 1.",
        "sec3_title": "🔮 Sección 3 — Inferencia y Búsqueda Semántica",
        "sec3_info_train_first": "⚠️ Completa primero el entrenamiento en la Sección 2.",
        "sec3_intro": "Introduce una palabra o frase parcial. El modelo sugerirá cómo continuar.",
        "sec3_query_label": "Texto para inferencia:",
        "sec3_query_placeholder": "ej. il gatto",
        "sec3_btn_infer": "🔮 Inferencia",
        "sec3_next_token_title": "🎯 Tokens siguientes predichos",
        "sec3_next_token_info": "Los tokens que el modelo considera más probables como continuación.",
        "sec3_emb_title": "🌐 Mapa de Embeddings",
        "sec3_emb_info": "Cada palabra es un punto en el espacio vectorial. Las similares están <b>cerca</b>.",
        "sec3_cos_title": "📐 Similitud Coseno",
        "sec3_cos_info": "Selecciona dos palabras para ver su cercanía semántica.",
        "sec3_word1_label": "Primera palabra:",
        "sec3_word2_label": "Segunda palabra:",
        "sec3_search_title": "🔍 Búsqueda Semántica",
        "sec3_search_info": "El modelo compara tu búsqueda con todas las frases y devuelve las más similares.",
        "sec3_search_btn": "🔍 Buscar",
        "sec3_heatmap_title": "🗺️ Mapa de Similitud",
        "sec3_heatmap_info": (
            "Cada celda muestra la similitud entre dos frases. "
            "<b style='color:#e05252'>Rojo</b> = muy similares | "
            "<b style='color:#aaaaaa'>Blanco</b> = no correlacionadas | "
            "<b style='color:#5278e0'>Azul</b> = significado opuesto"
        ),
        "not_trained_msg": "Completa el entrenamiento antes de usar esta sección.",
        "layer_input": "Entrada", "layer_emb": "Embedding", "layer_hid": "Oculta", "layer_out": "Salida",
        "step_label": "Paso", "loss_label": "Loss", "similarity_label": "Similitud coseno",
        "emb_2d_title": "Embeddings de Palabras (PCA 2D)", "emb_3d_title": "Embeddings de Palabras (PCA 3D)",
        "heatmap_title": "Similitud Coseno entre Frases",
        "probability": "Probabilidad", "word": "Palabra", "sentence": "Frase",
        "similarity": "Similitud", "rank": "Posición",
        "net_legend_title": "🔑 Leyenda de la red neuronal",
        "net_legend_text": (
            "<b>Cómo leer la red 3D:</b><br><br>"
            "🟢 <b>Nodo verde</b> = token de <b>entrada</b><br>"
            "⭐ <b>Nodo amarillo</b> = token <b>objetivo</b><br>"
            "🔵 <b>Nodo azul</b> = activación <b>negativa</b><br>"
            "🔴 <b>Nodo rojo</b> = activación <b>positiva</b><br>"
            "⚪ <b>Nodo gris</b> = activación cercana a <b>cero</b>"
        ),
        "loss_deep_explain": (
            "📉 <b>¿Qué es la LOSS?</b><br><br>"
            "La loss mide <b>cuánto se equivoca el modelo</b>.<br>"
            "• <b>Loss alta</b> → el modelo comete muchos errores<br>"
            "• <b>Loss baja</b> → el modelo ha aprendido bien<br>"
            "El gráfico debe mostrar una curva <b>descendente</b>."
        ),
        "weights_explain": (
            "⚖️ <b>¿Qué son los PESOS?</b><br><br>"
            "Los pesos son <b>números</b> en las conexiones entre neuronas. "
            "Codifican todo lo que el modelo ha aprendido."
        ),
        "cosine_sphere_title": "🌐 Esfera Vectorial — Similitud Coseno 3D",
        "cosine_sphere_info": (
            "Cada palabra es un <b>vector</b> — una flecha en el espacio.<br><br>"
            "🟢 = <b>{w1}</b> &nbsp;|&nbsp; 🔴 = <b>{w2}</b> &nbsp;|&nbsp; 🔵 = otras palabras<br><br>"
            "El <b>arco dorado</b> muestra el <b>ángulo θ</b> entre los dos vectores.<br>"
            "• θ = 0° → cos = <b>1.0</b> → palabras <b>idénticas</b><br>"
            "• θ = 90° → cos = <b>0.0</b> → palabras <b>ortogonales</b><br>"
            "• θ = 180° → cos = <b>-1.0</b> → palabras <b>opuestas</b>"
        ),
        "dataset_tip_semantic": (
            "💡 <b>¿Por qué se agrupan algunas palabras?</b> El modelo aprende que palabras "
            "en contextos similares tienen vectores similares."
        ),
        "log_init": "[INFO] Inicializando modelo...",
        "log_tokenize": "[INFO] Tokenizando dataset...",
        "log_vocab": "[INFO] Vocabulario: {n} palabras",
        "log_pairs": "[INFO] Pares de entrenamiento: {n}",
        "log_start": "[INFO] Iniciando entrenamiento por {steps} pasos...",
        "log_step": "[TRAIN] Paso {step}/{total}: Loss = {loss:.4f}",
        "log_weights": "[TRAIN] Actualizando pesos... |∇W_h|={wh:.3f} |∇W_out|={wo:.3f}",
        "log_done": "[INFO] ¡Entrenamiento completado! Loss final: {loss:.4f}",
        "log_infer": "[INFER] Token de entrada: '{token}' (idx={idx})",
        "log_predict": "[INFER] Mejor predicción: '{word}' (prob={prob:.3f})",
    },

    # ================================================================
    # 中文（简体）
    # ================================================================
    "zh": {
        "sidebar_lang": "语言 / Language",
        "sidebar_brain_size": "大脑大小（隐藏神经元数）",
        "sidebar_steps": "训练步数",
        "sidebar_hyperparams": "⚙️ 超参数",
        "sidebar_about": "ℹ️ 关于",
        "sidebar_about_text": (
            "**LLM Visual Lab** 是一个开源教育实验室，"
            "以可视化、交互式的方式展示语言模型的工作原理。"
        ),
        "sidebar_github": "⭐ GitHub 仓库",
        "sidebar_credits_title": "🛠️ 构建工具",
        "sidebar_credits_text": (
            "- [Streamlit](https://streamlit.io) — Web 应用框架\n"
            "- [NumPy](https://numpy.org) — 神经网络计算\n"
            "- [Plotly](https://plotly.com) — 交互式图表\n"
            "- [Claude Code](https://claude.ai/claude-code) — AI 辅助开发"
        ),
        "app_title": "LLM Visual Lab",
        "app_subtitle": "探索人工智能的工作原理",
        "intro_llm_title": "什么是 LLM？",
        "intro_llm_text": (
            "<b>大型语言模型（LLM）</b>是一种通过阅读海量文本学会理解和生成文字的人工智能系统。"
            "在这个实验室中，您将以小规模方式观察 ChatGPT、Gemini 和 Claude 所使用的相同机制。"
        ),
        "intro_training_title": "什么是训练？",
        "intro_training_text": (
            "<b>训练</b>是模型学习的阶段。"
            "我们向模型展示许多句子，它尝试猜测下一个词。"
            "每次出错，它都会调整内部「权重」以减少错误。"
        ),
        "intro_inference_title": "什么是推理？",
        "intro_inference_text": (
            "训练完成后，我们可以<b>使用</b>模型："
            "给它一个句子的开头，它会建议如何续写。这称为<b>推理</b>。"
        ),
        "intro_embedding_title": "什么是嵌入？",
        "intro_embedding_text": (
            "每个词被转换为一个<b>数字向量</b>（嵌入）。"
            "含义相近的词在向量空间中靠得更近。"
            "<b>余弦相似度</b>衡量两个向量&ldquo;指向同一方向&rdquo;的程度：越接近1，词越相似。"
        ),
        "sec1_title": "📝 第1节 — 数据集与分词",
        "sec1_intro": "输入您想用于训练模型的句子。每行一句。（最多100句）",
        "token_reality_note": (
            "⚠️ <b>关于Token的教学说明：</b>在真实模型（GPT、Claude等）中，"
            "一个token<b>不是完整的词</b>，而是<b>词的片段</b>（子词）。"
            "本实验室使用<b>一词 = 一token</b>来简化教学。"
        ),
        "sec1_input_label": "训练句子（每行一句）：",
        "sec1_default_sentences": (
            "il gatto mangia il pesce\n"
            "il cane mangia la carne\n"
            "il gatto beve il latte\n"
            "il cane beve l acqua\n"
            "the cat eats fish\n"
            "the dog eats meat\n"
            "il sole splende forte\n"
            "la luna brilla alta\n"
            "il vento soffia freddo"
        ),
        "sec1_btn_tokenize": "🔍 分析数据集",
        "sec1_vocab_title": "📚 构建的词汇表",
        "sec1_vocab_info": "每个唯一的词成为一个<b>token</b>，具有唯一的索引。",
        "sec1_pairs_title": "🔗 训练对（输入 → 目标）",
        "sec1_pairs_info": "模型学习预测下一个词。这些是训练对。",
        "sec1_tokenization_title": "🔬 每句分词结果",
        "sec2_title": "⚡ 第2节 — 交互式训练",
        "sec2_intro": "点击开始训练。3D神经网络将实时更新权重和激活值。",
        "sec2_btn_train": "🚀 开始训练",
        "sec2_btn_reset": "🔄 重置",
        "sec2_training_label": "训练中...",
        "sec2_net_title": "🕸️ 实时3D神经网络",
        "sec2_net_info": (
            "<b>节点颜色：</b>🔵 负激活 → ⚪ 零 → 🔴 正激活。"
            "<b>绿色节点：</b>输入token | <b>黄色★节点：</b>目标token。"
        ),
        "sec2_loss_title": "📉 损失曲线",
        "sec2_loss_info": "<b>损失</b>衡量模型的错误程度。应随时间下降。",
        "sec2_log_title": "🖥️ 训练控制台",
        "sec2_done": "✅ 训练完成！",
        "sec2_arch_title": "🏗️ 模型架构",
        "sec2_warn_first": "⚠️ 请先完成第1节。",
        "sec3_title": "🔮 第3节 — 推理与语义搜索",
        "sec3_info_train_first": "⚠️ 请先完成第2节的训练。",
        "sec3_intro": "输入一个词或部分句子。模型将建议如何继续。",
        "sec3_query_label": "推理文本：",
        "sec3_query_placeholder": "例如 il gatto",
        "sec3_btn_infer": "🔮 运行推理",
        "sec3_next_token_title": "🎯 预测的下一个Token",
        "sec3_next_token_info": "模型认为最可能作为延续的token。",
        "sec3_emb_title": "🌐 嵌入图",
        "sec3_emb_info": "每个词是向量空间中的一个点。语义相似的词<b>靠得更近</b>。",
        "sec3_cos_title": "📐 余弦相似度",
        "sec3_cos_info": "选择两个词查看它们的语义接近程度。",
        "sec3_word1_label": "第一个词：",
        "sec3_word2_label": "第二个词：",
        "sec3_search_title": "🔍 语义搜索",
        "sec3_search_info": "模型将您的查询与所有句子进行比较，返回最相似的结果。",
        "sec3_search_btn": "🔍 搜索",
        "sec3_heatmap_title": "🗺️ 相似度图",
        "sec3_heatmap_info": (
            "每个单元格显示两个句子之间的相似度。"
            "<b style='color:#e05252'>红色</b> = 非常相似 | "
            "<b style='color:#aaaaaa'>白色</b> = 不相关 | "
            "<b style='color:#5278e0'>蓝色</b> = 含义相反"
        ),
        "not_trained_msg": "请先完成训练再使用此部分。",
        "layer_input": "输入", "layer_emb": "嵌入", "layer_hid": "隐藏", "layer_out": "输出",
        "step_label": "步骤", "loss_label": "损失", "similarity_label": "余弦相似度",
        "emb_2d_title": "词嵌入（PCA 2D）", "emb_3d_title": "词嵌入（PCA 3D）",
        "heatmap_title": "句子间余弦相似度",
        "probability": "概率", "word": "词", "sentence": "句子",
        "similarity": "相似度", "rank": "排名",
        "net_legend_title": "🔑 神经网络图例",
        "net_legend_text": (
            "<b>如何阅读3D网络：</b><br><br>"
            "🟢 <b>绿色节点</b> = <b>输入</b>token<br>"
            "⭐ <b>黄色节点</b> = <b>目标</b>token<br>"
            "🔵 <b>蓝色节点</b> = <b>负</b>激活<br>"
            "🔴 <b>红色节点</b> = <b>正</b>激活<br>"
            "⚪ <b>灰色节点</b> = 激活接近<b>零</b>"
        ),
        "loss_deep_explain": (
            "📉 <b>什么是损失（Loss）？</b><br><br>"
            "损失衡量<b>模型的错误程度</b>。<br>"
            "• <b>高损失</b> → 模型错误很多<br>"
            "• <b>低损失</b> → 模型学得很好<br>"
            "图表应显示一条<b>下降曲线</b>。"
        ),
        "weights_explain": (
            "⚖️ <b>什么是权重？</b><br><br>"
            "权重是神经元之间连接中的<b>数字</b>。"
            "它们编码了模型学到的一切。"
        ),
        "cosine_sphere_title": "🌐 向量球 — 3D余弦相似度",
        "cosine_sphere_info": (
            "每个词是一个<b>向量</b>——空间中的一个箭头。<br><br>"
            "🟢 = <b>{w1}</b> &nbsp;|&nbsp; 🔴 = <b>{w2}</b> &nbsp;|&nbsp; 🔵 = 其他词<br><br>"
            "<b>金色弧线</b>显示两个向量之间的<b>角度θ</b>。<br>"
            "• θ = 0° → cos = <b>1.0</b> → <b>相同</b>的词<br>"
            "• θ = 90° → cos = <b>0.0</b> → <b>不相关</b>的词<br>"
            "• θ = 180° → cos = <b>-1.0</b> → <b>相反</b>的词"
        ),
        "dataset_tip_semantic": (
            "💡 <b>为什么某些词会聚集在一起？</b>模型学到，出现在相似语境中的词具有相似的向量。"
        ),
        "log_init": "[INFO] 初始化模型...",
        "log_tokenize": "[INFO] 数据集分词中...",
        "log_vocab": "[INFO] 词汇表：{n} 个词",
        "log_pairs": "[INFO] 训练对：{n}",
        "log_start": "[INFO] 开始训练 {steps} 步...",
        "log_step": "[TRAIN] 步骤 {step}/{total}：损失 = {loss:.4f}",
        "log_weights": "[TRAIN] 更新权重... |∇W_h|={wh:.3f} |∇W_out|={wo:.3f}",
        "log_done": "[INFO] 训练完成！最终损失：{loss:.4f}",
        "log_infer": "[INFER] 输入token：'{token}' (idx={idx})",
        "log_predict": "[INFER] 最佳预测：'{word}' (prob={prob:.3f})",
    },

    # ================================================================
    # РУССКИЙ
    # ================================================================
    "ru": {
        "sidebar_lang": "Язык / Language",
        "sidebar_brain_size": "Размер мозга (скрытые нейроны)",
        "sidebar_steps": "Шаги обучения",
        "sidebar_hyperparams": "⚙️ Гиперпараметры",
        "sidebar_about": "ℹ️ О проекте",
        "sidebar_about_text": (
            "**LLM Visual Lab** — образовательная лаборатория с открытым исходным кодом, "
            "которая наглядно и интерактивно показывает, как работает языковая модель."
        ),
        "sidebar_github": "⭐ Репозиторий GitHub",
        "sidebar_credits_title": "🛠️ Создано с помощью",
        "sidebar_credits_text": (
            "- [Streamlit](https://streamlit.io) — фреймворк для веб-приложений\n"
            "- [NumPy](https://numpy.org) — вычисления нейронной сети\n"
            "- [Plotly](https://plotly.com) — интерактивные графики\n"
            "- [Claude Code](https://claude.ai/claude-code) — разработка с помощью ИИ"
        ),
        "app_title": "LLM Visual Lab",
        "app_subtitle": "Откройте для себя, как работает Искусственный Интеллект",
        "intro_llm_title": "Что такое LLM?",
        "intro_llm_text": (
            "<b>Большая Языковая Модель (LLM)</b> — это система ИИ, которая научилась "
            "понимать и генерировать текст, читая огромные объёмы данных. "
            "В этой лаборатории вы увидите — в малом масштабе — те же механизмы, "
            "что используются в ChatGPT, Gemini и Claude."
        ),
        "intro_training_title": "Что такое Обучение?",
        "intro_training_text": (
            "<b>Обучение</b> — это фаза, в которой модель учится. "
            "Мы показываем ей предложения, и она пытается угадать следующее слово. "
            "Каждый раз, когда она ошибается, она корректирует свои \"внутренние веса\"."
        ),
        "intro_inference_title": "Что такое Инференс?",
        "intro_inference_text": (
            "После обучения мы можем <b>использовать</b> модель: "
            "даём ей начало предложения, и она предлагает продолжение. "
            "Это называется <b>инференсом</b>."
        ),
        "intro_embedding_title": "Что такое Эмбеддинги?",
        "intro_embedding_text": (
            "Каждое слово преобразуется в <b>вектор чисел</b> (эмбеддинг). "
            "Слова со схожим значением оказываются близко в векторном пространстве. "
            "<b>Косинусное сходство</b> измеряет, насколько два вектора \"указывают в одну сторону\"."
        ),
        "sec1_title": "📝 Раздел 1 — Датасет и Токенизация",
        "sec1_intro": (
            "Введите предложения для обучения модели. "
            "По одному предложению на строку. (макс. 100 предложений)"
        ),
        "token_reality_note": (
            "⚠️ <b>Учебная заметка о токенах:</b> В реальных моделях (GPT, Claude и др.) "
            "токен — это <b>не целое слово</b>, а <b>фрагмент слова</b> (подслово). "
            "В этой лаборатории мы используем <b>одно слово = один токен</b> для упрощения."
        ),
        "sec1_input_label": "Обучающие предложения (по одному на строку):",
        "sec1_default_sentences": (
            "il gatto mangia il pesce\n"
            "il cane mangia la carne\n"
            "il gatto beve il latte\n"
            "il cane beve l acqua\n"
            "the cat eats fish\n"
            "the dog eats meat\n"
            "il sole splende forte\n"
            "la luna brilla alta\n"
            "il vento soffia freddo"
        ),
        "sec1_btn_tokenize": "🔍 Анализировать датасет",
        "sec1_vocab_title": "📚 Построенный словарь",
        "sec1_vocab_info": "Каждое уникальное слово становится <b>токеном</b> с уникальным индексом.",
        "sec1_pairs_title": "🔗 Обучающие пары (вход → цель)",
        "sec1_pairs_info": "Модель учится предсказывать следующее слово.",
        "sec1_tokenization_title": "🔬 Токенизация по предложениям",
        "sec2_title": "⚡ Раздел 2 — Интерактивное Обучение",
        "sec2_intro": "Нажмите для запуска обучения. 3D-сеть обновляется в реальном времени.",
        "sec2_btn_train": "🚀 Начать обучение",
        "sec2_btn_reset": "🔄 Сброс",
        "sec2_training_label": "Обучение...",
        "sec2_net_title": "🕸️ 3D Нейронная сеть в реальном времени",
        "sec2_net_info": (
            "<b>Цвета узлов:</b> 🔵 отрицательная активация → ⚪ ноль → 🔴 положительная активация. "
            "<b>Зелёный узел:</b> входной токен | <b>Жёлтый ★ узел:</b> целевой токен."
        ),
        "sec2_loss_title": "📉 Кривая потерь",
        "sec2_loss_info": "<b>Потери</b> измеряют ошибку модели. Должны снижаться.",
        "sec2_log_title": "🖥️ Консоль обучения",
        "sec2_done": "✅ Обучение завершено!",
        "sec2_arch_title": "🏗️ Архитектура модели",
        "sec2_warn_first": "⚠️ Сначала завершите Раздел 1.",
        "sec3_title": "🔮 Раздел 3 — Инференс и Семантический Поиск",
        "sec3_info_train_first": "⚠️ Сначала завершите обучение в Разделе 2.",
        "sec3_intro": "Введите слово или часть предложения. Модель предложит продолжение.",
        "sec3_query_label": "Текст для инференса:",
        "sec3_query_placeholder": "напр. il gatto",
        "sec3_btn_infer": "🔮 Запустить инференс",
        "sec3_next_token_title": "🎯 Предсказанные следующие токены",
        "sec3_next_token_info": "Токены, которые модель считает наиболее вероятным продолжением.",
        "sec3_emb_title": "🌐 Карта эмбеддингов",
        "sec3_emb_info": "Каждое слово — точка в векторном пространстве. Похожие слова <b>ближе</b>.",
        "sec3_cos_title": "📐 Косинусное сходство",
        "sec3_cos_info": "Выберите два слова, чтобы увидеть их семантическую близость.",
        "sec3_word1_label": "Первое слово:",
        "sec3_word2_label": "Второе слово:",
        "sec3_search_title": "🔍 Семантический поиск",
        "sec3_search_info": "Модель сравнивает запрос со всеми предложениями и возвращает наиболее похожие.",
        "sec3_search_btn": "🔍 Поиск",
        "sec3_heatmap_title": "🗺️ Карта сходства",
        "sec3_heatmap_info": (
            "Каждая ячейка показывает сходство между двумя предложениями. "
            "<b style='color:#e05252'>Красный</b> = очень похожи | "
            "<b style='color:#aaaaaa'>Белый</b> = не связаны | "
            "<b style='color:#5278e0'>Синий</b> = противоположное значение"
        ),
        "not_trained_msg": "Завершите обучение перед использованием этого раздела.",
        "layer_input": "Вход", "layer_emb": "Эмбеддинг", "layer_hid": "Скрытый", "layer_out": "Выход",
        "step_label": "Шаг", "loss_label": "Потери", "similarity_label": "Косинусное сходство",
        "emb_2d_title": "Эмбеддинги слов (PCA 2D)", "emb_3d_title": "Эмбеддинги слов (PCA 3D)",
        "heatmap_title": "Косинусное сходство между предложениями",
        "probability": "Вероятность", "word": "Слово", "sentence": "Предложение",
        "similarity": "Сходство", "rank": "Ранг",
        "net_legend_title": "🔑 Легенда нейронной сети",
        "net_legend_text": (
            "<b>Как читать 3D-сеть:</b><br><br>"
            "🟢 <b>Зелёный узел</b> = входной токен<br>"
            "⭐ <b>Жёлтый узел</b> = целевой токен<br>"
            "🔵 <b>Синий узел</b> = <b>отрицательная</b> активация<br>"
            "🔴 <b>Красный узел</b> = <b>положительная</b> активация<br>"
            "⚪ <b>Серый узел</b> = активация около <b>нуля</b>"
        ),
        "loss_deep_explain": (
            "📉 <b>Что такое ПОТЕРИ (Loss)?</b><br><br>"
            "Потери измеряют <b>насколько ошибается модель</b>.<br>"
            "• <b>Высокие потери</b> → модель сильно ошибается<br>"
            "• <b>Низкие потери</b> → модель хорошо научилась<br>"
            "График должен показывать <b>нисходящую кривую</b>."
        ),
        "weights_explain": (
            "⚖️ <b>Что такое ВЕСА?</b><br><br>"
            "Веса — это <b>числа</b> в связях между нейронами. "
            "Они кодируют всё, чему научилась модель."
        ),
        "cosine_sphere_title": "🌐 Векторная сфера — 3D косинусное сходство",
        "cosine_sphere_info": (
            "Каждое слово — это <b>вектор</b> — стрелка в пространстве.<br><br>"
            "🟢 = <b>{w1}</b> &nbsp;|&nbsp; 🔴 = <b>{w2}</b> &nbsp;|&nbsp; 🔵 = другие слова<br><br>"
            "<b>Золотая дуга</b> показывает <b>угол θ</b> между двумя векторами.<br>"
            "• θ = 0° → cos = <b>1.0</b> → <b>одинаковые</b> слова<br>"
            "• θ = 90° → cos = <b>0.0</b> → <b>несвязанные</b> слова<br>"
            "• θ = 180° → cos = <b>-1.0</b> → <b>противоположные</b> слова"
        ),
        "dataset_tip_semantic": (
            "💡 <b>Почему некоторые слова группируются?</b> Модель учится, что слова "
            "в схожих контекстах имеют похожие векторы."
        ),
        "log_init": "[INFO] Инициализация модели...",
        "log_tokenize": "[INFO] Токенизация датасета...",
        "log_vocab": "[INFO] Словарь: {n} слов",
        "log_pairs": "[INFO] Обучающих пар: {n}",
        "log_start": "[INFO] Запуск обучения на {steps} шагов...",
        "log_step": "[TRAIN] Шаг {step}/{total}: Потери = {loss:.4f}",
        "log_weights": "[TRAIN] Обновление весов... |∇W_h|={wh:.3f} |∇W_out|={wo:.3f}",
        "log_done": "[INFO] Обучение завершено! Финальные потери: {loss:.4f}",
        "log_infer": "[INFER] Входной токен: '{token}' (idx={idx})",
        "log_predict": "[INFER] Лучший прогноз: '{word}' (prob={prob:.3f})",
    },
}


def t(key: str, lang: str = "en", **kwargs) -> str:
    """Lookup a translation key with optional format arguments.
    Falls back to English if the key is missing in the requested language.
    """
    lang_dict = TRANSLATIONS.get(lang, TRANSLATIONS["en"])
    text = lang_dict.get(key) or TRANSLATIONS["en"].get(key, key)
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, IndexError):
            pass
    return text
