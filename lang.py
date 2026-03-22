"""
Translations dictionary.
Default language: Italiano.
All UI strings live here; code identifiers remain in English.
"""

TRANSLATIONS = {
    # ================================================================
    # ITALIANO (default)
    # ================================================================
    "it": {
        # ---- Sidebar ----
        "sidebar_lang": "Lingua / Language",
        "sidebar_brain_size": "Dimensione cervello (neuroni hidden)",
        "sidebar_steps": "Step di training",
        "sidebar_about": "ℹ️ Informazioni",
        "sidebar_about_text": (
            "**LLM Visual Lab** è un laboratorio educativo open-source "
            "che mostra in modo visuale e interattivo come funziona "
            "un modello linguistico."
        ),

        # ---- Header ----
        "app_title": "🧠 LLM Visual Lab",
        "app_subtitle": "Scopri come funziona una Intelligenza Artificiale",
        "intro_llm_title": "Cos'è un LLM?",
        "intro_llm_text": (
            "Un **Large Language Model (LLM)** è un sistema di Intelligenza Artificiale "
            "che ha imparato a capire e generare testo leggendo enormi quantità di testi. "
            "In questo laboratorio potrai osservare – su scala ridotta – gli stessi "
            "meccanismi usati da ChatGPT, Gemini e Claude."
        ),
        "intro_training_title": "Cos'è il Training?",
        "intro_training_text": (
            "Il **training** è la fase in cui il modello impara. "
            "Mostriamo al modello molte frasi e lui cerca di indovinare "
            "quale parola viene dopo. Ogni volta che sbaglia, aggiusta i suoi "
            "\"pesi interni\" (i numeri nella rete neurale) per sbagliare meno "
            "la volta successiva."
        ),
        "intro_inference_title": "Cos'è l'Inferenza?",
        "intro_inference_text": (
            "Quando il training è finito, possiamo **usare** il modello: "
            "gli diamo l'inizio di una frase e lui suggerisce come continuarla. "
            "Questo si chiama **inferenza**."
        ),
        "intro_embedding_title": "Cosa sono gli Embedding?",
        "intro_embedding_text": (
            "Ogni parola viene trasformata in un **vettore di numeri** (embedding). "
            "Parole con significato simile finiscono vicine nello spazio vettoriale. "
            "La **similarità coseno** misura quanto due vettori \"puntano "
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
            "Ogni parola unica nel testo diventa un **token**. "
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
            "**Colori dei nodi:** 🔵 attivazione negativa → ⚪ zero → 🔴 attivazione positiva. "
            "**Nodo verde:** token in input | **Nodo giallo ★:** token target. "
            "I bordi illuminati mostrano il percorso attivo nella rete."
        ),
        "sec2_loss_title": "📉 Curva di Loss",
        "sec2_loss_info": (
            "La **loss** misura quanto il modello sbaglia. "
            "Deve scendere nel tempo: significa che il modello sta imparando!"
        ),
        "sec2_log_title": "🖥️ Console di Training",
        "sec2_done": "✅ Training completato!",
        "sec2_arch_title": "🏗️ Architettura del modello",

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
            "Le parole semanticamente simili tendono a essere **vicine** tra loro."
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
            "Ogni cella mostra quanto sono simili due parole secondo il modello. "
            "Rosso = molto simili, Blu = molto diverse."
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
        "heatmap_title": "Similarità Coseno tra Parole",
        "probability": "Probabilità",
        "word": "Parola",
        "sentence": "Frase",
        "similarity": "Similarità",
        "rank": "Posizione",

        # ---- Detailed explanations (new) ----
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
    # ENGLISH
    # ================================================================
    "en": {
        # ---- Sidebar ----
        "sidebar_lang": "Lingua / Language",
        "sidebar_brain_size": "Brain size (hidden neurons)",
        "sidebar_steps": "Training steps",
        "sidebar_about": "ℹ️ About",
        "sidebar_about_text": (
            "**LLM Visual Lab** is an open-source educational lab "
            "that visually and interactively shows how a language model works."
        ),

        # ---- Header ----
        "app_title": "🧠 LLM Visual Lab",
        "app_subtitle": "Explore how Artificial Intelligence works",
        "intro_llm_title": "What is an LLM?",
        "intro_llm_text": (
            "A **Large Language Model (LLM)** is an AI system that has learned "
            "to understand and generate text by reading enormous amounts of text. "
            "In this lab you'll observe – on a small scale – the same mechanisms "
            "used by ChatGPT, Gemini and Claude."
        ),
        "intro_training_title": "What is Training?",
        "intro_training_text": (
            "**Training** is the phase where the model learns. "
            "We show the model many sentences and it tries to guess which word "
            "comes next. Each time it fails, it adjusts its \"internal weights\" "
            "(the numbers in the neural network) to make fewer mistakes next time."
        ),
        "intro_inference_title": "What is Inference?",
        "intro_inference_text": (
            "Once training is done, we can **use** the model: "
            "we give it the beginning of a sentence and it suggests how to continue it. "
            "This is called **inference**."
        ),
        "intro_embedding_title": "What are Embeddings?",
        "intro_embedding_text": (
            "Each word is transformed into a **vector of numbers** (embedding). "
            "Words with similar meanings end up close together in vector space. "
            "**Cosine similarity** measures how much two vectors \"point in the same "
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
            "Every unique word in the text becomes a **token**. "
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
            "**Node colours:** 🔵 negative activation → ⚪ zero → 🔴 positive activation. "
            "**Green node:** input token | **Yellow ★ node:** target token. "
            "Bright edges show the active path through the network."
        ),
        "sec2_loss_title": "📉 Loss Curve",
        "sec2_loss_info": (
            "**Loss** measures how much the model is mistaken. "
            "It should decrease over time: that means the model is learning!"
        ),
        "sec2_log_title": "🖥️ Training Console",
        "sec2_done": "✅ Training complete!",
        "sec2_arch_title": "🏗️ Model architecture",

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
            "Semantically similar words tend to be **close** to each other."
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
            "The model finds the dataset sentence most semantically "
            "similar to your query."
        ),
        "sec3_search_btn": "🔍 Search",
        "sec3_heatmap_title": "🗺️ Similarity Map",
        "sec3_heatmap_info": (
            "Each cell shows how similar two words are according to the model. "
            "Red = very similar, Blue = very different."
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
        "heatmap_title": "Cosine Similarity between Words",
        "probability": "Probability",
        "word": "Word",
        "sentence": "Sentence",
        "similarity": "Similarity",
        "rank": "Rank",

        # ---- Detailed explanations (new) ----
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
}


def t(key: str, lang: str = "it", **kwargs) -> str:
    """Lookup a translation key with optional format arguments."""
    text = TRANSLATIONS.get(lang, TRANSLATIONS["it"]).get(key, key)
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, IndexError):
            pass
    return text
