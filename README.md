# Roman Urdu Sentiment Classifier — RNN vs LSTM vs GRU

A sentiment analysis project comparing vanilla RNN, LSTM, and GRU on Roman Urdu text — with an interactive Streamlit demo that shows all three models predicting side by side.



## Why this project

Roman Urdu (Urdu written in Latin script, e.g. *"yeh movie bohat acha tha"*) is how most Pakistanis actually text and comment online — but it has almost no standard spelling, making it a genuinely hard NLP problem. This project doesn't just train one model; it trains three identical architectures that differ only in their recurrent cell, to concretely demonstrate the vanishing gradient problem that motivated LSTM/GRU in the first place.

## What it does

Type any Roman Urdu sentence and get a live sentiment prediction (Positive / Negative / Neutral) from three models at once:

| Model | Parameters | What changes |
|---|---|---|
| Vanilla RNN | 1,030,027 | Baseline — simple recurrence, no gating |
| LSTM | 1,118,347 | Input/forget/output gates + cell state |
| GRU | 1,088,907 | Update/reset gates — lighter than LSTM |

## The key result

On sentences that flip tone midway (e.g. *"shuru mein bohat boring tha lekin end mein bilkul zabardast nikla"* — starts negative, ends positive), the vanilla RNN consistently struggles to retain the early context, while LSTM and GRU handle it far more reliably. This is the vanishing gradient problem, observed directly rather than just described.

```
Input: "waqt zaya hua bilkul bakwas"
  RNN   -> Neutral  (77.4% confidence)   ✗ misses it entirely
  LSTM  -> Negative (100.0% confidence)  ✓
  GRU   -> Negative (100.0% confidence)  ✓
```

## Dataset

[Roman Urdu Dataset](https://github.com/Smat26/Roman-Urdu-Dataset) — ~20,000 manually tagged sentences (Positive/Negative/Neutral), compiled from e-commerce reviews, social media comments, and other sources, originally hosted on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Roman+Urdu+Data+Set).

## Project structure

```
├── notebook/
│   └── roman_urdu_rnn_lstm_gru.ipynb   # data prep, training, evaluation (run on Colab, free T4 GPU)
├── app.py                              # Streamlit demo app
├── models/
│   ├── rnn_sentiment.pt
│   ├── lstm_sentiment.pt
│   └── gru_sentiment.pt
├── word2idx.pkl                        # vocabulary used at training time
├── requirements.txt
└── README.md
```

## How it works

1. **Preprocessing** — lowercase, strip punctuation/digits, build a 10,000-word vocabulary from the training data
2. **Encoding** — each sentence becomes a fixed-length (25-token) sequence of word IDs
3. **Model** — embedding layer → recurrent layer (RNN/LSTM/GRU) → linear classifier head, identical across all three except the recurrent cell
4. **Training** — 25 epochs, Adam optimizer, class-weighted cross-entropy loss (to offset the dataset's Neutral-heavy imbalance)

## Results

| Model | Test accuracy |
|---|---|
| RNN | ~51.5% |
| LSTM | ~61.0% |
| GRU | ~61.0% |

*(Trained on a free Colab T4 GPU in under a minute — see the notebook for full per-epoch logs.)*

## Run it yourself

**Train from scratch:**
Open `notebook/roman_urdu_rnn_lstm_gru.ipynb` in Google Colab, set runtime to T4 GPU, run all cells.

**Run the demo locally:**
```bash
git clone https://github.com/<shahnabi110>/roman-urdu-sentiment.git
cd roman-urdu-sentiment
pip install -r requirements.txt
streamlit run app.py
```

## Tech stack

Python · PyTorch · Streamlit · pandas · scikit-learn

## What I learned

Building this made the vanishing gradient problem concrete rather than theoretical — watching the same architecture succeed or fail on the exact same sentence, with the only difference being the recurrent cell, made LSTM/GRU's gating mechanisms click in a way that reading about them never did.


