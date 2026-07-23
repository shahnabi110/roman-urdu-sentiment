import streamlit as st
import torch
import torch.nn as nn
import pickle
import re

# ── Model definition (same as training) ──
EMBED_DIM = 100
HIDDEN_DIM = 128
NUM_CLASSES = 3
MAX_LEN = 25

class SentimentModel(nn.Module):
    def __init__(self, vocab_size, cell_type="RNN"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        rnn_cls = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[cell_type]
        self.rnn = rnn_cls(EMBED_DIM, HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM, NUM_CLASSES)
        self.cell_type = cell_type

    def forward(self, x):
        emb = self.embedding(x)
        if self.cell_type == "LSTM":
            _, (hidden, _) = self.rnn(emb)
        else:
            _, hidden = self.rnn(emb)
        return self.fc(hidden[-1])

# ── Load vocab + models (cached so it only loads once) ──
@st.cache_resource
def load_everything():
    with open("word2idx.pkl", "rb") as f:
        word2idx = pickle.load(f)
    vocab_size = len(word2idx)

    models = {}
    for cell_type in ["RNN", "LSTM", "GRU"]:
        model = SentimentModel(vocab_size, cell_type)
        model.load_state_dict(torch.load(f"{cell_type.lower()}_sentiment.pt", map_location="cpu"))
        model.eval()
        models[cell_type] = model
    return word2idx, models

word2idx, models = load_everything()
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

def clean_text(s):
    s = str(s).lower()
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def encode(text):
    ids = [word2idx.get(w, word2idx.get("<unk>", 1)) for w in text.split()][:MAX_LEN]
    ids += [word2idx.get("<pad>", 0)] * (MAX_LEN - len(ids))
    return ids
# ── UI ──
st.set_page_config(page_title="Roman Urdu Sentiment", page_icon="💬", layout="centered")

st.markdown("""
<style>
.stTextArea textarea { font-size: 16px; border-radius: 10px; }
.result-card {
    padding: 1.2rem; border-radius: 12px; text-align: center;
    border: 1px solid rgba(255,255,255,0.1);
}
.result-label { font-size: 13px; opacity: 0.7; margin-bottom: 6px; }
.result-value { font-size: 22px; font-weight: 700; margin-bottom: 4px; }
.result-conf { font-size: 13px; opacity: 0.6; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Roman Urdu sentiment classifier")
st.caption("Vanilla RNN vs LSTM vs GRU — same sentence, three brains, watch them disagree")

text = st.text_area("Enter a Roman Urdu sentence", "yeh movie start mein bohat boring tha lekin end mein bilkul zabardast nikla", height=100)

colors = {"Positive": "#22c55e", "Negative": "#ef4444", "Neutral": "#94a3b8"}

if st.button("🔍 Analyze sentiment", type="primary", use_container_width=True):
    encoded = torch.tensor([encode(clean_text(text))], dtype=torch.long)
    cols = st.columns(3)
    for col, (cell_type, model) in zip(cols, models.items()):
        with torch.no_grad():
            probs = torch.softmax(model(encoded), dim=1)[0]
        pred = probs.argmax().item()
        label = label_map[pred]
        confidence = probs[pred].item() * 100
        color = colors[label]
        with col:
            st.markdown(f"""
            <div class="result-card" style="background: {color}15; border-color: {color}55;">
                <div class="result-label">{cell_type}</div>
                <div class="result-value" style="color: {color};">{label}</div>
                <div class="result-conf">{confidence:.1f}% confident</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.caption("💡 Vanilla RNN struggles with sentences that flip tone midway — it tends to forget earlier context. LSTM/GRU handle this better because of their gating mechanisms.")