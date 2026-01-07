import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import plotly.express as px
import pandas as pd

# ============================
# Page Config
# ============================
st.set_page_config(
    page_title="🌍 Multilingual Emotion Analyzer",
    page_icon="🌍",
    layout="wide"
)

# ============================
# Custom Dark UI
# ============================
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #0f172a;
    color: #e5e7eb;
    font-family: 'Inter', system-ui, sans-serif;
}

.title {
    text-align: center;
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(90deg, #38bdf8, #22c55e, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 1.5rem;
}

.card { 
    background: linear-gradient(145deg, #020617, #0c122b); 
    border-radius: 20px; 
    padding: 1.5rem; 
    margin: 1rem auto; 
    width: 90%;
    max-width: 700px;
    box-shadow: 0 15px 35px rgba(0,0,0,0.45); 
}

.emotion { 
    display: flex; 
    justify-content: space-between; 
    font-weight: 600; 
    margin-bottom: 0.4rem; 
}

.progress {
    height: 12px;
    border-radius: 20px;
    background: #1e293b;
    overflow: hidden;
    margin-bottom: 0.8rem;
}

.progress span {
    display: block;
    height: 100%;
    background: linear-gradient(90deg, #38bdf8, #22c55e);
}

.primary {
    font-size: 1.3rem;
    font-weight: 700;
    text-align: center;
    margin-top: 1rem;
}

@media (max-width: 600px) {
    .title { font-size: 2rem; }
}
</style>
""", unsafe_allow_html=True)

# ============================
# Hero
# ============================
st.markdown('<div class="title">🌍 Multilingual Emotion Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">User-selected Language • Stable Analysis</div>', unsafe_allow_html=True)

# ============================
# Load Model (cached)
# ============================
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("AnasAlokla/multilingual_go_emotions_V1.2")
    model = AutoModelForSequenceClassification.from_pretrained("AnasAlokla/multilingual_go_emotions_V1.2")
    return tokenizer, model

tokenizer, model = load_model()

# ============================
# Emotion Analyzer
# ============================
def analyze_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.sigmoid(logits)[0].numpy()
    labels = [model.config.id2label[i] for i in range(len(probs))]
    emotions = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)
    return emotions[:5]

# ============================
# Session State Init
# ============================
if "text" not in st.session_state:
    st.session_state.text = ""

if "language" not in st.session_state:
    st.session_state.language = "English"

# ============================
# Form (CRITICAL FIX)
# ============================
with st.form("emotion_form", clear_on_submit=False):

    language = st.selectbox(
        "Select Language of the Comment",
        [
            "English",
            "Tamil",
            "Telugu",
            "Kannada",
            "Hindi",
            "French",
            "Other"
        ],
        index=0
    )

    text = st.text_area(
        "Paste your comment",
        height=150,
        value=st.session_state.text,
        placeholder="Type here..."
    )

    submit = st.form_submit_button("✨ Analyze Emotion")

# ============================
# Persist Inputs
# ============================
st.session_state.text = text
st.session_state.language = language

# ============================
# Analysis
# ============================
if submit:

    if not text.strip():
        st.warning("Please enter some text to analyze.")
        st.stop()

    st.markdown(f"**Selected Language:** `{language}`")

    with st.spinner("🧠 Analyzing emotions..."):
        emotions = analyze_emotion(text)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🎭 Emotion Analysis")

    for label, score in emotions:
        percent = round(score * 100, 2)
        st.markdown(
            f"""
            <div class="emotion">
                <span>{label}</span>
                <span>{percent:.2f}%</span>
            </div>
            <div class="progress">
                <span style="width:{percent}%"></span>
            </div>
            """,
            unsafe_allow_html=True
        )

    primary_label, primary_score = emotions[0]
    st.markdown(
        f'<div class="primary">✨ Primary Emotion: {primary_label} ({primary_score*100:.2f}%)</div>',
        unsafe_allow_html=True
    )

    df = pd.DataFrame(emotions, columns=["Emotion", "Score"])
    fig = px.bar(df, x="Emotion", y="Score")
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ============================
# Footer
# ============================
st.markdown(
    "<p style='text-align:center;color:#64748b;margin-top:2rem;'>Built by Akash 🤍</p>",
    unsafe_allow_html=True
)
