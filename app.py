import streamlit as st
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ============================
# Page Config
# ============================
st.set_page_config(
    page_title="🌍 Multilingual Emotion Analyzer",
    page_icon="🌍",
    layout="centered"
)

# ============================
# Custom Dark UI + Animations
# ============================
st.markdown("""
<style>
html, body, [class*="css"]  {
    background-color: #0f172a;
    color: #e5e7eb;
    font-family: 'Inter', system-ui, sans-serif;
}

/* Title & Subtitle */
.title { text-align: center; font-size: 2.4rem; font-weight: 900; margin-bottom: 0.2rem; animation: fadeIn 1s ease-in-out; }
.subtitle { text-align: center; color: #94a3b8; margin-bottom: 1.5rem; animation: fadeIn 1.2s ease-in-out; }

/* Card */
.card { 
    background: linear-gradient(145deg, #020617, #0c122b); 
    border-radius: 20px; 
    padding: 1.5rem; 
    margin-top: 1rem; 
    box-shadow: 0 15px 35px rgba(0,0,0,0.45); 
    transition: transform 0.3s ease; 
}
.card:hover { transform: translateY(-5px); }

/* Emotion Row */
.emotion { 
    display: flex; 
    justify-content: space-between; 
    font-weight: 600; 
    margin-bottom: 0.6rem; 
    padding: 0.2rem 0.5rem;
    border-radius: 8px;
    transition: background 0.3s ease;
}
.emotion:hover { background: rgba(56, 189, 248, 0.1); }

/* Progress Bar */
.progress { height: 12px; border-radius: 20px; background: #1e293b; overflow: hidden; margin-bottom: 0.8rem; }
.progress span { display: block; height: 100%; background: linear-gradient(90deg, #38bdf8, #22c55e); animation: grow 1.5s ease forwards; }

/* Primary Emotion */
.primary { font-size: 1.2rem; font-weight: 700; text-align: center; color: #22c55e; text-shadow: 0 0 15px #38bdf8; margin-top: 1rem; }

/* Animations */
@keyframes grow { from { width: 0%; } }
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
</style>
""", unsafe_allow_html=True)

# ============================
# Title
# ============================
st.markdown('<div class="title">🌍 Multilingual Emotion Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by Transformers • Supports English + Indian Languages</div>', unsafe_allow_html=True)

# ============================
# Load Model
# ============================
@st.cache_resource(show_spinner=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("AnasAlokla/multilingual_go_emotions_V1.2")
    model = AutoModelForSequenceClassification.from_pretrained("AnasAlokla/multilingual_go_emotions_V1.2")
    return tokenizer, model

tokenizer, model = load_model()

# ============================
# Analyze Emotion
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
# Text Input
# ============================
text = st.text_area(
    "Paste a comment (any language)",
    height=160,
    placeholder="Example: I feel proud and happy today!"
)

if st.button("✨ Analyze Emotion"):
    if not text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        try:
            lang = detect(text)
        except:
            lang = "unknown"
        st.markdown(f"**Detected Language:** `{lang.upper()}`")

        with st.spinner("🧠 Analyzing emotions..."):
            emotions = analyze_emotion(text)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🎭 Top Emotions")
        for label, score in emotions:
            percent = round(score * 100, 2)
            st.markdown(
                f"""
                <div class="emotion">
                    <span>{label}</span>
                    <span>{percent}%</span>
                </div>
                <div class="progress">
                    <span style="width:{percent}%"></span>
                </div>
                """,
                unsafe_allow_html=True
            )

        primary_label, primary_score = emotions[0]
        st.markdown(f'<div class="primary">✨ Primary Emotion: {primary_label} ({round(primary_score*100,2)}%)</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================
# Footer
# ============================
st.markdown(
    "<p style='text-align:center; color:#64748b; margin-top:2rem;'>Built by Akash • Streamlit + Transformers 🤍</p>",
    unsafe_allow_html=True
)
