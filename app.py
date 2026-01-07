import streamlit as st
from langdetect import detect
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
# Custom Dark UI + Animations
# ============================
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #0f172a;
    color: #e5e7eb;
    font-family: 'Inter', system-ui, sans-serif;
    scroll-behavior: smooth;
}

/* Hero Title */
.title {
    text-align: center;
    font-size: 3rem;
    font-weight: 900;
    margin-bottom: 0.2rem;
    background: linear-gradient(90deg, #38bdf8, #22c55e, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientMove 5s ease infinite;
}
@keyframes gradientMove {
  0% { background-position: 0% }
  50% { background-position: 100% }
  100% { background-position: 0% }
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 1.5rem;
    animation: fadeIn 1.2s ease-in-out;
}

/* Particle Background */
#particles-js {
    position: fixed;
    width: 100%;
    height: 100%;
    z-index: -1;
}

/* Card */
.card { 
    background: linear-gradient(145deg, #020617, #0c122b); 
    border-radius: 20px; 
    padding: 1.5rem; 
    margin: 1rem auto; 
    width: 90%;
    max-width: 700px;
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
.primary { font-size: 1.4rem; font-weight: 700; text-align: center; text-shadow: 0 0 15px #38bdf8; margin-top: 1rem; animation: pulse 2s infinite; }

/* Animations */
@keyframes grow { from { width: 0%; } }
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
@keyframes pulse { 0% { text-shadow: 0 0 10px #22c55e; } 50% { text-shadow: 0 0 25px #38bdf8; } 100% { text-shadow: 0 0 10px #22c55e; } }

/* Responsive */
@media (max-width: 600px) {
    .title { font-size: 2rem; }
    .subtitle { font-size: 1rem; }
    .primary { font-size: 1.1rem; }
}
</style>

<div id="particles-js"></div>
<script src="https://cdn.jsdelivr.net/npm/particles.js"></script>
<script>
particlesJS("particles-js", {
  "particles": {
    "number": {"value": 60},
    "size": {"value": 3},
    "move": {"speed": 2},
    "line_linked": {"enable": true, "distance": 150}
  }
});
</script>
""", unsafe_allow_html=True)

# ============================
# Hero Section
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
# Persistent Input
# ============================
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

text = st.text_area(
    "Paste a comment (any language)",
    height=160,
    value=st.session_state.user_input,
    placeholder="Type your text here..."
)
st.session_state.user_input = text

# ============================
# Button & Analysis
# ============================
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

        # ============================
        # Emotion Card
        # ============================
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🎭 Top Emotions")

        emotion_colors = {
            "joy": "#facc15", "anger": "#ef4444", "sadness": "#3b82f6",
            "fear": "#8b5cf6", "love": "#ec4899", "neutral": "#94a3b8"
        }
        emoji_map = {"joy":"😊","anger":"😡","sadness":"😢","fear":"😨","love":"❤️","neutral":"😐"}

        # Progress bars + labels
        for label, score in emotions:
            percent = round(score * 100, 2)
            color = emotion_colors.get(label.lower(), "#22c55e")
            st.markdown(
                f"""
                <div class="emotion">
                    <span>{emoji_map.get(label.lower(),'✨')} {label}</span>
                    <span>{percent:.2f}%</span>
                </div>
                <div class="progress">
                    <span style="width:{percent}%; background:{color};"></span>
                </div>
                """,
                unsafe_allow_html=True
            )

        primary_label, primary_score = emotions[0]
        st.markdown(f'<div class="primary">✨ Primary Emotion: {emoji_map.get(primary_label.lower(),"✨")} {primary_label} ({primary_score*100:.2f}%)</div>', unsafe_allow_html=True)

        # ============================
        # Line Chart
        # ============================
        df = pd.DataFrame(emotions, columns=["Emotion","Score"])
        fig = px.line(
            df, 
            x="Emotion", 
            y="Score", 
            markers=True, 
            line_shape="spline",
            color="Emotion", 
            color_discrete_map=emotion_colors
        )
        fig.update_layout(
            height=300,
            showlegend=False,
            plot_bgcolor="#0f172a",
            paper_bgcolor="#0f172a",
            font_color="#e5e7eb",
            xaxis_title="Emotion",
            yaxis_title="Probability",
            yaxis=dict(range=[0,1])
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================
# Footer
# ============================
st.markdown(
    "<p style='text-align:center; color:#64748b; margin-top:2rem;'>Built by Akash • Streamlit + Transformers 🤍 | <a href='https://github.com/akashm-1322' target='_blank'>GitHub</a></p>",
    unsafe_allow_html=True
)
