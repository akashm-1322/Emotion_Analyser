import streamlit as st
import requests
import os
from langdetect import detect

# ============================
# Page Config
# ============================
st.set_page_config(
    page_title="Multilingual Emotion Analyzer",
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
.title {
    text-align: center;
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
    animation: fadeIn 1s ease-in-out;
}
.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 1.5rem;
}
.card {
    background: linear-gradient(145deg, #020617, #020617);
    border-radius: 18px;
    padding: 1.2rem;
    margin-top: 1rem;
    box-shadow: 0 10px 25px rgba(0,0,0,0.35);
    animation: slideUp 0.8s ease-in-out;
}
.emotion {
    display: flex;
    justify-content: space-between;
    font-weight: 600;
    margin-bottom: 0.4rem;
}
.progress {
    height: 10px;
    border-radius: 20px;
    background: #1e293b;
    overflow: hidden;
    margin-bottom: 0.8rem;
}
.progress span {
    display: block;
    height: 100%;
    background: linear-gradient(90deg, #38bdf8, #22c55e);
    animation: grow 1.3s ease forwards;
}
@keyframes grow { from { width: 0%; } }
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
@keyframes slideUp { from { transform: translateY(20px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
</style>
""", unsafe_allow_html=True)

# ============================
# Hugging Face Config
# ============================
API_URL = "https://router.huggingface.co/api/models/joeddav/xlm-roberta-large-xnli-go-emotions"
HF_TOKEN = st.secrets.get("HF_TOKEN", None) or os.getenv("HF_TOKEN", None)
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else None

# ============================
# Function to analyze emotion (cached to avoid repeated API calls)
# ============================
@st.cache_data(show_spinner=False)
def analyze_emotion(text):
    try:
        response = requests.post(
            API_URL,
            headers=HEADERS,
            json={"inputs": text},
            timeout=20
        )
        if response.status_code != 200:
            return {"error": f"API returned status {response.status_code}: {response.text}"}

        # Safely parse JSON
        try:
            result = response.json()
            return result
        except ValueError:
            return {"error": "Received empty or invalid JSON from Hugging Face API."}

    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}


# ============================
# UI
# ============================
st.markdown('<div class="title">🌍 Multilingual Emotion Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by LLMs • Supports any language</div>', unsafe_allow_html=True)

text = st.text_area(
    "Paste a comment (any language)",
    height=160,
    placeholder="Example: I feel proud and happy today!"
)

if st.button("✨ Analyze Emotion"):
    if not text.strip():
        st.warning("Please enter some text to analyze.")
    elif not HF_TOKEN:
        st.error("HF_TOKEN not configured. Set it in Streamlit Secrets or environment variable.")
    else:
        with st.spinner("🧠 Detecting language and analyzing emotions..."):
            # Detect language safely
            try:
                lang = detect(text)
            except:
                lang = "unknown"

            st.markdown(f"**Detected Language:** `{lang.upper()}`")

            # Call API
            result = analyze_emotion(text)

        if "error" in result:
            st.error(f"⚠️ API Error: {result['error']}")
        else:
            emotions = result[0][:5]  # Top 5 emotions
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🎭 Detected Emotions")

            for emo in emotions:
                percent = round(emo["score"] * 100, 2)
                st.markdown(
                    f"""
                    <div class="emotion">
                        <span>{emo['label']}</span>
                        <span>{percent}%</span>
                    </div>
                    <div class="progress">
                        <span style="width:{percent}%"></span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            primary = emotions[0]
            st.success(f"✨ Primary Emotion: **{primary['label']}** ({round(primary['score']*100,2)}%)")
            st.markdown('</div>', unsafe_allow_html=True)

# ============================
# Footer
# ============================
st.markdown(
    "<p style='text-align:center; color:#64748b; margin-top:2rem;'>Built by Akash • Streamlit + HuggingFace 🤍</p>",
    unsafe_allow_html=True
)
