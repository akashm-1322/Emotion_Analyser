import streamlit as st
import requests
from langdetect import detect

# ----------------------------
# Hugging Face Inference API
# ----------------------------
API_URL = "https://api-inference.huggingface.co/models/joeddav/xlm-roberta-large-xnli-go-emotions"
HF_TOKEN = st.secrets["HF_TOKEN"]

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def analyze_emotion(text):
    response = requests.post(
        API_URL,
        headers=headers,
        json={"inputs": text}
    )
    return response.json()

# ----------------------------
# UI
# ----------------------------
st.set_page_config(
    page_title="Multilingual Emotion Analyzer (LLM)",
    page_icon="🌍"
)

st.title("🌍 Multilingual Emotion Analyzer using LLM")

text = st.text_area(
    "Paste a comment (any language)",
    height=150
)

if st.button("Analyze Emotion"):
    if not text.strip():
        st.warning("Please enter text")
    else:
        # Language detection
        try:
            lang = detect(text)
        except:
            lang = "unknown"

        st.write("**Detected Language:**", lang.upper())

        with st.spinner("Analyzing emotions using LLM..."):
            result = analyze_emotion(text)

        if isinstance(result, dict) and "error" in result:
            st.error(result["error"])
        else:
            emotions = result[0][:5]

            st.subheader("🎭 Detected Emotions")
            for emo in emotions:
                st.write(f"**{emo['label']}** — {round(emo['score']*100,2)}%")
                st.progress(emo["score"])

            primary = emotions[0]
            st.success(
                f"Primary Emotion: {primary['label']} ({round(primary['score']*100,2)}%)"
            )
