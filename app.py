import streamlit as st
from transformers import pipeline
from langdetect import detect

# ----------------------------
# Load LLM Emotion Model
# ----------------------------
@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="joeddav/xlm-roberta-large-xnli-go-emotions",
        top_k=5
    )

emotion_classifier = load_emotion_model()

# ----------------------------
# UI CONFIG
# ----------------------------
st.set_page_config(
    page_title="Multilingual Emotion Analyzer (LLM)",
    page_icon="🌍",
    layout="centered"
)

st.title("🌍 Multilingual Emotion Analyzer using LLM")
st.write(
    "Analyze comments in **any language** (Tamil, Telugu, Spanish, French, German, English, etc.) "
    "and detect **complex human emotions** using a large multilingual language model."
)

text = st.text_area(
    "Paste a comment (any language)",
    height=160,
    placeholder="உங்கள் கருத்தை இங்கே எழுதுங்கள் / Escribe aquí / Write here..."
)

if st.button("Analyze Emotion"):
    if not text.strip():
        st.warning("Please enter some text")
    else:
        # ----------------------------
        # Language Detection
        # ----------------------------
        try:
            language = detect(text)
        except:
            language = "Unknown"

        st.subheader("🌐 Detected Language")
        st.write(language.upper())

        # ----------------------------
        # Emotion Prediction
        # ----------------------------
        results = emotion_classifier(text)[0]

        st.subheader("🎭 Detected Emotions (Top 5)")
        for emo in results:
            st.write(
                f"**{emo['label'].capitalize()}** — {round(emo['score']*100, 2)}%"
            )
            st.progress(emo["score"])

        # ----------------------------
        # Primary Emotion Logic
        # ----------------------------
        primary_emotion = results[0]["label"]
        confidence = round(results[0]["score"] * 100, 2)

        st.subheader("🧠 Primary Emotion")
        st.success(f"{primary_emotion.capitalize()} ({confidence}%)")

        # ----------------------------
        # Ethical Insight
        # ----------------------------
        st.subheader("⚖️ Emotional Insight")
        if primary_emotion in ["anger", "disgust", "sadness", "fear", "remorse"]:
            st.error(
                "This content reflects strong negative emotions. "
                "Consider responding with empathy and care."
            )
        elif primary_emotion in ["joy", "love", "gratitude", "optimism"]:
            st.info(
                "This content expresses positive emotions. "
                "It can encourage healthy interactions."
            )
        else:
            st.warning(
                "This content reflects a mixed or neutral emotional state."
            )
        st.markdown("---")