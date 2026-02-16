import streamlit as st
import joblib
import re

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# ===============================
# CUSTOM CSS (UI DESIGN)
# ===============================
st.markdown("""
<style>

body {
    background-color: #0f172a;
    color: white;
}

.main {
    background-color: #0f172a;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #38bdf8;
}

.subtitle {
    text-align: center;
    font-size: 16px;
    color: #94a3b8;
}

.card {
    background-color: #1e293b;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 0px 15px rgba(0,0,0,0.5);
    margin-top: 20px;
}

textarea {
    background-color: #0f172a !important;
    color: white !important;
}

.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 50px;
    font-size: 18px;
    background-color: #38bdf8;
    color: black;
    font-weight: bold;
}

.result-real {
    background-color: #065f46;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-size: 20px;
}

.result-fake {
    background-color: #7f1d1d;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-size: 20px;
}

.footer {
    text-align: center;
    color: #94a3b8;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODEL
# ===============================
model = joblib.load("fake_news_model.pkl")

# ===============================
# CLEAN TEXT
# ===============================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ===============================
# PREDICT FUNCTION
# ===============================
def predict_news(text):
    cleaned = clean_text(text)
    probs = model.predict_proba([cleaned])[0]
    fake = probs[0]
    real = probs[1]

    if real > fake:
        return "REAL", real
    else:
        return "FAKE", fake

# ===============================
# UI
# ===============================

# Title
st.markdown('<div class="title">📰 Fake News Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered news verification system</div>', unsafe_allow_html=True)

# Card
st.markdown('<div class="card">', unsafe_allow_html=True)

user_input = st.text_area("Enter news text", height=180)

if st.button("Analyze News 🔍"):

    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        result, confidence = predict_news(user_input)

        st.markdown("<br>", unsafe_allow_html=True)

        if result == "REAL":
            st.markdown(f'<div class="result-real">🟢 REAL NEWS<br>{confidence*100:.2f}% confidence</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-fake">🔴 FAKE NEWS<br>{confidence*100:.2f}% confidence</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.progress(float(confidence))

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">Developed with ❤️ by You</div>', unsafe_allow_html=True)

