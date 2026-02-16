import streamlit as st
import joblib
import re

# Try OpenAI
try:
    from openai import OpenAI
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    ai_enabled = True
except:
    ai_enabled = False

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# ==========================================
# CUSTOM CSS (PRO UI)
# ==========================================
st.markdown("""
<style>

/* Hide default header */
header {visibility: hidden;}
footer {visibility: hidden;}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

/* Main container */
.block-container {
    max-width: 700px;
    margin: auto;
}

/* Title */
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #38bdf8;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 25px;
}

/* Card */
.card {
    background: #1e293b;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 0px 20px rgba(0,0,0,0.4);
}

/* Input */
textarea {
    background: #ffffff !important;
    color: black !important;
    border-radius: 10px !important;
}

/* Button */
.stButton>button {
    width: 100%;
    height: 50px;
    border-radius: 10px;
    font-size: 18px;
    background: linear-gradient(90deg, #38bdf8, #0ea5e9);
    color: white;
    font-weight: bold;
    border: none;
}

/* Result */
.result-box {
    text-align: center;
    padding: 20px;
    border-radius: 10px;
    font-size: 22px;
    font-weight: bold;
}

.real {
    background: #065f46;
}

.fake {
    background: #7f1d1d;
}

/* Metrics */
.metric {
    text-align: center;
    font-size: 18px;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 30px;
    color: #94a3b8;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL
# ==========================================
model = joblib.load("fake_news_model.pkl")

# ==========================================
# CLEAN TEXT
# ==========================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return text

# ==========================================
# PREDICT
# ==========================================
def predict_news(text):
    cleaned = clean_text(text)
    probs = model.predict_proba([cleaned])[0]

    fake = probs[0]
    real = probs[1]

    if real > fake:
        return "REAL", real, fake
    else:
        return "FAKE", fake, real

# ==========================================
# AI EXPLANATION
# ==========================================
def get_ai_explanation(text, prediction):
    if not ai_enabled:
        return "AI explanation not available"

    prompt = f"Explain why this news is {prediction} in simple words: {text}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# ==========================================
# UI
# ==========================================

st.markdown('<div class="title">📰 Fake News Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered verification system</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)

user_input = st.text_area("Enter news text", height=150)

if st.button("Analyze News 🔍"):

    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        result, main_conf, other_conf = predict_news(user_input)

        st.markdown("<br>", unsafe_allow_html=True)

        # RESULT
        if result == "REAL":
            st.markdown(f'<div class="result-box real">🟢 REAL NEWS<br>{main_conf*100:.2f}%</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-box fake">🔴 FAKE NEWS<br>{main_conf*100:.2f}%</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # METRICS
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Confidence", f"{main_conf*100:.2f}%")

        with col2:
            st.metric("Opposite", f"{other_conf*100:.2f}%")

        st.progress(float(main_conf))

        # AI EXPLANATION
        with st.expander("🤖 AI Explanation"):
            explanation = get_ai_explanation(user_input, result)
            st.write(explanation)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">Developed by Muhammed and Ouku 🚀</div>', unsafe_allow_html=True)
