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

/* Background gradient */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b, #334155);
    color: white;
}

/* Center container */
.block-container {
    max-width: 700px;
    margin: auto;
}

/* Title */
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #38bdf8;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #cbd5f5;
    margin-bottom: 25px;
}

/* GLASS CARD */
.card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px);
    padding: 25px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.2);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

/* Input box */
textarea {
    background: rgba(255,255,255,0.15) !important;
    color: white !important;
    border-radius: 10px !important;
}

/* Button */
.stButton>button {
    width: 100%;
    height: 50px;
    border-radius: 12px;
    font-size: 18px;
    background: linear-gradient(90deg, #38bdf8, #0ea5e9);
    color: white;
    font-weight: bold;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.03);
}

/* Result */
.result-box {
    text-align: center;
    padding: 20px;
    border-radius: 15px;
    font-size: 22px;
    font-weight: bold;
    backdrop-filter: blur(10px);
}

/* Real */
.real {
    background: rgba(16,185,129,0.2);
    border: 1px solid rgba(16,185,129,0.5);
}

/* Fake */
.fake {
    background: rgba(239,68,68,0.2);
    border: 1px solid rgba(239,68,68,0.5);
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 30px;
    color: #cbd5f5;
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
