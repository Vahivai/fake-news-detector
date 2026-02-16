import streamlit as st
import joblib
import re

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #38bdf8;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 30px;
}

.card {
    background: #1e293b;
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0 0 25px rgba(0,0,0,0.4);
}

textarea {
    background-color: #ffffff !important;
    color: black !important;
    border-radius: 10px !important;
    padding: 10px !important;
}

.stButton>button {
    width: 100%;
    border-radius: 12px;
    height: 55px;
    font-size: 18px;
    background: linear-gradient(90deg, #38bdf8, #0ea5e9);
    color: white;
    font-weight: bold;
    border: none;
}

.result {
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    padding: 20px;
    border-radius: 12px;
}

.real {
    background-color: #065f46;
}

.fake {
    background-color: #7f1d1d;
}

.footer {
    text-align: center;
    color: #94a3b8;
    margin-top: 40px;
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
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ==========================================
# EXPLANATION FUNCTION
# ==========================================
def generate_explanation(text, prediction):
    text_lower = text.lower()

    fake_keywords = ["aliens", "miracle", "cure all", "shocking", "secret", "invisible", "instant", "forever"]
    real_keywords = ["report", "study", "research", "official", "announced", "data", "according"]

    explanation = ""

    if prediction == "FAKE":
        explanation += "This news may be fake because:\n"
        
        if any(word in text_lower for word in fake_keywords):
            explanation += "- It contains sensational or unrealistic claims.\n"
        
        if len(text.split()) < 8:
            explanation += "- The text is too short and lacks details.\n"

        explanation += "- Fake news often uses exaggerated or emotional language.\n"
    
    else:
        explanation += "This news may be real because:\n"
        
        if any(word in text_lower for word in real_keywords):
            explanation += "- It uses formal or factual language.\n"
        
        if len(text.split()) > 8:
            explanation += "- It provides more detailed information.\n"

        explanation += "- It looks like a normal informational statement.\n"

    return explanation

# ==========================================
# PREDICT
# ==========================================
def predict_news(text):
    cleaned = clean_text(text)
    probs = model.predict_proba([cleaned])[0]

    fake_prob = probs[0]
    real_prob = probs[1]

    if real_prob > fake_prob:
        return "REAL", real_prob
    else:
        return "FAKE", fake_prob

# ==========================================
# UI
# ==========================================
st.markdown('<div class="title">📰 Fake News Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered news verification system</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)

user_input = st.text_area("Enter news text", height=180)

if st.button("Analyze News 🔍"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        result, confidence = predict_news(user_input)

        st.markdown("<br>", unsafe_allow_html=True)

        # RESULT
        if result == "REAL":
            st.markdown(f'<div class="result real">🟢 REAL NEWS<br>{confidence*100:.2f}% confidence</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result fake">🔴 FAKE NEWS<br>{confidence*100:.2f}% confidence</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Confidence bar
        st.progress(float(confidence))

        # EXPLANATION
        explanation = generate_explanation(user_input, result)

        st.markdown("### 🤖 Explanation")
        st.write(explanation)

        st.info("⚠️ This prediction is based on AI and may not always be correct.")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">Developed with ❤️ by You</div>', unsafe_allow_html=True)
