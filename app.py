import streamlit as st
import joblib

# ==========================================
# LOAD MODEL
# ==========================================
model = joblib.load("fake_news_model.pkl")

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# ==========================================
# CSS (GLASS UI + ANIMATION)
# ==========================================
st.markdown("""
<style>

/* Background */
body {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}

/* Glass Container */
.glass {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(15px);
    border-radius: 20px;
    padding: 30px;
    margin: 20px auto;
    width: 85%;
    border: 1px solid rgba(255,255,255,0.2);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    animation: fadeIn 1s ease-in-out;
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
    margin-top: 5px;
}

/* Input */
textarea {
    background: rgba(255,255,255,0.1) !important;
    color: white !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
    padding: 10px !important;
    transition: 0.3s;
}

/* Focus effect */
textarea:focus {
    border: 1px solid #38bdf8 !important;
    box-shadow: 0 0 12px #38bdf8;
}

/* Button */
button {
    background: linear-gradient(135deg, #38bdf8, #0ea5e9);
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    transition: 0.3s;
}

/* Hover */
button:hover {
    transform: scale(1.05);
}

/* Result */
.result {
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    animation: fadeIn 0.8s ease-in-out;
}

/* Real */
.real {
    background: rgba(34,197,94,0.2);
    color: #22c55e;
}

/* Fake */
.fake {
    background: rgba(239,68,68,0.2);
    color: #ef4444;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 40px;
    color: #94a3b8;
}

/* Animation */
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# GLASS HEADER (TITLE INSIDE GLASS)
# ==========================================
st.markdown('<div class="glass">', unsafe_allow_html=True)

st.markdown('<div class="title">📰 Fake News Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered news verification system</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# INPUT SECTION
# ==========================================
st.markdown('<div class="glass">', unsafe_allow_html=True)

user_input = st.text_area(
    "Enter news text",
    height=150,
    placeholder="Type or paste news here..."
)

predict_btn = st.button("🔍 Analyze News")

st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# PREDICTION
# ==========================================
if predict_btn:

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        prediction = model.predict([user_input])[0]
        proba = model.predict_proba([user_input])[0]

        confidence = max(proba) * 100

        if prediction == 1:
            st.markdown(
                f'<div class="result real">🟢 REAL NEWS<br>{confidence:.2f}% confidence</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result fake">🔴 FAKE NEWS<br>{confidence:.2f}% confidence</div>',
                unsafe_allow_html=True
            )

# ==========================================
# FOOTER
# ==========================================
st.markdown('<div class="footer">Developed by Muhammed and Ouku 🚀</div>', unsafe_allow_html=True)
