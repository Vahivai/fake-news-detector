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
# CSS
# ==========================================
st.markdown("""
<style>

/* Background */
body {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}

/* Glass Card */
.glass {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(15px);
    border-radius: 20px;
    padding: 30px;
    margin: 40px auto;
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
    margin-bottom: 20px;
}

/* Section headings */
h3 {
    color: #38bdf8;
}

/* Divider */
hr {
    border: none;
    height: 1px;
    background: rgba(255,255,255,0.2);
    margin: 20px 0;
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

/* Focus */
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

.real {
    background: rgba(34,197,94,0.2);
    color: #22c55e;
}

.fake {
    background: rgba(239,68,68,0.2);
    color: #ef4444;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 30px;
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
# GLASS UI
# ==========================================
st.markdown('<div class="glass">', unsafe_allow_html=True)

# Title
st.markdown('<div class="title">📰 Fake News Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered news verification system</div>', unsafe_allow_html=True)

st.markdown("---")

# HOW IT WORKS
st.markdown("### 🔎 How it works")
st.markdown("""
- Paste any news text 📰  
- Click **Analyze News**  
- Get instant result ⚡  
""")

# EXAMPLES
st.markdown("### 💡 Try examples")
st.markdown("""
- "Aliens have landed in New York City"
- "India successfully landed Chandrayaan-3 on the Moon in 2023"
- "Scientists say drinking water makes you immortal"
""")

# INPUT
st.markdown("### ✍️ Enter news text")

user_input = st.text_area(
    "",
    height=150,
    placeholder="Paste your news here..."
)

# BUTTON
predict_btn = st.button("🚀 Analyze News")

# ==========================================
# PREDICTION
# ==========================================
if predict_btn:

    if user_input.strip() == "":
        st.warning("Please enter text")
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

# CLOSE GLASS
st.markdown('</div>', unsafe_allow_html=True)

# FOOTER
st.markdown('<div class="footer">Developed by Muhammed and Ouku 🚀</div>', unsafe_allow_html=True)
