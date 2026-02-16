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
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

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

    prompt = f"Explain why this news might be {prediction}: {text}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("⚙️ Options")

examples = [
    "India successfully landed Chandrayaan-3 on the Moon in 2023.",
    "Aliens have landed on Earth and taken control of governments.",
    "Scientists discovered a new vaccine for malaria.",
    "Drinking hot water cures all diseases instantly."
]

selected_example = st.sidebar.selectbox("Try Example News", ["None"] + examples)

if selected_example != "None":
    default_text = selected_example
else:
    default_text = ""

clear = st.sidebar.button("Clear Text")

# ==========================================
# MAIN UI
# ==========================================
st.title("📰 Fake News Detector")
st.caption("AI-powered verification system")

# Input
user_input = st.text_area("Enter news text", value=default_text, height=150)

if clear:
    user_input = ""

# Analyze
if st.button("Analyze News 🔍"):

    if user_input.strip() == "":
        st.warning("Enter text first")
    else:
        result, main_conf, other_conf = predict_news(user_input)

        # ===============================
        # RESULT
        # ===============================
        if result == "REAL":
            st.success(f"🟢 REAL NEWS ({main_conf*100:.2f}%)")
        else:
            st.error(f"🔴 FAKE NEWS ({main_conf*100:.2f}%)")

        # ===============================
        # STATS
        # ===============================
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Confidence", f"{main_conf*100:.2f}%")

        with col2:
            st.metric("Opposite Score", f"{other_conf*100:.2f}%")

        st.progress(float(main_conf))

        # ===============================
        # TABS
        # ===============================
        tab1, tab2 = st.tabs(["Result Details", "AI Explanation"])

        with tab1:
            st.write("### Analysis")
            st.write(f"Prediction: {result}")
            st.write(f"Confidence: {main_conf*100:.2f}%")

        with tab2:
            with st.spinner("Generating explanation..."):
                explanation = get_ai_explanation(user_input, result)

            st.write(explanation)

        # ===============================
        # SHARE
        # ===============================
        st.code(user_input)

        st.info("⚠️ AI prediction may not be 100% accurate")

# Footer
st.markdown("---")
st.caption("Developed by You 🚀")
