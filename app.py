import streamlit as st
import joblib
import re

# Load model
model = joblib.load("fake_news_model.pkl")

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Prediction
def check_news(text):
    cleaned = clean_text(text)

    probs = model.predict_proba([cleaned])[0]

    fake_prob = probs[0]
    real_prob = probs[1]

    if real_prob > fake_prob:
        return "REAL", real_prob
    else:
        return "FAKE", fake_prob

# UI
st.title("📰 Fake News Detector")

st.write("Enter news text below:")

user_input = st.text_area("News Text")

if st.button("Check"):

    if len(user_input.strip()) == 0:
        st.warning("Enter some text")

    else:
        result, confidence = check_news(user_input)

        if result == "REAL":
            st.success(f"REAL NEWS ({confidence*100:.2f}%)")
        else:
            st.error(f"FAKE NEWS ({confidence*100:.2f}%)")

