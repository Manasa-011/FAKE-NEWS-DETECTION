# app.py
import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detection App")
st.markdown("Type or paste a news article below to check whether it's **REAL** or **FAKE**.")

news_input = st.text_area("Enter news article:", height=250)

if st.button("Check News"):
    if not news_input.strip():
        st.warning("⚠️ Please enter some news text.")
    else:
        input_vector = vectorizer.transform([news_input])
        prediction = model.predict(input_vector)[0]

        if prediction == "REAL" or prediction == 1:
            st.success("✅ This appears to be REAL NEWS.")
        else:
            st.error("❌ This appears to be FAKE NEWS.")

        # Optional: Save to log
        log_data = {
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "news_input": news_input,
            "prediction": "REAL" if prediction == 1 or prediction == "REAL" else "FAKE"
        }
        try:
            df_log = pd.read_csv("prediction_log.csv")
            df_log = pd.concat([df_log, pd.DataFrame([log_data])], ignore_index=True)
        except FileNotFoundError:
            df_log = pd.DataFrame([log_data])

        df_log.to_csv("prediction_log.csv", index=False)

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
