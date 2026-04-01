import tkinter as tk
from tkinter import messagebox
import joblib
import pandas as pd
from datetime import datetime

# Load trained model & vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def check_news():
    text = news_entry.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Input Required", "Please enter a news article.")
        return

    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]

    # Display result
    if prediction.upper() == "REAL":
        result_label.config(text="✅ REAL NEWS", fg="green")
    else:
        result_label.config(text="❌ FAKE NEWS", fg="red")

    # Log prediction
    log = {
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "news_input": text,
        "prediction": prediction
    }
    try:
        df_log = pd.read_csv("prediction_log.csv")
        df_log = pd.concat([df_log, pd.DataFrame([log])], ignore_index=True)
    except FileNotFoundError:
        df_log = pd.DataFrame([log])
    df_log.to_csv("prediction_log.csv", index=False)

# Build GUI
root = tk.Tk()
root.title("📰 Fake News Detector")
root.geometry("650x450")
root.resizable(False, False)

tk.Label(root, text="Fake News Detector", font=("Helvetica", 20, "bold")).pack(pady=10)
tk.Label(root, text="Paste a news article below:", font=("Helvetica", 12)).pack()

news_entry = tk.Text(root, height=10, width=75, font=("Helvetica", 12), wrap="word")
news_entry.pack(padx=20, pady=10)

tk.Button(root, text="Check News", command=check_news,
          font=("Helvetica", 14), bg="#4CAF50", fg="white", padx=20, pady=5).pack(pady=10)

result_label = tk.Label(root, text="", font=("Helvetica", 16, "bold"))
result_label.pack(pady=10)

root.mainloop()
