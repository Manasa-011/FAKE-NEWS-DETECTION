import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# 1. Load and prepare dataset
df = pd.read_csv("realistic_fake_news_dataset.csv")
df = df.dropna(subset=['text', 'label'])

# 2. Features and labels
X = df['text']
y = df['label']

# 3. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = vectorizer.fit_transform(X_train)
tfidf_test  = vectorizer.transform(X_test)

# 5. Train model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(tfidf_train, y_train)

# 6. Evaluate
y_pred = model.predict(tfidf_test)
acc = accuracy_score(y_test, y_pred)
cm  = confusion_matrix(y_test, y_pred)
print(f"\n✅ Model trained. Accuracy: {acc*100:.2f}%")
print("Confusion Matrix:\n", cm)

# 7. Save artifacts
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("\nArtifacts saved: model.pkl, vectorizer.pkl")
