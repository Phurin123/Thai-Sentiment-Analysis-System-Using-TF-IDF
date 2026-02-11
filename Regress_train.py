import re
import pandas as pd
import uuid
from datetime import datetime
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# === 1. Preprocessing function ===
def preprocess(text):
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


# === 2. Load and prepare data ===
df = pd.read_csv("data/1.synthetic_wisesight_like_thai_sentiment_5000.csv")
df = df.rename(columns={"sentiment": "label"})
df["text"] = df["text"].apply(preprocess)

# === 3. Train-test split ===
X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

# === 4. Train baseline model (TF-IDF + Logistic Regression) ===
vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=10000)

X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)

# === 5. Generate model UID ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
uid = uuid.uuid4().hex[:8]
model_uid = f"{timestamp}_{uid}"

# === 6. Save model & vectorizer ===
os.makedirs("models_regress", exist_ok=True)

model_path = f"models_regress/sentiment_model_{model_uid}.joblib"
vectorizer_path = f"models_regress/vectorizer_{model_uid}.joblib"

joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print(f"Model saved as: {model_path}")
print(f"Vectorizer saved as: {vectorizer_path}")

# === 7. Evaluation ===
X_test_vec = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vec)

acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")
cm = confusion_matrix(y_test, y_pred)

print("\n=== EVALUATION RESULTS ===")
print("Accuracy:", round(acc, 4))
print("Macro-F1:", round(f1_macro, 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# === 7.1 Save evaluation image ===
os.makedirs("results_regress", exist_ok=True)
results_path = f"results_regress/evaluation_{model_uid}.png"

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].axis("off")
axes[0].text(
    0.1,
    0.5,
    f"Model UID: {model_uid}\n\nAccuracy: {acc:.4f}\nMacro-F1: {f1_macro:.4f}",
    fontsize=14,
    verticalalignment="center",
    fontfamily="monospace",
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=axes[1],
    xticklabels=model.classes_,
    yticklabels=model.classes_,
)

axes[1].set_title("Confusion Matrix")
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")

plt.tight_layout()
plt.savefig(results_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"✅ Saved evaluation image: {results_path}")

# === 8. Create misclassified DataFrame (FIXED BUG) ===
errors_df = pd.DataFrame(
    {"text": X_test.values, "true_label": y_test.values, "pred_label": y_pred}
)

errors_df = errors_df[errors_df["true_label"] != errors_df["pred_label"]]

# === 9. Save 10 misclassified examples ===
os.makedirs("data", exist_ok=True)
ERRORS_OUTPUT_PATH = "data/error_examples.csv"

errors_df.head(10).to_csv(ERRORS_OUTPUT_PATH, index=False, encoding="utf-8")

print(f"✅ Saved 10 misclassified examples to: {ERRORS_OUTPUT_PATH}")


# === 10. Error Analysis ===
def categorize_error(text, true_label, pred_label):
    # Mixed signal
    neg_words = ["ไม่", "แย่", "ผิดหวัง", "ควรปรับปรุง"]
    pos_words = ["ดีมาก", "ประทับใจ", "โอเค", "ชอบ"]

    has_neg = any(w in text for w in neg_words)
    has_pos = any(w in text for w in pos_words)

    if has_neg and has_pos:
        return "Mixed Signal / Ambiguity"

    # Sarcasm / Informal
    if any(e in text for e in ["😤", "🙄", "😒", "🙂", "😊"]) or "แม่ง" in text:
        return "Sarcasm / Informal Expression"

    # Ambiguous neutral
    neutral_phrases = ["ยังไม่มีอะไรโดดเด่น", "อยู่ในระดับปกติ", "ไม่ได้แย่ แต่ก็ไม่ได้ดี"]
    if any(p in text for p in neutral_phrases):
        return "Ambiguous Neutral Expression"

    return "Other"


errors_df["error_type"] = errors_df.apply(
    lambda row: categorize_error(row["text"], row["true_label"], row["pred_label"]),
    axis=1,
)

print("\n=== ERROR ANALYSIS ===")
error_counts = errors_df["error_type"].value_counts()
print(error_counts)

most_common_error = error_counts.idxmax()
print(
    f"\nMost common error type: {most_common_error} "
    f"({error_counts[most_common_error]} cases)"
)

if most_common_error == "Mixed Signal / Ambiguity":
    suggestion = "ใช้ bigram เพิ่มเติมหรือ fine-tune Thai BERT"
elif most_common_error == "Sarcasm / Informal Expression":
    suggestion = "normalize emoji/slang หรือเพิ่มข้อมูล informal"
elif most_common_error == "Ambiguous Neutral Expression":
    suggestion = "ใช้ confidence threshold แทน hard label"
else:
    suggestion = "ใช้ transformer-based model"

print(f"\nSuggested improvement: {suggestion}")
