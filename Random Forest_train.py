# train.py
import re
import pandas as pd
import uuid
from datetime import datetime
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
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

# =========================
# Directory setup
# =========================
os.makedirs("models_tree", exist_ok=True)
os.makedirs("results_tree", exist_ok=True)
os.makedirs("data", exist_ok=True)

# =========================
# 1. Preprocessing
# =========================
def preprocess(text):
    """
    - แปลงเป็น string และลบช่องว่างหน้า-หลัง
    - normalize whitespace
    - ไม่ lowercase / ไม่ลบ emoji (ตาม requirement)
    """
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


# =========================
# 2. Load data
# =========================
df = pd.read_csv("data/5.ultimate_sentiment_100k.csv")
df = df.rename(columns={"sentiment": "label"})
df["text"] = df["text"].apply(preprocess)

# =========================
# 3. Train-test split
# =========================
X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 4. TF-IDF (WORD-LEVEL)
# =========================
vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    max_features=10000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =========================
# 5. NON-LINEAR MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_vec, y_train)

# =========================
# 6. Save model
# =========================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
uid = uuid.uuid4().hex[:8]
model_uid = f"{timestamp}_{uid}"

model_path = f"models_tree/sentiment_model_{model_uid}.joblib"
vectorizer_path = f"models_tree/vectorizer_{model_uid}.joblib"

joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print(f"✅ Model saved: {model_path}")
print(f"✅ Vectorizer saved: {vectorizer_path}")

# =========================
# 7. Evaluation
# =========================
y_pred = model.predict(X_test_vec)

acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")
cm = confusion_matrix(y_test, y_pred)

print("\n=== EVALUATION RESULTS ===")
print("Accuracy:", round(acc, 4))
print("Macro-F1:", round(f1_macro, 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# =========================
# 7.1 Save evaluation image
# =========================
results_path = f"results_tree/evaluation_{model_uid}.png"

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Metrics
axes[0].axis("off")
metrics_text = (
    f"Model UID: {model_uid}\n\n"
    f"Classifier: RandomForest (Non-linear)\n"
    f"Accuracy: {acc:.4f}\n"
    f"Macro-F1: {f1_macro:.4f}"
)
axes[0].text(
    0.1, 0.5, metrics_text,
    fontsize=14,
    verticalalignment="center",
    fontfamily="monospace"
)

# Confusion matrix
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=axes[1],
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

axes[1].set_title("Confusion Matrix")
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")

plt.tight_layout()
plt.savefig(results_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"✅ Saved evaluation image: {results_path}")

# =========================
# 8. Misclassified examples
# =========================
errors_df = df.loc[X_test.index].copy()
errors_df["true_label"] = y_test.values
errors_df["pred_label"] = y_pred
errors_df = errors_df[errors_df["true_label"] != errors_df["pred_label"]]

errors_df.head(10).to_csv(
    "data/error_examples_tree.csv",
    index=False,
    encoding="utf-8"
)

print("✅ Saved misclassified examples")

# =========================
# 9. Error analysis
# =========================
def categorize_error(text, true_label, pred_label):
    neg_words = ["ไม่", "แย่", "ผิดหวัง", "ควรปรับปรุง", "แม่ง"]
    pos_words = ["ดีมาก", "ประทับใจ", "โอเค", "ชอบ"]

    has_neg = any(w in text for w in neg_words)
    has_pos = any(w in text for w in pos_words)

    if has_neg and has_pos:
        return "Mixed Signal / Ambiguity"

    if any(e in text for e in ["😤", "🙄", "😒", "🙂", "😊"]):
        return "Sarcasm / Informal Expression"

    neutral_phrases = [
        "ยังไม่มีอะไรโดดเด่น",
        "อยู่ในระดับปกติ",
        "ไม่ได้แย่ แต่ก็ไม่ได้ดี"
    ]
    if any(p in text for p in neutral_phrases):
        return "Ambiguous Neutral Expression"

    return "Other"


errors_df["error_type"] = errors_df.apply(
    lambda r: categorize_error(r["text"], r["true_label"], r["pred_label"]),
    axis=1
)

print("\n=== ERROR ANALYSIS ===")
print(errors_df["error_type"].value_counts())
