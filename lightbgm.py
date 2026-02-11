# train_lgbm.py
import re
import pandas as pd
import uuid
from datetime import datetime
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


os.makedirs("models_lgbm", exist_ok=True)
os.makedirs("results_lgbm", exist_ok=True)

def preprocess(text):
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text

df = pd.read_csv("data/1.synthetic_wisesight_like_thai_sentiment_5000.csv")
df = df.rename(columns={"sentiment": "label"})
df["text"] = df["text"].apply(preprocess)

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LGBMClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    num_leaves=31,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_vec, y_train)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
uid = uuid.uuid4().hex[:8]
model_uid = f"{timestamp}_{uid}"

joblib.dump(model, f"models_lgbm/sentiment_model_{model_uid}.joblib")
joblib.dump(vectorizer, f"models_lgbm/vectorizer_{model_uid}.joblib")

y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")
cm = confusion_matrix(y_test, y_pred)

print(f"✅ Model saved: models_lgbm/sentiment_model_{model_uid}.joblib")
print("\n=== EVALUATION ===")
print(f"Accuracy: {acc:.4f}, Macro-F1: {f1_macro:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].axis("off")
metrics_text = f"Model UID: {model_uid}\n\nClassifier: LightGBM\nAccuracy: {acc:.4f}\nMacro-F1: {f1_macro:.4f}"
axes[0].text(0.1, 0.5, metrics_text, fontsize=14, verticalalignment="center", fontfamily="monospace")
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1], xticklabels=model.classes_, yticklabels=model.classes_)
axes[1].set_title("Confusion Matrix")
plt.tight_layout()
plt.savefig(f"results_lgbm/evaluation_{model_uid}.png", dpi=150, bbox_inches="tight")
plt.close()

errors_df = df.loc[X_test.index].copy()
errors_df["true_label"] = y_test.values
errors_df["pred_label"] = y_pred
errors_df = errors_df[errors_df["true_label"] != errors_df["pred_label"]]
errors_df.head(10).to_csv("data/error_examples_lgbm.csv", index=False, encoding="utf-8")