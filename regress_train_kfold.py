import re
import pandas as pd
import uuid
from datetime import datetime
import os
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# ==============================
# 1. Preprocessing
# ==============================
def preprocess(text):
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


# ==============================
# 2. Load Data
# ==============================
df = pd.read_csv("data/1.synthetic_wisesight_like_thai_sentiment_5000.csv")
df = df.rename(columns={"sentiment": "label"})
df["text"] = df["text"].apply(preprocess)

X = df["text"]
y = df["label"]


# ==============================
# 3. Stratified K-Fold Setup
# ==============================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

acc_scores = []
f1_scores = []

all_true = []
all_pred = []

fold = 1

print("\n===== START STRATIFIED 5-FOLD CROSS VALIDATION =====")

for train_index, test_index in skf.split(X, y):
    print(f"\n----- FOLD {fold} -----")

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Vectorizer (ต้อง fit ใหม่ทุก fold)
    vectorizer = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), max_features=10000
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Model
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)

    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")

    acc_scores.append(acc)
    f1_scores.append(f1_macro)

    all_true.extend(y_test)
    all_pred.extend(y_pred)

    print("Accuracy:", round(acc, 4))
    print("Macro-F1:", round(f1_macro, 4))

    fold += 1


# ==============================
# 4. Cross-Validation Summary
# ==============================
avg_acc = np.mean(acc_scores)
std_acc = np.std(acc_scores)

avg_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)

print("\n===== CROSS VALIDATION RESULTS =====")
print("Average Accuracy:", round(avg_acc, 4))
print("Std Accuracy:", round(std_acc, 4))
print("Average Macro-F1:", round(avg_f1, 4))
print("Std Macro-F1:", round(std_f1, 4))

print("\nClassification Report (All Folds Combined):\n")
print(classification_report(all_true, all_pred))


# ==============================
# 5. Save Final Model (train on full data)
# ==============================
print("\n===== TRAIN FINAL MODEL ON FULL DATA =====")

final_vectorizer = TfidfVectorizer(
    analyzer="word", ngram_range=(1, 2), max_features=10000
)

X_full_vec = final_vectorizer.fit_transform(X)

final_model = LogisticRegression(
    class_weight="balanced", max_iter=1000, random_state=42
)

final_model.fit(X_full_vec, y)


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
uid = uuid.uuid4().hex[:8]
model_uid = f"{timestamp}_{uid}"

os.makedirs("models_regress", exist_ok=True)

model_path = f"models_regress/sentiment_model_{model_uid}.joblib"
vectorizer_path = f"models_regress/vectorizer_{model_uid}.joblib"

joblib.dump(final_model, model_path)
joblib.dump(final_vectorizer, vectorizer_path)

print(f"Model saved as: {model_path}")
print(f"Vectorizer saved as: {vectorizer_path}")


# ==============================
# 6. Confusion Matrix (All Folds)
# ==============================
cm = confusion_matrix(all_true, all_pred)

os.makedirs("results_regress", exist_ok=True)
results_path = f"results_regress/cv_evaluation_{model_uid}.png"

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].axis("off")
axes[0].text(
    0.1,
    0.5,
    f"Model UID: {model_uid}\n\n"
    f"Avg Accuracy: {avg_acc:.4f} ± {std_acc:.4f}\n"
    f"Avg Macro-F1: {avg_f1:.4f} ± {std_f1:.4f}",
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
    xticklabels=final_model.classes_,
    yticklabels=final_model.classes_,
)

axes[1].set_title("Confusion Matrix (All Folds Combined)")
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")

plt.tight_layout()
plt.savefig(results_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"✅ Saved cross-validation evaluation image: {results_path}")
print("\n===== DONE =====")
