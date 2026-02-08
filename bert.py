# ============================================================
# Unified Training Script
# - Baseline: TF-IDF (word-level) + Logistic Regression
# - Improved: Thai BERT (Transfer Learning)
# ============================================================

import re
import os
import uuid
from datetime import datetime
import pandas as pd
import joblib
import numpy as np

# ---------- Baseline (Classic ML) ----------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# ---------- BERT ----------
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# ============================================================
# 1. Preprocessing
# ============================================================
def preprocess(text):
    """
    Minimal preprocessing (ตาม requirement):
    - strip whitespace
    - normalize whitespace
    - ไม่ลบ emoji / slang
    """
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


# ============================================================
# 2. Load Dataset
# ============================================================
DATA_PATH = "data/5.ultimate_sentiment_100k.csv"

df = pd.read_csv(DATA_PATH)
df = df.rename(columns={"sentiment": "label"})
df["text"] = df["text"].apply(preprocess)

print("Dataset size:", len(df))
print(df["label"].value_counts())


# ============================================================
# 3. Train / Test Split
# ============================================================
X = df["text"]
y = df["label"]

df_train, df_test = train_test_split(df[["text", "label"]], test_size=0.2, random_state=42, stratify=df["label"])
X_train, y_train = df_train["text"], df_train["label"]
X_test, y_test = df_test["text"], df_test["label"]

# ============================================================
# 4. BASELINE MODEL
#    TF-IDF (word-level) + Logistic Regression
# ============================================================
print("\n===== TRAINING BASELINE MODEL =====")

vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    max_features=10000,
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

baseline_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42,
)
baseline_model.fit(X_train_vec, y_train)

y_pred = baseline_model.predict(X_test_vec)

baseline_acc = accuracy_score(y_test, y_pred)
baseline_f1 = f1_score(y_test, y_pred, average="macro")
baseline_cm = confusion_matrix(y_test, y_pred)

print("\n[Baseline Evaluation]")
print("Accuracy:", round(baseline_acc, 4))
print("Macro-F1:", round(baseline_f1, 4))
print("Confusion Matrix:\n", baseline_cm)
print(classification_report(y_test, y_pred))

# ---- Save Baseline Model (.joblib ตาม requirement) ----
os.makedirs("models", exist_ok=True)
uid = uuid.uuid4().hex[:8]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

baseline_model_path = f"models/baseline_lr_{timestamp}_{uid}.joblib"
baseline_vectorizer_path = f"models/baseline_vectorizer_{timestamp}_{uid}.joblib"

joblib.dump(baseline_model, baseline_model_path)
joblib.dump(vectorizer, baseline_vectorizer_path)

print("Saved baseline model:", baseline_model_path)


# ============================================================
# 5. ERROR ANALYSIS (Baseline)
# ============================================================
errors_df = pd.DataFrame({
    "text": X_test.values,
    "true_label": y_test.values,
    "pred_label": y_pred,
})
errors_df = errors_df[errors_df["true_label"] != errors_df["pred_label"]]

errors_df.head(10).to_csv(
    "data/error_examples_baseline.csv",
    index=False,
    encoding="utf-8"
)

print("Saved baseline error examples (10+ rows)")


# ============================================================
# 6. IMPROVED MODEL: Thai BERT (Transfer Learning)
# ============================================================
print("\n===== TRAINING BERT MODEL =====")

df_bert = df.copy()
# แปลง label เป็นตัวพิมพ์เล็ก แล้ว map เป็นตัวเลขทันที
df_bert["label"] = df_bert["label"].str.lower().map({
    "negative": 0,
    "neutral": 1,
    "positive": 2
})

# ตรวจสอบว่าไม่มี NaN (ถ้ามี แสดงว่า label ไม่ตรงกับ 3 ค่านี้)
if df_bert["label"].isnull().any():
    unique_labels = df["label"].unique()
    raise ValueError(f"พบ label ที่ไม่รู้จัก! Labels ในข้อมูล: {unique_labels}")

# สร้าง Dataset
dataset = Dataset.from_pandas(df_bert[["text", "label"]])

# 🔥 แปลงคอลัมน์ 'label' ให้เป็น ClassLabel (จำเป็นสำหรับ stratify)
from datasets import ClassLabel
class_labels = ClassLabel(num_classes=3, names=["negative", "neutral", "positive"])
dataset = dataset.cast_column("label", class_labels)

# แบ่งข้อมูลแบบ stratify
dataset = dataset.train_test_split(
    test_size=0.2,
    stratify_by_column="label",
    seed=42
)

MODEL_NAME = "FlukeTJ/distilbert-base-thai-sentiment"

# 🔥 แก้ไขหลัก: ใช้ use_fast=False เพราะ WangChanBERTa ใช้ SentencePiece tokenizer (ไม่มี fast version)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

bert_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3
)

training_args = TrainingArguments(
    output_dir="bert-output",
    eval_strategy="epoch",      # 🔥 เปลี่ยนชื่อ
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir="bert-logs",
    report_to="none",
)

trainer = Trainer(
    model=bert_model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

trainer.train()

# ============================================================
# 7. Save BERT Model (for Web Deployment)
# ============================================================
BERT_SAVE_PATH = "models/bert_thai_sentiment"
os.makedirs(BERT_SAVE_PATH, exist_ok=True)

bert_model.save_pretrained(BERT_SAVE_PATH)
tokenizer.save_pretrained(BERT_SAVE_PATH)

print("Saved BERT model to:", BERT_SAVE_PATH)


# ============================================================
# 8. FINAL SUMMARY
# ============================================================
print("\n===== SUMMARY =====")
print("Baseline Accuracy:", round(baseline_acc, 4))
print("Baseline Macro-F1:", round(baseline_f1, 4))
print("Baseline model (.joblib):", baseline_model_path)
print("Improved model (BERT):", BERT_SAVE_PATH)
print("Done ✅")