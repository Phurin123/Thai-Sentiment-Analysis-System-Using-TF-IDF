# train.py
import re
import pandas as pd
import uuid
from datetime import datetime

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
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import seaborn as sns

import os
os.makedirs("models_linear", exist_ok=True)

# === 1. Preprocessing function ===
def preprocess(text):
    """
    เหตุผลของการ preprocessing:
    - แปลงเป็น string และลบช่องว่างหน้า-หลัง: ป้องกันค่า null หรือช่องว่างรบกวน
    - normalize whitespace: รวมช่องว่างซ้อนเป็นช่องเดียว → ลด noise จากการพิมพ์
    - ไม่ lowercase เพราะภาษาไทยไม่มีตัวพิมพ์ใหญ่/เล็ก
    - ไม่ลบ emoji/slang เพราะโจทย์ห้าม over-cleaning
    """
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)  # normalize whitespace
    return text


# === 2. Load and prepare data ===
df = pd.read_csv("data/5.ultimate_sentiment_100k.csv")
df = df.rename(columns={"sentiment": "label"})
df["text"] = df["text"].apply(preprocess)

# === 3. Train-test split ===
X = df["text"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === 4. Train model (WORD-LEVEL TF-IDF ตาม requirement) ===
vectorizer = TfidfVectorizer(
    analyzer="word",        # ← แก้เป็น word-level ตามโจทย์
    ngram_range=(1, 2),     # unigram + bigram เพื่อจับ context ได้ดีขึ้น
    max_features=10000      # จำกัด feature เพื่อความเร็วและลด overfit
)
X_train_vec = vectorizer.fit_transform(X_train)
model = LinearSVC(
    class_weight="balanced",
    random_state=42
)
model.fit(X_train_vec, y_train)
model.fit(X_train_vec, y_train)

# === 5. Generate UID for model version ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
uid = uuid.uuid4().hex[:8]
model_uid = f"{timestamp}_{uid}"

# === 6. Save model and vectorizer ===
model_path = f"models_linear/sentiment_model_{model_uid}.joblib"
vectorizer_path = f"models_linear/vectorizer_{model_uid}.joblib"

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
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# === 7.1 Save evaluation results as image ===
os.makedirs("results_linear", exist_ok=True)
results_path = f"results_linear/evaluation_{model_uid}.png"

# สร้าง figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- ซ้าย: Metrics as text ---
axes[0].axis('off')
metrics_text = (
    f"Model UID: {model_uid}\n\n"
    f"Classifier: LinearSVC\n"
    f"Accuracy: {acc:.4f}\n"
    f"Macro-F1:  {f1_macro:.4f}"
)
axes[0].text(0.1, 0.5, metrics_text, fontsize=14, verticalalignment='center', fontfamily='monospace')

# --- ขวา: Confusion Matrix ---
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=axes[1],
    xticklabels=model.classes_,
    yticklabels=model.classes_
)
axes[1].set_title('Confusion Matrix')
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('True Label')

plt.tight_layout()
plt.savefig(results_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"✅ Saved evaluation results as image: {results_path}")

# === 8. Show ≥10 misclassified examples ===
test_indices = X_test.index
errors_df = df.loc[test_indices].copy()
errors_df["true_label"] = y_test.values
errors_df["pred_label"] = y_pred

print("\n=== 10 MISCLASSIFIED EXAMPLES ===")
for idx, (_, row) in enumerate(errors_df.head(10).iterrows()):
    print(f"{idx+1}. Text: {row['text']}")
    print(f"    True: {row['true_label']} | Pred: {row['pred_label']}\n")

# === 9. Error Analysis (ต้องมี ≥3 ประเภท) ===
def categorize_error(text, true_label, pred_label):
    text_lower = text.lower()
    # 1. Negation / Mixed signal
    neg_words = ["ไม่", "แม่ง", "แย่กว่าที่คิด", "ผิดหวัง", "ควรปรับปรุง"]
    pos_words = ["ดีมาก", "ประทับใจ", "โอเคเกินคาด", "ชอบ"]
    
    has_neg = any(w in text for w in neg_words)
    has_pos = any(w in text for w in pos_words)
    
    if has_neg and has_pos:
        return "Mixed Signal / Ambiguity"
    
    # 2. Sarcasm / Informal tone (เช่น ใช้ emoji หรือคำหยาบแต่ sentiment ไม่ชัด)
    if any(emoji in text for emoji in ["😤", "🙄", "😒", "🙂", "😊"]) or "แม่ง" in text:
        return "Sarcasm / Informal Expression"
    
    # 3. Domain-specific ambiguity (เช่น พูดถึง "ระบบ" แต่ไม่ชัดเจนว่าดี/แย่)
    neutral_phrases = ["ยังไม่มีอะไรโดดเด่น", "อยู่ในระดับปกติ", "ไม่ได้แย่ แต่ก็ไม่ได้ดี"]
    if any(phrase in text for phrase in neutral_phrases):
        return "Ambiguous Neutral Expression"
    
    # Default
    return "Other"

# จัดกลุ่ม error
errors_df["error_type"] = errors_df.apply(
    lambda row: categorize_error(row["text"], row["true_label"], row["pred_label"]), axis=1
)

print("\n=== ERROR ANALYSIS (Grouped by Type) ===")
error_counts = errors_df["error_type"].value_counts()
print(error_counts)

most_common_error = error_counts.idxmax()
print(f"\nMost common error type: '{most_common_error}' ({error_counts[most_common_error]} cases)")

# เสนอแนวทางแก้ไข
if most_common_error == "Mixed Signal / Ambiguity":
    suggestion = "เพิ่มการ detect negation และ contextual cues ด้วย rule-based หรือใช้โมเดล sequence เช่น BERT"
elif most_common_error == "Sarcasm / Informal Expression":
    suggestion = "เพิ่มการ normalize emoji และ slang อย่างระมัดระวัง หรือ fine-tune บนข้อมูล informal Thai"
elif most_common_error == "Ambiguous Neutral Expression":
    suggestion = "สร้าง class 'Neutral' ให้ชัดเจน หรือใช้ threshold-based confidence แทน hard label"
else:
    suggestion = "พิจารณาใช้โมเดลที่เข้าใจบริบทลึกขึ้น เช่น transformer-based model"

print(f"\nSuggested improvement: {suggestion}")