from fastapi import FastAPI, Request, Body
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import time
import numpy as np
import os

# ======================
# Setup FastAPI
# ======================
app = FastAPI()

TEMPLATES_DIR = "templates"
STATIC_DIR = "static"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ======================
# Load Model
# ======================
MODEL_PATH = r"C:\Users\lovew\OneDrive\เอกสาร\datasicene-miniproject\sentiment_tfidf_lr_20260124_084230.joblib"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ ไม่พบไฟล์โมเดลที่: {MODEL_PATH}")

pipeline = joblib.load(MODEL_PATH)
vectorizer = pipeline.named_steps["features"]
classifier = pipeline.named_steps["clf"]

MODEL_VERSION = "1.0-baseline"


# ======================
# Helper: Get global sentiment of each feature
# ======================
def get_global_word_sentiment():
    """
    สร้าง mapping: feature_name -> 'positive' / 'neutral' / 'negative'
    โดยดูจาก class ที่มี coefficient สูงสุด
    """
    coefs = classifier.coef_  # shape: (n_classes, n_features)

    # ตรวจสอบลำดับ class (สำคัญมาก!)
    classes = classifier.classes_
    print("Model classes:", classes)  # เช่น ['negative', 'neutral', 'positive']

    # สร้าง mapping จาก class index → sentiment label
    idx_to_sentiment = {}
    for i, cls in enumerate(classes):
        if cls.lower() in ["positive", "pos", "ดี"]:
            idx_to_sentiment[i] = "positive"
        elif cls.lower() in ["negative", "neg", "แย่", "ห่วย"]:
            idx_to_sentiment[i] = "negative"
        else:
            idx_to_sentiment[i] = "neutral"

    # ดึงชื่อฟีเจอร์
    feature_names = []
    for name, trans in vectorizer.transformer_list:
        if hasattr(trans, "get_feature_names_out"):
            feature_names.extend(trans.get_feature_names_out())
        else:
            feature_names.extend(trans.get_feature_names())

    # หา sentiment ของแต่ละ feature
    word_sentiment = {}
    for i, name in enumerate(feature_names):
        # หา class ที่มี weight สูงสุดสำหรับ feature นี้
        class_idx = int(np.argmax(coefs[:, i]))
        sent = idx_to_sentiment.get(class_idx, "neutral")
        word_sentiment[name] = sent

    return word_sentiment


# สร้าง global mapping ตอนเริ่มโปรแกรม
GLOBAL_WORD_SENTIMENT = get_global_word_sentiment()


# ======================
# Helper: Get Important Words
# ======================
def get_important_words(text: str, top_k: int = 5):
    """
    ดึงคำสำคัญที่มีอิทธิพลต่อ prediction ของข้อความนี้
    และกำหนด sentiment ตาม global meaning ของคำ
    """
    X = vectorizer.transform([text])
    pred_proba = classifier.predict_proba(X)[0]
    pred_class_idx = int(np.argmax(pred_proba))
    coef = classifier.coef_[pred_class_idx]

    contributions = X.toarray()[0] * coef

    feature_names = []
    for name, trans in vectorizer.transformer_list:
        if hasattr(trans, "get_feature_names_out"):
            feature_names.extend(trans.get_feature_names_out())
        else:
            feature_names.extend(trans.get_feature_names())

    top_indices = np.argsort(np.abs(contributions))[::-1][:top_k]

    words = []
    sentiments = []
    for i in top_indices:
        contrib = contributions[i]
        if contrib == 0:
            continue
        word = feature_names[i]
        words.append(word)
        # ใช้ global sentiment แทนการดู sign ของ contribution
        sent = GLOBAL_WORD_SENTIMENT.get(word, "neutral")
        sentiments.append(sent)

    return words, sentiments


# ======================
# Routes
# ======================
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
def predict(text: str = Body(..., embed=True)):
    start = time.time()
    pred = pipeline.predict([text])[0]
    prob = max(pipeline.predict_proba([text])[0])
    latency = (time.time() - start) * 1000

    words, word_sentiments = get_important_words(text, top_k=5)

    return {
        "label": pred.upper(),
        "confidence": round(float(prob), 2),
        "latency_ms": round(latency, 2),
        "model": "TF-IDF + Logistic Regression",
        "version": MODEL_VERSION,
        "important_words": words,
        "word_sentiments": word_sentiments,
    }


@app.post("/predict-ab")
def predict_ab(text: str = Body(..., embed=True)):
    start = time.time()
    pred_a = pipeline.predict([text])[0]
    prob_a = max(pipeline.predict_proba([text])[0])
    latency_a = (time.time() - start) * 1000

    # จำลอง Model B
    pred_b = pred_a
    prob_b = min(1.0, prob_a + 0.02) if prob_a < 0.98 else prob_a - 0.02
    latency_b = latency_a + 1.5

    words, word_sentiments = get_important_words(text, top_k=5)

    return {
        "model_a": {
            "label": pred_a.upper(),
            "confidence": round(float(prob_a), 2),
            "latency_ms": round(latency_a, 2),
            "model_name": "sentiment_v1_tfidf_lr",
            "version": "v1.0",
            "important_words": words,
            "word_sentiments": word_sentiments,
        },
        "model_b": {
            "label": pred_b.upper(),
            "confidence": round(float(prob_b), 2),
            "latency_ms": round(latency_b, 2),
            "model_name": "sentiment_v2_char_tfidf_lr",
            "version": "v2.1",
            "important_words": words,
            "word_sentiments": word_sentiments,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
