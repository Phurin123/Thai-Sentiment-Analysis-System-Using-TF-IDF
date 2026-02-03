from fastapi import FastAPI, Request, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import joblib
import time
import numpy as np
import pandas as pd
from pathlib import Path

# ======================
# Setup FastAPI
# ======================
app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ======================
# Load Models
# ======================
VECTORIZER_A_PATH = MODELS_DIR / "vectorizer_20260203_161646_c03014a2.joblib"
MODEL_A_PATH = MODELS_DIR / "sentiment_model_20260203_161646_c03014a2.joblib"

VECTORIZER_B_PATH = MODELS_DIR / "vectorizer_20260203_161702_e63c055e.joblib"
MODEL_B_PATH = MODELS_DIR / "sentiment_model_20260203_161702_e63c055e.joblib"

for p in [
    VECTORIZER_A_PATH,
    MODEL_A_PATH,
    VECTORIZER_B_PATH,
    MODEL_B_PATH,
]:
    if not p.exists():
        raise FileNotFoundError(f"❌ ไม่พบไฟล์: {p}")

vectorizer_a = joblib.load(VECTORIZER_A_PATH)
classifier_a = joblib.load(MODEL_A_PATH)  # Logistic Regression

vectorizer_b = joblib.load(VECTORIZER_B_PATH)
classifier_b = joblib.load(MODEL_B_PATH)  # LinearSVC

MODEL_VERSION_A = "TF-IDF + Logistic Regression (Linear, Probabilistic)"
MODEL_VERSION_B = "TF-IDF + Linear SVM (Max-Margin)"


# ======================
# Helper: Global word sentiment (Model A only)
# ======================
def get_global_word_sentiment():
    coefs = classifier_a.coef_
    classes = classifier_a.classes_
    feature_names = vectorizer_a.get_feature_names_out()

    idx_to_sentiment = {}
    for i, cls in enumerate(classes):
        c = str(cls).lower()
        if c in ["positive", "pos", "ดี"]:
            idx_to_sentiment[i] = "positive"
        elif c in ["negative", "neg", "แย่", "ห่วย"]:
            idx_to_sentiment[i] = "negative"
        else:
            idx_to_sentiment[i] = "neutral"

    word_sentiment = {}
    for i, word in enumerate(feature_names):
        class_idx = int(np.argmax(coefs[:, i]))
        word_sentiment[word] = idx_to_sentiment.get(class_idx, "neutral")

    return word_sentiment


GLOBAL_WORD_SENTIMENT = get_global_word_sentiment()


# ======================
# Helper: Important words
# ======================
def get_important_words(text: str, vectorizer, classifier, top_k: int = 5):
    X = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()

    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(X)[0]
        class_idx = int(np.argmax(probs))
        coef = classifier.coef_[class_idx]
    else:
        coef = classifier.coef_[0]

    contributions = X.toarray()[0] * coef
    top_indices = np.argsort(np.abs(contributions))[::-1][:top_k]

    words, sentiments = [], []
    for i in top_indices:
        if contributions[i] == 0:
            continue
        word = feature_names[i]
        words.append(word)
        sentiments.append(GLOBAL_WORD_SENTIMENT.get(word, "neutral"))

    return words, sentiments


# ======================
# Routes
# ======================


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model/info")
def model_info():
    return {
        "model_a": {
            "name": "sentiment_lr",
            "version": MODEL_VERSION_A,
            "file": MODEL_A_PATH.name,
        },
        "model_b": {
            "name": "sentiment_linear_svm",
            "version": MODEL_VERSION_B,
            "file": MODEL_B_PATH.name,
        },
    }


@app.get("/errors", response_class=HTMLResponse)
def show_errors(request: Request):
    all_errors = []
    seen = set()  # ใช้สำหรับตรวจสอบซ้ำ

    # 1. Static errors (ไม่ซ้ำแน่นอน ไม่ต้อง dedup)
    try:
        errors_path = DATA_DIR / "error_examples.csv"
        if errors_path.exists():
            df_static = pd.read_csv(errors_path)
            for _, row in df_static.iterrows():
                text = str(row.get("text", "")).strip()
                true_label = str(row.get("true_label", "?"))
                pred_label = str(row.get("pred_label", "?"))
                key = f"{text}|{true_label}|{pred_label}"  # key สำหรับ static
                if key not in seen:
                    seen.add(key)
                    all_errors.append(
                        {
                            "text": text,
                            "true_label": true_label,
                            "pred_label": pred_label,
                            "source": "train_misclassified",
                        }
                    )
    except Exception as e:
        print(f"Error loading static errors: {e}")

    # 2. User feedback (deduplicate by text + model)
    try:
        feedback_path = DATA_DIR / "feedback_log.jsonl"
        if feedback_path.exists():
            with open(feedback_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # อ่านจากล่างขึ้นบน เพื่อเก็บรายการล่าสุดก่อน (ถ้ามีซ้ำ)
            for line in reversed(lines):
                if line.strip():
                    try:
                        fb = json.loads(line)
                        if fb.get("feedback") == "incorrect":
                            text = str(fb.get("text", "")).strip()
                            model = fb.get("model", "")
                            # สร้าง key สำหรับ dedup
                            key = f"{text}|{model}"

                            if key not in seen:
                                seen.add(key)
                                all_errors.append(
                                    {
                                        "text": text,
                                        "true_label": str(
                                            fb.get("true_label", "UNKNOWN")
                                        ),
                                        "pred_label": str(
                                            fb.get("predicted_label", "?")
                                        ),
                                        "source": "user_feedback",
                                        "model": model,
                                        "timestamp": fb.get("timestamp", ""),
                                    }
                                )
                    except Exception as ex:
                        print(f"Invalid feedback line: {ex}")
                        continue
    except Exception as e:
        print(f"Error loading feedback: {e}")

    # เรียงตาม timestamp (ใหม่ไปเก่า) — แต่ตอนนี้ all_errors มีแค่ unique แล้ว
    all_errors.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    errors_to_show = all_errors[:20]

    return templates.TemplateResponse(
        "errors.html", {"request": request, "errors": errors_to_show}
    )


# ======================
# Predict Endpoints
# ======================


@app.post("/predict")
def predict(text: str = Body(..., embed=True)):
    start = time.time()
    X = vectorizer_a.transform([text])
    pred = classifier_a.predict(X)[0]
    prob = float(np.max(classifier_a.predict_proba(X)[0]))
    latency = (time.time() - start) * 1000
    words, sents = get_important_words(text, vectorizer_a, classifier_a)

    return {
        "label": str(pred).upper(),
        "confidence": round(prob, 2),
        "latency_ms": round(latency, 2),
        "model": "TF-IDF + Logistic Regression",
        "version": MODEL_VERSION_A,
        "important_words": words,
        "word_sentiments": sents,
    }


@app.post("/predict-ab")
def predict_ab(text: str = Body(..., embed=True)):
    # Model A
    start_a = time.time()
    Xa = vectorizer_a.transform([text])
    pred_a = classifier_a.predict(Xa)[0]
    prob_a = float(np.max(classifier_a.predict_proba(Xa)[0]))
    latency_a = (time.time() - start_a) * 1000
    words_a, sents_a = get_important_words(text, vectorizer_a, classifier_a)

    # Model B
    start_b = time.time()
    Xb = vectorizer_b.transform([text])
    pred_b = classifier_b.predict(Xb)[0]
    score_b = classifier_b.decision_function(Xb)
    raw_score = float(score_b.ravel()[0])
    confidence_b = float(1 / (1 + np.exp(-abs(raw_score))))
    latency_b = (time.time() - start_b) * 1000
    words_b, sents_b = get_important_words(text, vectorizer_b, classifier_b)

    return {
        "model_a": {
            "label": str(pred_a).upper(),
            "confidence": round(prob_a, 2),
            "latency_ms": round(latency_a, 2),
            "model_name": "sentiment_lr",
            "version": MODEL_VERSION_A,
            "important_words": words_a,
            "word_sentiments": sents_a,
        },
        "model_b": {
            "label": str(pred_b).upper(),
            "confidence": round(confidence_b, 2),
            "latency_ms": round(latency_b, 2),
            "model_name": "sentiment_linear_svm",
            "version": MODEL_VERSION_B,
            "important_words": words_b,
            "word_sentiments": sents_b,
        },
    }


import json
from datetime import datetime

FEEDBACK_LOG_PATH = DATA_DIR / "feedback_log.jsonl"


@app.post("/feedback")
async def log_feedback(request: Request):
    try:
        data = await request.json()
        required_fields = ["text", "model", "predicted_label", "feedback"]
        if not all(field in data for field in required_fields):
            return JSONResponse({"error": "Missing required fields"}, status_code=400)

        # บันทึก true_label ถ้ามี (ไม่บังคับ)
        # ไม่ต้อง validate เพราะอาจไม่มี (กรณี 👍)

        if "timestamp" not in data:
            data["timestamp"] = datetime.utcnow().isoformat()

        with open(FEEDBACK_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

        return {"status": "success", "message": "Feedback recorded"}
    except Exception as e:
        print(f"Feedback error: {e}")
        return JSONResponse({"error": "Failed to log feedback"}, status_code=500)

import asyncio
import os
from datetime import datetime

FEEDBACK_LOG_PATH = DATA_DIR / "feedback_log.jsonl"

# === เพิ่มฟังก์ชันล้าง feedback ===
# แทนที่ฟังก์ชัน clear_feedback_periodically() ด้วย:
async def rotate_feedback_periodically(max_lines=100):
    while True:
        await asyncio.sleep(600)
        try:
            if FEEDBACK_LOG_PATH.exists():
                with open(FEEDBACK_LOG_PATH, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                
                if len(lines) > max_lines:
                    # เก็บแค่ max_lines รายการล่าสุด
                    recent_lines = lines[-max_lines:]
                    with open(FEEDBACK_LOG_PATH, "w", encoding="utf-8") as f:
                        f.writelines(recent_lines)
                    print(f"[{datetime.now()}] 🔄 ลดขนาด feedback เหลือ {max_lines} รายการ")
        except Exception as e:
            print(f"[{datetime.now()}] ❌ rotate feedback error: {e}")

# === เริ่ม background task เมื่อแอปเริ่มทำงาน ===
@app.on_event("startup")
async def start_background_tasks():
    asyncio.create_task(rotate_feedback_periodically())