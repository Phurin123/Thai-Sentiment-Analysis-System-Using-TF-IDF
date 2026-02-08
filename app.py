from fastapi import FastAPI, Request, Body, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import joblib
import time
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import asyncio

# ======================
# Setup FastAPI
# ======================
app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

DATA_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ======================
# Load Baseline Models (A/B and more)
# ======================

MODELS_REGRESS_DIR = BASE_DIR / "models_regress"
MODELS_LINEAR_DIR  = BASE_DIR / "models_linear"
MODELS_RF_DIR      = BASE_DIR / "models_tree"   # ⚠️ ใช้ models_tree ตามที่คุณเทรน
MODELS_NB_DIR      = BASE_DIR / "models_nb"
MODELS_XGB_DIR     = BASE_DIR / "models_xgb"
MODELS_LGBM_DIR    = BASE_DIR / "models_lgbm"
MODELS_ET_DIR      = BASE_DIR / "models_et"

# ===== Model A: Logistic Regression (default) =====
VECTORIZER_A_PATH = MODELS_REGRESS_DIR / "vectorizer_20260208_114252_968ddfe2.joblib"
MODEL_A_PATH      = MODELS_REGRESS_DIR / "sentiment_model_20260208_114252_968ddfe2.joblib"

for p in [VECTORIZER_A_PATH, MODEL_A_PATH]:
    if not p.exists():
        raise FileNotFoundError(f"❌ ไม่พบไฟล์ Baseline A: {p}")

vectorizer_a = joblib.load(VECTORIZER_A_PATH)
classifier_a = joblib.load(MODEL_A_PATH)

# ===== Helper: Load model + vectorizer pair =====
def load_model_pair(vec_path, model_path, name):
    if vec_path.exists() and model_path.exists():
        vec = joblib.load(vec_path)
        model = joblib.load(model_path)
        return vec, model, True
    else:
        print(f"⚠️ ไม่พบโมเดล: {name}")
        return None, None, False

# Define all models — ✅ อัปเดตชื่อไฟล์ให้ตรงกับที่คุณให้มา
model_configs = {
    "linear": {
        "vec": MODELS_LINEAR_DIR / "vectorizer_20260208_114518_03291797.joblib",
        "model": MODELS_LINEAR_DIR / "sentiment_model_20260208_114518_03291797.joblib",
        "name": "Linear SVM",
        "version": "TF-IDF + Linear SVM (Max-Margin)"
    },
    "rf": {
        "vec": MODELS_RF_DIR / "vectorizer_20260208_130453_6dd9ac36.joblib",
        "model": MODELS_RF_DIR / "sentiment_model_20260208_130453_6dd9ac36.joblib",
        "name": "Random Forest",
        "version": "TF-IDF + Random Forest"
    },
    "nb": {
        "vec": MODELS_NB_DIR / "vectorizer_20260208_132319_2e7b58b4.joblib",
        "model": MODELS_NB_DIR / "sentiment_model_20260208_132319_2e7b58b4.joblib",
        "name": "Naive Bayes",
        "version": "TF-IDF + Multinomial Naive Bayes"
    },
    "xgb": {
        "vec": MODELS_XGB_DIR / "vectorizer_20260208_132828_21da7a69.joblib",
        "model": MODELS_XGB_DIR / "sentiment_model_20260208_132828_21da7a69.joblib",
        "name": "XGBoost",
        "version": "TF-IDF + XGBoost Classifier"
    },
    "lgbm": {
        "vec": MODELS_LGBM_DIR / "vectorizer_20260208_132441_a70099f8.joblib",
        "model": MODELS_LGBM_DIR / "sentiment_model_20260208_132441_a70099f8.joblib",
        "name": "LightGBM",
        "version": "TF-IDF + LightGBM"
    },
    "et": {
        "vec": MODELS_ET_DIR / "vectorizer_20260208_132039_237644a6.joblib",
        "model": MODELS_ET_DIR / "sentiment_model_20260208_132039_237644a6.joblib",
        "name": "Extra Trees",
        "version": "TF-IDF + Extra Trees Classifier"
    }
}

# Load all optional models
loaded_models = {}
for key, cfg in model_configs.items():
    vec, model, ok = load_model_pair(cfg["vec"], cfg["model"], cfg["name"])
    if ok:
        loaded_models[key] = {
            "vectorizer": vec,
            "classifier": model,
            "name": cfg["name"],
            "version": cfg["version"]
        }

# ======================
# Load BERT Model
# ======================
BERT_MODEL_LOADED = False
bert_tokenizer = None
bert_model = None
torch = None

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    BERT_PATH = MODELS_DIR / "bert_thai_sentiment"
    if BERT_PATH.exists():
        bert_tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
        bert_model = AutoModelForSequenceClassification.from_pretrained(
            BERT_PATH,
            num_labels=3
        )
        bert_model.eval()
        BERT_MODEL_LOADED = True
        print("✅ โหลด BERT สำเร็จ (3-class)")
    else:
        print("⚠️ ไม่พบโฟลเดอร์ BERT ที่ models/bert_thai_sentiment")
except Exception as e:
    print(f"❌ โหลด BERT ไม่ได้: {e}")
    BERT_MODEL_LOADED = False

# ======================
# Global word sentiment (Model A only)
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
# Label mapping helper — ✅ เพิ่มใหม่!
# ======================
LABEL_MAP = {
    0: "NEGATIVE",
    1: "NEUTRAL",
    2: "POSITIVE",
    "0": "NEGATIVE",
    "1": "NEUTRAL",
    "2": "POSITIVE",
}

def normalize_label(label):
    """แปลง label ใดๆ → 'NEGATIVE' / 'NEUTRAL' / 'POSITIVE'"""
    if isinstance(label, str):
        label_lower = label.lower()
        if label_lower in ["negative", "neg", "แย่", "ห่วย", "ลบ", "0"]:
            return "NEGATIVE"
        elif label_lower in ["neutral", "neu", "กลาง", "เฉยๆ", "กลางๆ", "1"]:
            return "NEUTRAL"
        elif label_lower in ["positive", "pos", "ดี", "บวก", "ชอบ", "2"]:
            return "POSITIVE"
        else:
            return LABEL_MAP.get(label, label.upper())
    else:
        return LABEL_MAP.get(label, str(label).upper())

# ======================
# Helper: Important words (Baseline models) — ✅ แก้ไขแล้ว!
# ======================
def get_important_words(text: str, vectorizer, classifier, top_k: int = 5):
    X = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()

    # 🌳 1. Tree-based models: RF, XGBoost, LightGBM, Extra Trees
    if hasattr(classifier, "feature_importances_"):
        imp = classifier.feature_importances_
        present = X.toarray()[0]
        contributions = present * imp
        top_indices = np.argsort(contributions)[::-1][:top_k]
        words = [feature_names[i] for i in top_indices if contributions[i] > 0]
        sents = [GLOBAL_WORD_SENTIMENT.get(w, "neutral") for w in words]
        return words[:top_k], sents[:top_k]

    # 📈 2. Linear models: LogisticRegression, LinearSVC, Naive Bayes
    elif hasattr(classifier, "coef_"):
        if hasattr(classifier, "predict_proba"):
            probs = classifier.predict_proba(X)[0]
            class_idx = int(np.argmax(probs))
            if classifier.coef_.shape[0] > 1:
                coef = classifier.coef_[class_idx]
            else:
                coef = classifier.coef_[0]
        else:
            if classifier.coef_.shape[0] == 1:
                coef = classifier.coef_[0]
            else:
                decision = classifier.decision_function(X)
                class_idx = int(np.argmax(decision))
                coef = classifier.coef_[class_idx]

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

    # ❓ 3. Fallback
    else:
        return [], []

# ======================
# BERT + LIME Explanation
# ======================
def bert_predict_proba(texts):
    inputs = bert_tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
        return_attention_mask=True
    )
    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
    return probs

def get_bert_important_words(text: str, top_k: int = 5):
    if not BERT_MODEL_LOADED:
        return [], []

    try:
        import lime
        import lime.lime_text

        explainer = lime.lime_text.LimeTextExplainer(
            class_names=["NEGATIVE", "NEUTRAL", "POSITIVE"],
            verbose=False
        )

        exp = explainer.explain_instance(
            text,
            bert_predict_proba,
            num_features=top_k,
            num_samples=200,
            labels=[0, 1, 2]
        )

        pred_proba = bert_predict_proba([text])[0]
        pred_label_idx = int(np.argmax(pred_proba))

        word_weights = exp.as_list(label=pred_label_idx)
        words = []
        sentiments = []

        for word, weight in word_weights[:top_k]:
            words.append(word)
            if pred_label_idx == 0:  # NEGATIVE
                s = "negative" if weight > 0 else "positive"
            elif pred_label_idx == 2:  # POSITIVE
                s = "positive" if weight > 0 else "negative"
            else:  # NEUTRAL
                s = "neutral"
            sentiments.append(s)

        return words, sentiments

    except Exception as e:
        print(f"LIME error: {e}")
        return [], []

# ======================
# Routes
# ======================

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health():
    return {
        "status": "ok",
        "baseline_a": True,
        "available_models": list(loaded_models.keys()) + (["bert"] if BERT_MODEL_LOADED else []),
        "bert": BERT_MODEL_LOADED,
    }

@app.get("/model/info")
def model_info():
    info = {
        "model_a": {
            "name": "sentiment_lr",
            "version": "TF-IDF + Logistic Regression (Linear, Probabilistic)",
            "file": MODEL_A_PATH.name,
        }
    }
    for key, mdl in loaded_models.items():
        info[key] = {
            "name": mdl["name"],
            "version": mdl["version"],
        }
    if BERT_MODEL_LOADED:
        info["bert"] = {
            "name": "Thai BERT (wangchanberta)",
            "path": "models/bert_thai_sentiment",
        }
    return info

@app.get("/errors", response_class=HTMLResponse)
def show_errors(request: Request):
    all_errors = []
    seen = set()

    try:
        errors_path = DATA_DIR / "error_examples.csv"
        if errors_path.exists():
            df_static = pd.read_csv(errors_path)
            for _, row in df_static.iterrows():
                text = str(row.get("text", "")).strip()
                true_label = str(row.get("true_label", "?"))
                pred_label = str(row.get("pred_label", "?"))
                key = f"{text}|{true_label}|{pred_label}"
                if key not in seen:
                    seen.add(key)
                    all_errors.append({
                        "text": text,
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "source": "train_misclassified",
                    })
    except Exception as e:
        print(f"Error loading static errors: {e}")

    try:
        feedback_path = DATA_DIR / "feedback_log.jsonl"
        if feedback_path.exists():
            with open(feedback_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in reversed(lines):
                if line.strip():
                    try:
                        fb = json.loads(line)
                        if fb.get("feedback") == "incorrect":
                            text = str(fb.get("text", "")).strip()
                            model = fb.get("model", "")
                            key = f"{text}|{model}"
                            if key not in seen:
                                seen.add(key)
                                all_errors.append({
                                    "text": text,
                                    "true_label": str(fb.get("true_label", "UNKNOWN")),
                                    "pred_label": str(fb.get("predicted_label", "?")),
                                    "source": "user_feedback",
                                    "model": model,
                                    "timestamp": fb.get("timestamp", ""),
                                })
                    except Exception as ex:
                        print(f"Invalid feedback line: {ex}")
                        continue
    except Exception as e:
        print(f"Error loading feedback: {e}")

    all_errors.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    errors_to_show = all_errors[:20]

    return templates.TemplateResponse(
        "errors.html", {"request": request, "errors": errors_to_show}
    )

# ======================
# Predict Endpoints — ✅ แก้ไขให้ใช้ normalize_label()
# ======================

@app.post("/predict")
def predict(text: str = Body(..., embed=True)):
    start = time.time()
    X = vectorizer_a.transform([text])
    pred_raw = classifier_a.predict(X)[0]
    pred = normalize_label(pred_raw)  # ✅ แปลง label
    prob = float(np.max(classifier_a.predict_proba(X)[0]))
    latency = (time.time() - start) * 1000
    words, sents = get_important_words(text, vectorizer_a, classifier_a)

    return {
        "label": pred,
        "confidence": round(prob, 2),
        "latency_ms": round(latency, 2),
        "model": "sentiment_lr",
        "version": "TF-IDF + Logistic Regression (Linear, Probabilistic)",
        "important_words": words,
        "word_sentiments": sents,
    }

@app.post("/predict-ab")
def predict_ab(
    text: str = Body(..., embed=True),
    model_b_type: str = Body("linear", embed=True)
):
    # ===== Model A =====
    start_a = time.time()
    Xa = vectorizer_a.transform([text])
    pred_a_raw = classifier_a.predict(Xa)[0]
    pred_a = normalize_label(pred_a_raw)
    prob_a = float(np.max(classifier_a.predict_proba(Xa)[0]))
    latency_a = (time.time() - start_a) * 1000
    words_a, sents_a = get_important_words(text, vectorizer_a, classifier_a)

    result = {
        "model_a": {
            "label": pred_a,
            "confidence": round(prob_a, 2),
            "latency_ms": round(latency_a, 2),
            "model_name": "sentiment_lr",
            "version": "TF-IDF + Logistic Regression",
            "important_words": words_a,
            "word_sentiments": sents_a,
        },
        "model_b": None
    }

    # ===== Model B =====
    if model_b_type == "bert" and BERT_MODEL_LOADED:
        start_b = time.time()
        inputs = bert_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
            return_attention_mask=True
        )
        with torch.no_grad():
            outputs = bert_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_idx].item()

        latency_b = (time.time() - start_b) * 1000
        words_b, sents_b = get_bert_important_words(text, top_k=5)
        
        if hasattr(bert_model.config, 'id2label'):
            raw_label = bert_model.config.id2label[pred_idx].lower()
            label_b = normalize_label(raw_label)
        else:
            id2label = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
            label_b = id2label.get(pred_idx, "UNKNOWN")

        result["model_b"] = {
            "label": label_b,
            "confidence": round(confidence, 2),
            "latency_ms": round(latency_b, 2),
            "model_name": "Thai BERT",
            "version": "wangchanberta + LIME",
            "important_words": words_b,
            "word_sentiments": sents_b,
        }

    elif model_b_type in loaded_models:
        mdl = loaded_models[model_b_type]
        start_b = time.time()
        Xb = mdl["vectorizer"].transform([text])
        pred_b_raw = mdl["classifier"].predict(Xb)[0]
        pred_b = normalize_label(pred_b_raw)  # ✅ แปลง label

        # Confidence logic
        if hasattr(mdl["classifier"], "predict_proba"):
            prob_b = float(np.max(mdl["classifier"].predict_proba(Xb)[0]))
        elif hasattr(mdl["classifier"], "decision_function"):
            score = mdl["classifier"].decision_function(Xb)
            prob_b = float(1 / (1 + np.exp(-abs(score.ravel()[0]))))
        else:
            prob_b = 0.95  # fallback

        latency_b = (time.time() - start_b) * 1000
        words_b, sents_b = get_important_words(text, mdl["vectorizer"], mdl["classifier"])

        result["model_b"] = {
            "label": pred_b,
            "confidence": round(prob_b, 2),
            "latency_ms": round(latency_b, 2),
            "model_name": mdl["name"],
            "version": mdl["version"],
            "important_words": words_b,
            "word_sentiments": sents_b,
        }

    else:
        available = list(loaded_models.keys()) + (["bert"] if BERT_MODEL_LOADED else [])
        raise HTTPException(
            status_code=400,
            detail=f"Model B type '{model_b_type}' not available. Available: {available}"
        )

    return result

@app.post("/predict-bert")
def predict_bert(text: str = Body(..., embed=True)):
    if not BERT_MODEL_LOADED:
        raise HTTPException(status_code=500, detail="BERT model not available")

    start = time.time()
    try:
        inputs = bert_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
            return_attention_mask=True
        )

        with torch.no_grad():
            outputs = bert_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_idx].item()

        latency = (time.time() - start) * 1000
        words, sents = get_bert_important_words(text, top_k=5)

        if hasattr(bert_model.config, 'id2label'):
            raw_label = bert_model.config.id2label[pred_idx].lower()
            label = normalize_label(raw_label)
        else:
            id2label = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
            label = id2label.get(pred_idx, "UNKNOWN")

        return {
            "label": label,
            "confidence": round(confidence, 2),
            "latency_ms": round(latency, 2),
            "model": "Thai BERT (wangchanberta + LIME)",
            "important_words": words,
            "word_sentiments": sents,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BERT prediction error: {str(e)}")

# ======================
# Feedback Logging
# ======================
FEEDBACK_LOG_PATH = DATA_DIR / "feedback_log.jsonl"

@app.post("/feedback")
async def log_feedback(request: Request):
    try:
        data = await request.json()
        required_fields = ["text", "model", "predicted_label", "feedback"]
        if not all(field in data for field in required_fields):
            return JSONResponse({"error": "Missing required fields"}, status_code=400)

        if "timestamp" not in data:
            data["timestamp"] = datetime.utcnow().isoformat()

        with open(FEEDBACK_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

        return {"status": "success", "message": "Feedback recorded"}
    except Exception as e:
        print(f"Feedback error: {e}")
        return JSONResponse({"error": "Failed to log feedback"}, status_code=500)

# ======================
# Background Task
# ======================
async def rotate_feedback_periodically(max_lines=100):
    while True:
        await asyncio.sleep(600)
        try:
            if FEEDBACK_LOG_PATH.exists():
                with open(FEEDBACK_LOG_PATH, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                if len(lines) > max_lines:
                    recent_lines = lines[-max_lines:]
                    with open(FEEDBACK_LOG_PATH, "w", encoding="utf-8") as f:
                        f.writelines(recent_lines)
                    print(f"[{datetime.now()}] 🔄 ลดขนาด feedback เหลือ {max_lines} รายการ")
        except Exception as e:
            print(f"[{datetime.now()}] ❌ rotate feedback error: {e}")

@app.on_event("startup")
async def start_background_tasks():
    asyncio.create_task(rotate_feedback_periodically())