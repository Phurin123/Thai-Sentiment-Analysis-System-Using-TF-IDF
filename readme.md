# Thai Sentiment Analysis System Using TF-IDF

ระบบวิเคราะห์ความรู้สึกภาษาไทยแบบ Multi-Model โดยใช้ **TF-IDF** และ **BERT** พร้อม Web UI สำหรับเปรียบเทียบประสิทธิภาพของโมเดลต่างๆ แบบ A/B Testing

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 สารบัญ

- [ภาพรวมโปรเจค](#ภาพรวมโปรเจค)
- [ฟีเจอร์หลัก](#ฟีเจอร์หลัก)
- [เทคโนโลยีที่ใช้](#เทคโนโลยีที่ใช้)
- [โครงสร้างโปรเจค](#โครงสร้างโปรเจค)
- [การติดตั้ง](#การติดตั้ง)
- [วิธีรันโปรเจค](#วิธีรันโปรเจค)
- [การเทรนโมเดล](#การเทรนโมเดล)
- [API Documentation](#api-documentation)
- [วิธี Deploy Production](#วิธี-deploy-production)
- [การใช้งาน Web UI](#การใช้งาน-web-ui)
- [โมเดลที่รองรับ](#โมเดลที่รองรับ)

---

## 🎯 ภาพรวมโปรเจค

ระบบนี้พัฒนาขึ้นเพื่อวิเคราะห์ความรู้สึก (Sentiment Analysis) ของข้อความภาษาไทย โดยจำแนกออกเป็น 3 ประเภท:
- **POSITIVE** (บวก) 😊
- **NEGATIVE** (ลบ) 😠
- **NEUTRAL** (กลาง) 😐

ระบบรองรับการเปรียบเทียบผลลัพธ์จากหลายโมเดล ML พร้อมกัน (A/B Testing) และมีระบบ Feedback เพื่อปรับปรุงความแม่นยำ

---

## ✨ ฟีเจอร์หลัก

✅ **Multi-Model Support**: รองรับ 7 โมเดล TF-IDF และ 1 โมเดล BERT  
✅ **A/B Testing UI**: เปรียบเทียบประสิทธิภาพของโมเดลต่างๆ ในหน้าเดียว  
✅ **Explainable AI**: แสดงคำสำคัญที่มีอิทธิพลต่อการทำนาย (Important Words)  
✅ **Feedback System**: รวบรวม feedback จากผู้ใช้เพื่อปรับปรุงโมเดล  
✅ **Error Tracking**: ติดตามและแสดงข้อผิดพลาดที่เกิดขึ้นจากการทำนาย  
✅ **RESTful API**: API endpoints สำหรับการ integrate กับระบบอื่น  
✅ **Real-time Analysis**: วิเคราะห์ความรู้สึกแบบ real-time พร้อมแสดงค่า latency

---

## 🛠️ เทคโนโลยีที่ใช้

### Backend & ML
- **FastAPI** - Modern web framework สำหรับสร้าง API
- **Uvicorn** - ASGI server สำหรับรัน FastAPI
- **scikit-learn** - Machine learning library สำหรับโมเดล TF-IDF
- **XGBoost** - Gradient boosting framework
- **LightGBM** - Gradient boosting framework จาก Microsoft
- **transformers** - Hugging Face library สำหรับ BERT model
- **PyTorch** - Deep learning framework
- **pythainlp** - Thai NLP library
- **LIME** - Explainable AI library สำหรับอธิบายการทำนาย

### Frontend
- **Bootstrap 5** - CSS framework
- **Vanilla JavaScript** - ไม่ใช้ framework เพิ่มเติม

### Data Processing
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **joblib** - Model serialization

---

## 📁 โครงสร้างโปรเจค

```
Thai-Sentiment-Analysis-System-Using-TF-IDF/
│
├── app.py                          # 🚀 FastAPI main application
├── requirements.txt                # 📦 Python dependencies
├── information.txt                 # ℹ️ Quick start info
│
├── data/                           # 📊 Training datasets
│   ├── 1.synthetic_wisesight_like_thai_sentiment_5000.csv
│   └── 5.ultimate_sentiment_100k.csv
│
├── models_regress/                 # 🤖 Logistic Regression models
├── models_linear/                  # 🤖 Linear SVM models
├── models_tree/                    # 🌳 Random Forest models
├── models_nb/                      # 🤖 Naive Bayes models
├── models_xgb/                     # 🚀 XGBoost models
├── models_lgbm/                    # 💡 LightGBM models
├── models_et/                      # 🌲 Extra Trees models
├── models/bert_thai_sentiment/     # 🧠 BERT model (optional)
│
├── templates/                      # 🎨 HTML templates
│   ├── index.html                  # Main UI page
│   └── errors.html                 # Error tracking page
│
├── static/                         # 🎨 Static files
│   └── style.css
│
└── Training Scripts:
    ├── Regress_train.py            # เทรน Logistic Regression
    ├── Renear_train.py             # เทรน Linear SVM
    ├── Random Forest_train.py      # เทรน Random Forest
    ├── naivebay                    # เทรน Naive Bayes
    ├── xgboots.py                  # เทรน XGBoost
    ├── lightbgm.py                 # เทรน LightGBM
    ├── extratree.py                # เทรน Extra Trees
    └── bert.py                     # เทรน BERT (optional)
```

---

## 📥 การติดตั้ง

### ความต้องการของระบบ

- **Python** 3.8 หรือสูงกว่า
- **pip** (Python package manager)
- **Virtual Environment** (แนะนำ)
- **RAM**: อย่างน้อย 4GB (8GB+ สำหรับ BERT)
- **Disk Space**: อย่างน้อย 2GB

### ขั้นตอนการติดตั้ง

#### 1. Clone โปรเจค

```bash
git clone https://github.com/Phurin123/Thai-Sentiment-Analysis-System-Using-TF-IDF.git
cd Thai-Sentiment-Analysis-System-Using-TF-IDF
```

#### 2. สร้าง Virtual Environment (แนะนำ)

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 3. ติดตั้ง Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **หมายเหตุ**: การติดตั้ง `transformers` และ `torch` อาจใช้เวลานาน ขึ้นอยู่กับความเร็วอินเทอร์เน็ต

#### 4. เตรียมข้อมูลและโมเดล

ตรวจสอบว่ามีโฟลเดอร์โมเดลและไฟล์ที่จำเป็น:

```
models_regress/
  ├── vectorizer_*.joblib
  └── sentiment_model_*.joblib

models_linear/
models_tree/
models_nb/
models_xgb/
models_lgbm/
models_et/
```

> 💡 **คำแนะนำ**: ถ้ายังไม่มีโมเดล ให้รันสคริปต์เทรนโมเดลก่อน (ดูในส่วน [การเทรนโมเดล](#การเทรนโมเดล))

---

## 🚀 วิธีรันโปรเจค

### รัน Development Server

เปิด terminal ที่โฟลเดอร์โปรเจคและรันคำสั่ง:

```bash
uvicorn app:app --reload
```

**หรือ** (ตามที่ระบุใน `information.txt`):

```bash
uvicorn app:app --reload
```

### เข้าถึงเว็บแอป

เปิดเบราว์เซอร์และไปที่:

```
http://127.0.0.1:8000/
```

### ตัวเลือกการรันเพิ่มเติม

#### กำหนด Port และ Host

```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

#### รันโหมด Production (ไม่มี --reload)

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

#### ตรวจสอบสถานะระบบ

เข้าไปที่:
```
http://127.0.0.1:8000/health
```

จะได้ response:
```json
{
  "status": "ok",
  "baseline_a": true,
  "available_models": ["linear", "rf", "nb", "xgb", "lgbm", "et", "bert"],
  "bert": true
}
```

---

## 🎓 การเทรนโมเดล

### ข้อมูลสำหรับการเทรน

โปรเจคนี้ใช้ dataset จาก `data/5.ultimate_sentiment_100k.csv` (100,000 รายการ)

รูปแบบข้อมูล:
```csv
text,sentiment
"สินค้าดีมาก ส่งไว","POSITIVE"
"แย่มาก ไม่ตรงปก","NEGATIVE"
"โอเคนะ ใช้ได้","NEUTRAL"
```

### วิธีเทรนโมเดลแต่ละประเภท

#### 1. Logistic Regression (Model A - Baseline)

```bash
python Regress_train.py
```

**Output:**
- โมเดล: `models_regress/sentiment_model_*.joblib`
- Vectorizer: `models_regress/vectorizer_*.joblib`
- Evaluation: `results_regress/evaluation_*.png`

#### 2. Linear SVM

```bash
python Renear_train.py
```

#### 3. Random Forest

```bash
python "Random Forest_train.py"
```

#### 4. Naive Bayes

```bash
python naivebay
```

#### 5. XGBoost

```bash
python xgboots.py
```

#### 6. LightGBM

```bash
python lightbgm.py
```

#### 7. Extra Trees

```bash
python extratree.py
```

#### 8. BERT (Optional - ใช้ GPU แนะนำ)

```bash
python bert.py
```

> ⚠️ **คำเตือน**: การเทรน BERT ต้องการ GPU และ RAM สูง (8GB+)

### โครงสร้างการเทรน

แต่ละสคริปต์จะ:
1. โหลดและ preprocess ข้อมูล
2. Split train/test (80/20)
3. เทรนโมเดลด้วย TF-IDF vectorizer
4. ประเมินผล (Accuracy, F1-Score, Confusion Matrix)
5. บันทึกโมเดลพร้อม UID สำหรับ version control
6. บันทึก misclassified examples สำหรับการวิเคราะห์

---

## 📡 API Documentation

### Base URL

```
http://127.0.0.1:8000
```

### Endpoints

#### 1. **GET** `/` - หน้า Web UI หลัก

**Description**: แสดงหน้าเว็บสำหรับทดสอบโมเดล

**Response**: HTML page

---

#### 2. **POST** `/predict` - ทำนายด้วย Model A (Logistic Regression)

**Description**: ทำนายความรู้สึกด้วยโมเดล baseline

**Request Body:**
```json
{
  "text": "สินค้าดีมาก ประทับใจ 😊"
}
```

**Response:**
```json
{
  "label": "POSITIVE",
  "confidence": 0.95,
  "latency_ms": 12.34,
  "model": "sentiment_lr",
  "version": "TF-IDF + Logistic Regression (Linear, Probabilistic)",
  "important_words": ["ดีมาก", "ประทับใจ"],
  "word_sentiments": ["positive", "positive"]
}
```

**cURL Example:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"สินค้าดีมาก ประทับใจ"}'
```

---

#### 3. **POST** `/predict-ab` - เปรียบเทียบ Model A และ Model B

**Description**: ทำนายด้วย Model A และ Model B พร้อมกัน

**Request Body:**
```json
{
  "text": "สินค้าแย่มาก ผิดหวัง",
  "model_b_type": "bert"
}
```

**model_b_type options:**
- `"linear"` - Linear SVM
- `"rf"` - Random Forest
- `"nb"` - Naive Bayes
- `"xgb"` - XGBoost
- `"lgbm"` - LightGBM
- `"et"` - Extra Trees
- `"bert"` - Thai BERT (WangChanBERTa)

**Response:**
```json
{
  "model_a": {
    "label": "NEGATIVE",
    "confidence": 0.92,
    "latency_ms": 8.5,
    "model_name": "sentiment_lr",
    "version": "TF-IDF + Logistic Regression",
    "important_words": ["แย่มาก", "ผิดหวัง"],
    "word_sentiments": ["negative", "negative"]
  },
  "model_b": {
    "label": "NEGATIVE",
    "confidence": 0.98,
    "latency_ms": 156.7,
    "model_name": "Thai BERT",
    "version": "wangchanberta + LIME",
    "important_words": ["แย่", "ผิดหวัง"],
    "word_sentiments": ["negative", "negative"]
  }
}
```

---

#### 4. **POST** `/predict-bert` - ทำนายด้วย BERT เท่านั้น

**Description**: ทำนายด้วย Thai BERT model

**Request Body:**
```json
{
  "text": "สินค้าโอเคนะ ไม่ได้ดีหรือแย่"
}
```

**Response:**
```json
{
  "label": "NEUTRAL",
  "confidence": 0.87,
  "latency_ms": 145.2,
  "model": "Thai BERT (wangchanberta + LIME)",
  "important_words": ["โอเค", "ไม่ได้"],
  "word_sentiments": ["neutral", "neutral"]
}
```

---

#### 5. **POST** `/feedback` - ส่ง Feedback

**Description**: บันทึก feedback จากผู้ใช้เพื่อปรับปรุงโมเดล

**Request Body:**
```json
{
  "text": "สินค้าดีมาก",
  "model": "model_a",
  "predicted_label": "POSITIVE",
  "feedback": "correct",
  "true_label": "POSITIVE",
  "confidence": 0.95,
  "model_name": "sentiment_lr",
  "timestamp": "2026-02-09T18:50:00"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Feedback recorded"
}
```

---

#### 6. **GET** `/errors` - ดูข้อผิดพลาด

**Description**: แสดงหน้ารายการข้อผิดพลาดจากการทำนาย

**Response**: HTML page แสดง 20 ข้อผิดพลาดล่าสุด

---

#### 7. **GET** `/health` - ตรวจสอบสถานะระบบ

**Description**: ตรวจสอบว่าระบบทำงานปกติหรือไม่

**Response:**
```json
{
  "status": "ok",
  "baseline_a": true,
  "available_models": ["linear", "rf", "nb", "xgb", "lgbm", "et", "bert"],
  "bert": true
}
```

---

#### 8. **GET** `/model/info` - ดูข้อมูลโมเดล

**Description**: แสดงข้อมูลโมเดลทั้งหมดที่โหลดไว้

**Response:**
```json
{
  "model_a": {
    "name": "sentiment_lr",
    "version": "TF-IDF + Logistic Regression",
    "file": "sentiment_model_20260208_114252_968ddfe2.joblib"
  },
  "linear": {
    "name": "Linear SVM",
    "version": "TF-IDF + Linear SVM (Max-Margin)"
  },
  "bert": {
    "name": "Thai BERT (wangchanberta)",
    "path": "models/bert_thai_sentiment"
  }
}
```

---

## 🚢 วิธี Deploy Production

### Option 1: Deploy ด้วย Uvicorn + Systemd (Linux)

#### 1. สร้าง systemd service file

```bash
sudo nano /etc/systemd/system/thai-sentiment.service
```

เพิ่มเนื้อหา:

```ini
[Unit]
Description=Thai Sentiment Analysis API
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/Thai-Sentiment-Analysis-System-Using-TF-IDF
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always

[Install]
WantedBy=multi-user.target
```

#### 2. Enable และ Start service

```bash
sudo systemctl daemon-reload
sudo systemctl enable thai-sentiment
sudo systemctl start thai-sentiment
sudo systemctl status thai-sentiment
```

---

### Option 2: Deploy ด้วย Gunicorn + Uvicorn Workers

#### 1. ติดตั้ง Gunicorn

```bash
pip install gunicorn
```

#### 2. รันด้วย Gunicorn

```bash
gunicorn app:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --log-level info
```

**คำอธิบาย parameters:**
- `--workers 4`: จำนวน worker processes (แนะนำ: 2-4 x CPU cores)
- `--worker-class`: ใช้ UvicornWorker สำหรับ async support
- `--timeout 120`: timeout สำหรับ BERT model (ต้องการเวลานาน)

---

### Option 3: Deploy ด้วย Docker

#### 1. สร้าง Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

#### 2. สร้าง .dockerignore

```
venv/
__pycache__/
*.pyc
.git/
.gitignore
results_*/
*.log
```

#### 3. Build และ Run

```bash
# Build image
docker build -t thai-sentiment-api .

# Run container
docker run -d \
  --name thai-sentiment \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  thai-sentiment-api
```

#### 4. ใช้ Docker Compose (แนะนำ)

สร้าง `docker-compose.yml`:

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
      - ./models_regress:/app/models_regress
      - ./models_linear:/app/models_linear
      - ./models_tree:/app/models_tree
      - ./models_nb:/app/models_nb
      - ./models_xgb:/app/models_xgb
      - ./models_lgbm:/app/models_lgbm
      - ./models_et:/app/models_et
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
```

รันด้วย:

```bash
docker-compose up -d
```

---

### Option 4: Deploy บน Cloud Platform

#### Heroku

```bash
# สร้าง Procfile
echo "web: uvicorn app:app --host 0.0.0.0 --port \$PORT" > Procfile

# Deploy
heroku create thai-sentiment-api
git push heroku main
```

#### Google Cloud Run

```bash
gcloud run deploy thai-sentiment-api \
  --source . \
  --platform managed \
  --region asia-southeast1 \
  --allow-unauthenticated
```

#### AWS EC2

1. Launch EC2 instance (Ubuntu 22.04)
2. SSH เข้า instance
3. ติดตั้ง Python และ dependencies
4. ใช้ systemd หรือ Docker ตามด้านบน
5. ตั้งค่า Security Group เปิด port 8000

---

### เพิ่มประสิทธิภาพ Production

#### 1. ใช้ NGINX เป็น Reverse Proxy

สร้าง `/etc/nginx/sites-available/thai-sentiment`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
    }
}
```

Enable site:
```bash
sudo ln -s /etc/nginx/sites-available/thai-sentiment /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

#### 2. เพิ่ม HTTPS ด้วย Let's Encrypt

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

---

## 🖥️ การใช้งาน Web UI

### หน้าหลัก (/)

1. **กรอกข้อความ**: พิมพ์ข้อความที่ต้องการวิเคราะห์
2. **เลือกโหมด**:
   - ปิด A/B Testing: ใช้ Model A เพียงอย่างเดียว
   - เปิด A/B Testing: เปรียบเทียบ Model A กับ Model B
3. **เลือก Model B**: เลือกโมเดลที่ต้องการเปรียบเทียบ
4. **กดปุ่ม "วิเคราะห์"**: ดูผลลัพธ์

### ฟีเจอร์เพิ่มเติม

- **โหลดตัวอย่างข้อความ**: สุ่มข้อความตัวอย่างเพื่อทดสอบ
- **ดูตัวอย่างข้อผิดพลาด**: ดูรายการข้อความที่โมเดลทำนายผิด
- **Feedback System**: 
  - กด 👍 ถ้าผลลัพธ์ถูกต้อง
  - กด 👎 ถ้าผลลัพธ์ผิด พร้อมระบุคำตอบที่ถูกต้อง

### ผลลัพธ์ที่แสดง

- **Label**: ประเภทความรู้สึก (POSITIVE/NEGATIVE/NEUTRAL)
- **Confidence**: ความมั่นใจของโมเดล (0.00 - 1.00)
- **Latency**: เวลาที่ใช้ในการทำนาย (milliseconds)
- **Important Words**: คำที่มีอิทธิพลต่อการทำนาย พร้อมสี:
  - 🟢 เขียว = คำบวก
  - 🔴 แดง = คำลบ
  - 🟡 เหลือง = คำกลาง

---

## 🤖 โมเดลที่รองรับ

### Model A (Baseline) - Logistic Regression

**เทคนิค**: TF-IDF + Logistic Regression  
**ข้อดี**:
- เร็ว (< 10ms)
- ใช้ RAM น้อย
- ให้ probability scores ที่เชื่อถือได้
- Explainable (ดูได้ว่าคำไหนมีน้ำหนักมาก)

**ข้อเสีย**:
- ไม่เข้าใจบริบทลึก
- จับ sarcasm ไม่ได้ดี

---

### Model B Options

#### 1. Linear SVM
**เทคนิค**: TF-IDF + Linear Support Vector Machine  
**ข้อดี**: ดีกับ high-dimensional data, effective กับ text classification  
**ข้อเสีย**: ช้ากว่า Logistic Regression เล็กน้อย

#### 2. Random Forest
**เทคนิค**: TF-IDF + Random Forest Classifier  
**ข้อดี**: จัดการ feature interaction ได้ดี, ป้องกัน overfitting  
**ข้อเสีย**: ช้ากว่า linear models, ใช้ RAM มากกว่า

#### 3. Naive Bayes
**เทคนิค**: TF-IDF + Multinomial Naive Bayes  
**ข้อดี**: เร็วมาก, ทำงานดีกับข้อมูลน้อย  
**ข้อเสีย**: สมมติฐาน independence ของคำไม่เป็นจริง

#### 4. XGBoost
**เทคนิค**: TF-IDF + XGBoost Classifier  
**ข้อดี**: แม่นยำสูง, จัดการ imbalanced data ได้ดี  
**ข้อเสีย**: ช้า, ปรับ hyperparameters ยาก

#### 5. LightGBM
**เทคนิค**: TF-IDF + LightGBM  
**ข้อดี**: เร็วกว่า XGBoost, ใช้ RAM น้อยกว่า  
**ข้อเสีย**: อาจ overfit ง่ายกับข้อมูลน้อย

#### 6. Extra Trees
**เทคนิค**: TF-IDF + Extra Trees Classifier  
**ข้อดี**: เร็วกว่า Random Forest, reduce variance  
**ข้อเสีย**: อาจมี bias สูงกว่า Random Forest

#### 7. Thai BERT (WangChanBERTa)
**เทคนิค**: Pre-trained Thai BERT + Fine-tuning  
**ข้อดี**:
- เข้าใจบริบทลึก
- จับ sarcasm และ nuance ได้ดีกว่า
- SOTA สำหรับ Thai NLP

**ข้อเสีย**:
- ช้ามาก (100-200ms หรือมากกว่า)
- ต้องการ RAM และ GPU
- Explainability ต้องใช้ LIME

---

## 🔧 Troubleshooting

### ปัญหา: ImportError: No module named 'xxx'
**แก้ไข**: ติดตั้ง dependencies ใหม่
```bash
pip install -r requirements.txt
```

### ปัญหา: FileNotFoundError: model file not found
**แก้ไข**: รัน training script ก่อน
```bash
python Regress_train.py
```

### ปัญหา: BERT model ไม่โหลด
**สาเหตุ**: โฟลเดอร์ `models/bert_thai_sentiment` ไม่มี  
**แก้ไข**: BERT เป็น optional ระบบจะทำงานได้ปกติโดยไม่มี BERT

### ปัญหา: uvicorn command not found
**แก้ไข**: ตรวจสอบว่า activate virtual environment แล้วหรือยัง
```bash
# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

---

## 📊 Performance Benchmarks

| Model | Latency (avg) | Accuracy | F1-Score | RAM Usage |
|-------|---------------|----------|----------|-----------|
| Logistic Regression | 8ms | ~85% | ~0.83 | 150MB |
| Linear SVM | 12ms | ~86% | ~0.84 | 180MB |
| Random Forest | 45ms | ~84% | ~0.82 | 400MB |
| Naive Bayes | 5ms | ~80% | ~0.78 | 100MB |
| XGBoost | 35ms | ~87% | ~0.85 | 350MB |
| LightGBM | 25ms | ~86% | ~0.84 | 250MB |
| Extra Trees | 40ms | ~85% | ~0.83 | 380MB |
| BERT | 150ms+ | ~90% | ~0.89 | 2GB+ |

> ⚠️ ผลลัพธ์ข้างต้นเป็นเพียงตัวอย่าง ผลจริงขึ้นอยู่กับ hardware และ dataset

---

## 📝 License

MIT License - สามารถใช้งานได้อย่างอิสระ

---

## 👨‍💻 Author

**Phurin (Phurin123)**

GitHub: [https://github.com/Phurin123](https://github.com/Phurin123)

---

## 🙏 Acknowledgments

- **Wisesight Sentiment Corpus** - สำหรับ training data
- **AIResearch Thailand** - WangChanBERTa model
- **pythainlp** - Thai NLP tools
- **FastAPI** - Modern web framework

---

## 📮 Contact & Support

หากพบปัญหาหรือมีคำถาม:
- เปิด Issue ใน GitHub Repository
- ติดต่อผ่าน Email หรือ social media
