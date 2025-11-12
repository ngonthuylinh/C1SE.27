# Form Agent AI - Backend API

## 📋 Giới thiệu
Backend API cho hệ thống Form Agent AI, sử dụng model đã được train để generate câu hỏi từ keywords.

## 🚀 Cài đặt và Chạy

### 1. Cài đặt dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Chạy development server
```bash
python run.py
```

### 3. Chạy production server
```bash
gunicorn -w 4 -b 0.0.0.0:8000 run:app
```

## 📚 API Endpoints

### 1. Health Check
- **GET** `/api/health`
- Kiểm tra trạng thái server

### 2. Generate Questions
- **POST** `/api/questions/generate`
- Tạo câu hỏi từ keyword
- Body:
```json
{
  "keyword": "artificial intelligence",
  "num_questions": 5,
  "category": "it" // optional
}
```

### 3. Predict Category
- **POST** `/api/predict/category`
- Dự đoán category từ keyword
- Body:
```json
{
  "keyword": "machine learning"
}
```

### 4. Model Info
- **GET** `/api/model/info`
- Thông tin về model đã load

## 🏗️ Cấu trúc thư mục
```
backend/
├── app/
│   ├── api/               # API routes
│   ├── services/          # Business logic
│   ├── models/           # Data models
│   ├── utils/            # Utilities
│   └── __init__.py
├── requirements.txt      # Dependencies
├── run.py               # Entry point
└── config.py           # Configuration
```

## 🔧 Configuration

Tạo file `.env` trong thư mục backend:
```
FLASK_ENV=development
MODEL_PATH=../models/real_data_question_model.pkl
HOST=0.0.0.0
PORT=8000
DEBUG=True
```

## 🧪 Testing

Test API endpoints:
```bash
# Health check
curl http://localhost:8000/api/health

# Generate questions
curl -X POST http://localhost:8000/api/questions/generate \
  -H "Content-Type: application/json" \
  -d '{"keyword": "financial modeling", "num_questions": 3}'
```