# Form Agent AI

Hệ thống AI tự động tạo form chuyên nghiệp từ keyword, hỗ trợ 3 lĩnh vực: **Công nghệ thông tin**, **Kinh tế - Tài chính**, và **Marketing**.

## 🚀 Tính năng chính

- **AI Classification**: Tự động phân loại keyword vào đúng lĩnh vực
- **Smart Form Generation**: Tạo form với fields, validation và styling phù hợp
- **Multiple Complexity Levels**: Hỗ trợ từ form đơn giản đến phức tạp
- **REST API**: API hoàn chỉnh cho tích hợp
- **Web Interface**: Giao diện thân thiện, không sử dụng icon
- **Database Integration**: Lưu trữ và quản lý forms, submissions
- **Analytics**: Thống kê và phân tích sử dụng

## 📋 Yêu cầu hệ thống

- Python 3.8+
- 4GB RAM minimum (8GB recommended)
- 2GB disk space

## 🛠️ Cài đặt

### 1. Clone repository và cài đặt dependencies

```bash
git clone <repository-url>
cd form-agent-AI-project
pip install -r requirements.txt
```

### 2. Tạo dataset (tùy chọn)

```bash
# Tạo dataset 500,000 mẫu (mất khoảng 30-60 phút)
python dataset_generator.py

# Hoặc tạo dataset mẫu nhỏ hơn
python dataset_generator.py --sample-size 10000
```

### 3. Khởi tạo database

```bash
python database.py
```

### 4. Huấn luyện model AI (tùy chọn)

```bash
# Nếu đã có dataset
python form_agent_ai.py

# Hoặc chạy với dataset mẫu
python form_agent_ai.py --dataset datasets/form_agent_dataset_sample_10000.csv
```

### 5. Chạy server

```bash
# Development mode
python main.py

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 6. Truy cập ứng dụng

- **Web Interface**: http://localhost:8000/static/index.html
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health

## 📚 Sử dụng API

### Tạo form từ keyword

```bash
curl -X POST "http://localhost:8000/api/generate-form" \
     -H "Content-Type: application/json" \
     -d '{
       "keyword": "đánh giá bảo mật cloud",
       "category": "it",
       "complexity": "Complex"
     }'
```

### Lấy thông tin form

```bash
curl "http://localhost:8000/api/forms/{form_id}"
```

### Submit form

```bash
curl "http://localhost:8000/api/submit-form" \
     -H "Content-Type: application/json" \
     -d '{
       "form_id": "{form_id}",
       "form_data": {
         "name": "Nguyen Van A",
         "email": "test@example.com"
       }
     }'
```

## 🏗️ Kiến trúc hệ thống

```
form-agent-AI-project/
├── dataset_generator.py      # Tạo dataset huấn luyện
├── form_agent_ai.py         # Model AI classification
├── form_generation_engine.py # Engine tạo form structure  
├── main.py                  # FastAPI backend server
├── database.py              # Database models & ORM
├── static/
│   └── index.html          # Web interface
├── models/                 # Trained AI models
├── datasets/               # Training datasets
└── requirements.txt        # Dependencies
```

---

**Form Agent AI** - Tự động hóa việc tạo form với sức mạnh AI 🤖
