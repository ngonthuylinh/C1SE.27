#!/usr/bin/env python3
"""
FastAPI Backend cho Form Agent AI
REST API để serve model và form generation logic
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import asyncio
import json
import os
import uuid
from datetime import datetime, timedelta
import logging
from contextlib import asynccontextmanager
import uvicorn

# Import our custom modules
from form_agent_ai import FormAgentAI
from form_generation_engine import FormGenerationEngine, ComplexityLevel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to store models
ai_model = None
form_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    global ai_model, form_engine
    logger.info("Starting up Form Agent AI Backend...")
    
    # Initialize models
    ai_model = FormAgentAI(model_dir="models")
    form_engine = FormGenerationEngine()
    
    # Try to load pre-trained models
    try:
        ai_model.load_models()
        logger.info("Pre-trained models loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load pre-trained models: {e}")
        logger.info("Will use default templates and basic classification")
    
    logger.info("Form Agent AI Backend started successfully!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Form Agent AI Backend...")

# Initialize FastAPI app
app = FastAPI(
    title="Form Agent AI API",
    description="API để tạo form tự động từ keyword sử dụng AI",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic models for request/response
class KeywordRequest(BaseModel):
    keyword: str = Field(..., min_length=1, max_length=500, description="Từ khóa để tạo form")
    category: Optional[str] = Field(None, regex="^(it|economics|marketing)$", description="Lĩnh vực (tự động phát hiện nếu không cung cấp)")
    complexity: Optional[str] = Field(None, regex="^(Simple|Moderate|Complex)$", description="Độ phức tạp")
    form_type: Optional[str] = Field(None, description="Loại form cụ thể")
    additional_requirements: Optional[str] = Field(None, max_length=1000, description="Yêu cầu bổ sung")
    
    @validator('keyword')
    def validate_keyword(cls, v):
        if not v.strip():
            raise ValueError('Keyword không được để trống')
        return v.strip()

class FormField(BaseModel):
    name: str
    type: str
    label: str
    required: bool = False
    placeholder: Optional[str] = None
    description: Optional[str] = None
    options: Optional[List[str]] = None
    validation: Optional[Dict[str, Any]] = None
    default_value: Optional[Any] = None

class FormResponse(BaseModel):
    form_id: str
    title: str
    description: str
    category: str
    form_type: str
    complexity: str
    keyword: str
    fields: List[Dict[str, Any]]
    sections: List[Dict[str, Any]]
    validation_rules: Dict[str, Any]
    styling: Dict[str, Any]
    behavior: Dict[str, Any]
    metadata: Dict[str, Any]

class FormSubmissionRequest(BaseModel):
    form_id: str
    form_data: Dict[str, Any]
    submission_metadata: Optional[Dict[str, Any]] = None

class FormSubmissionResponse(BaseModel):
    submission_id: str
    form_id: str
    status: str
    submitted_at: datetime
    validation_errors: Optional[List[Dict[str, Any]]] = None

class TrainingRequest(BaseModel):
    dataset_path: str = Field(..., description="Đường dẫn đến dataset")
    retrain: bool = Field(False, description="Có huấn luyện lại từ đầu không")

class AnalyticsResponse(BaseModel):
    total_forms_generated: int
    total_submissions: int
    popular_keywords: List[Dict[str, Any]]
    category_distribution: Dict[str, int]
    complexity_distribution: Dict[str, int]

# In-memory storage (in production, use proper database)
generated_forms = {}
form_submissions = {}
analytics_data = {
    "form_generations": [],
    "submissions": [],
    "keywords": {}
}

# Helper functions
def get_ai_model():
    """Dependency to get AI model"""
    if ai_model is None:
        raise HTTPException(status_code=503, detail="AI model not initialized")
    return ai_model

def get_form_engine():
    """Dependency to get form generation engine"""
    if form_engine is None:
        raise HTTPException(status_code=503, detail="Form generation engine not initialized")
    return form_engine

def update_analytics(keyword: str, category: str, complexity: str):
    """Update analytics data"""
    analytics_data["form_generations"].append({
        "keyword": keyword,
        "category": category,
        "complexity": complexity,
        "timestamp": datetime.now()
    })
    
    # Update keyword count
    if keyword not in analytics_data["keywords"]:
        analytics_data["keywords"][keyword] = 0
    analytics_data["keywords"][keyword] += 1

def validate_form_data(form_structure: Dict, form_data: Dict) -> List[Dict[str, Any]]:
    """Validate form submission data"""
    errors = []
    
    # Check required fields
    required_fields = form_structure["validation_rules"]["required_fields"]
    for field_name in required_fields:
        if field_name not in form_data or not form_data[field_name]:
            errors.append({
                "field": field_name,
                "error": "required",
                "message": f"Trường {field_name} là bắt buộc"
            })
    
    # Validate field types and constraints
    for field in form_structure["fields"]:
        field_name = field["name"]
        if field_name in form_data:
            value = form_data[field_name]
            field_errors = validate_field_value(field, value)
            errors.extend(field_errors)
    
    return errors

def validate_field_value(field: Dict, value: Any) -> List[Dict[str, Any]]:
    """Validate individual field value"""
    errors = []
    field_name = field["name"]
    field_type = field["type"]
    validation = field.get("validation", {})
    
    if value is None or value == "":
        return errors  # Skip validation for empty optional fields
    
    # Type-specific validation
    if field_type == "email":
        import re
        email_pattern = r'^[^\s@]+@[^\s@]+\.[^\s@]+$'
        if not re.match(email_pattern, str(value)):
            errors.append({
                "field": field_name,
                "error": "invalid_email",
                "message": "Email không hợp lệ"
            })
    
    elif field_type == "number":
        try:
            num_value = float(value)
            if "min_value" in validation and num_value < validation["min_value"]:
                errors.append({
                    "field": field_name,
                    "error": "min_value",
                    "message": f"Giá trị phải >= {validation['min_value']}"
                })
            if "max_value" in validation and num_value > validation["max_value"]:
                errors.append({
                    "field": field_name,
                    "error": "max_value",
                    "message": f"Giá trị phải <= {validation['max_value']}"
                })
        except ValueError:
            errors.append({
                "field": field_name,
                "error": "invalid_number",
                "message": "Giá trị phải là số"
            })
    
    elif field_type in ["text", "textarea"]:
        if "min_length" in validation and len(str(value)) < validation["min_length"]:
            errors.append({
                "field": field_name,
                "error": "min_length",
                "message": f"Độ dài tối thiểu {validation['min_length']} ký tự"
            })
        if "max_length" in validation and len(str(value)) > validation["max_length"]:
            errors.append({
                "field": field_name,
                "error": "max_length",
                "message": f"Độ dài tối đa {validation['max_length']} ký tự"
            })
    
    return errors

# API Endpoints

@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint"""
    return {
        "message": "Form Agent AI API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "generate_form": "/api/generate-form",
            "submit_form": "/api/submit-form",
            "get_form": "/api/forms/{form_id}",
            "analytics": "/api/analytics",
            "health": "/health"
        }
    }

@app.get("/health", response_class=JSONResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": ai_model is not None,
        "engine_ready": form_engine is not None
    }

@app.post("/api/generate-form", response_model=FormResponse)
async def generate_form(
    request: KeywordRequest,
    background_tasks: BackgroundTasks,
    engine: FormGenerationEngine = Depends(get_form_engine),
    model: FormAgentAI = Depends(get_ai_model)
):
    """Tạo form từ keyword"""
    try:
        keyword = request.keyword
        
        # Auto-detect category if not provided
        if request.category:
            category = request.category
            category_confidence = 1.0
        else:
            category, category_confidence = model.predict_category(keyword)
        
        # Auto-detect complexity if not provided
        if request.complexity:
            complexity = ComplexityLevel(request.complexity)
        else:
            complexity_str, _ = model.predict_complexity(keyword)
            complexity = ComplexityLevel(complexity_str)
        
        # Generate form structure
        form_structure = engine.generate_form_structure(
            keyword=keyword,
            category=category,
            form_type=request.form_type,
            complexity=complexity
        )
        
        # Add additional requirements if provided
        if request.additional_requirements:
            form_structure["additional_requirements"] = request.additional_requirements
            form_structure["metadata"]["has_additional_requirements"] = True
        
        # Store generated form
        form_id = form_structure["form_id"]
        generated_forms[form_id] = form_structure
        
        # Update analytics in background
        background_tasks.add_task(
            update_analytics,
            keyword,
            category,
            complexity.value
        )
        
        # Add confidence scores to metadata
        form_structure["metadata"]["category_confidence"] = category_confidence
        
        logger.info(f"Generated form {form_id} for keyword: {keyword}")
        
        return FormResponse(**form_structure)
        
    except Exception as e:
        logger.error(f"Error generating form: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi tạo form: {str(e)}")

@app.get("/api/forms/{form_id}", response_model=FormResponse)
async def get_form(form_id: str):
    """Lấy thông tin form theo ID"""
    if form_id not in generated_forms:
        raise HTTPException(status_code=404, detail="Form không tồn tại")
    
    form_structure = generated_forms[form_id]
    return FormResponse(**form_structure)

@app.post("/api/submit-form", response_model=FormSubmissionResponse)
async def submit_form(request: FormSubmissionRequest):
    """Submit form data"""
    form_id = request.form_id
    
    if form_id not in generated_forms:
        raise HTTPException(status_code=404, detail="Form không tồn tại")
    
    form_structure = generated_forms[form_id]
    
    # Validate form data
    validation_errors = validate_form_data(form_structure, request.form_data)
    
    # Generate submission ID
    submission_id = str(uuid.uuid4())
    
    # Create submission record
    submission = {
        "submission_id": submission_id,
        "form_id": form_id,
        "form_data": request.form_data,
        "metadata": request.submission_metadata or {},
        "submitted_at": datetime.now(),
        "validation_errors": validation_errors,
        "status": "invalid" if validation_errors else "valid"
    }
    
    # Store submission
    form_submissions[submission_id] = submission
    analytics_data["submissions"].append(submission)
    
    logger.info(f"Form submission {submission_id} for form {form_id}")
    
    return FormSubmissionResponse(
        submission_id=submission_id,
        form_id=form_id,
        status=submission["status"],
        submitted_at=submission["submitted_at"],
        validation_errors=validation_errors if validation_errors else None
    )

@app.get("/api/forms/{form_id}/submissions")
async def get_form_submissions(form_id: str):
    """Lấy tất cả submissions của form"""
    if form_id not in generated_forms:
        raise HTTPException(status_code=404, detail="Form không tồn tại")
    
    submissions = [
        submission for submission in form_submissions.values()
        if submission["form_id"] == form_id
    ]
    
    return {
        "form_id": form_id,
        "total_submissions": len(submissions),
        "submissions": submissions
    }

@app.get("/api/submissions/{submission_id}")
async def get_submission(submission_id: str):
    """Lấy thông tin submission theo ID"""
    if submission_id not in form_submissions:
        raise HTTPException(status_code=404, detail="Submission không tồn tại")
    
    return form_submissions[submission_id]

@app.get("/api/analytics", response_model=AnalyticsResponse)
async def get_analytics():
    """Lấy analytics data"""
    # Calculate statistics
    total_forms = len(generated_forms)
    total_submissions = len(form_submissions)
    
    # Popular keywords
    keyword_counts = analytics_data["keywords"]
    popular_keywords = [
        {"keyword": keyword, "count": count}
        for keyword, count in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    ]
    
    # Category distribution
    category_dist = {}
    complexity_dist = {}
    
    for generation in analytics_data["form_generations"]:
        category = generation["category"]
        complexity = generation["complexity"]
        
        category_dist[category] = category_dist.get(category, 0) + 1
        complexity_dist[complexity] = complexity_dist.get(complexity, 0) + 1
    
    return AnalyticsResponse(
        total_forms_generated=total_forms,
        total_submissions=total_submissions,
        popular_keywords=popular_keywords,
        category_distribution=category_dist,
        complexity_distribution=complexity_dist
    )

@app.post("/api/train-model")
async def train_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    model: FormAgentAI = Depends(get_ai_model)
):
    """Huấn luyện/cập nhật model"""
    dataset_path = request.dataset_path
    
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset không tồn tại")
    
    # Start training in background
    background_tasks.add_task(train_model_background, dataset_path, request.retrain)
    
    return {
        "message": "Bắt đầu huấn luyện model",
        "dataset_path": dataset_path,
        "retrain": request.retrain,
        "status": "training_started"
    }

async def train_model_background(dataset_path: str, retrain: bool):
    """Background task to train model"""
    try:
        logger.info(f"Starting model training with dataset: {dataset_path}")
        
        global ai_model
        if retrain or ai_model is None:
            ai_model = FormAgentAI(model_dir="models")
        
        ai_model.train_models(dataset_path)
        logger.info("Model training completed successfully")
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")

@app.post("/api/predict-category")
async def predict_category(
    keyword: str,
    model: FormAgentAI = Depends(get_ai_model)
):
    """Dự đoán category từ keyword"""
    try:
        category, confidence = model.predict_category(keyword)
        return {
            "keyword": keyword,
            "predicted_category": category,
            "confidence": confidence,
            "categories": ["it", "economics", "marketing"]
        }
    except Exception as e:
        logger.error(f"Error predicting category: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/categories")
async def get_categories():
    """Lấy danh sách categories"""
    return {
        "categories": [
            {
                "id": "it",
                "name": "Công nghệ thông tin",
                "description": "Lĩnh vực công nghệ, phần mềm, hệ thống"
            },
            {
                "id": "economics",
                "name": "Kinh tế - Tài chính",
                "description": "Lĩnh vực tài chính, đầu tư, kinh doanh"
            },
            {
                "id": "marketing",
                "name": "Marketing",
                "description": "Lĩnh vực marketing, quảng cáo, bán hàng"
            }
        ]
    }

@app.get("/api/form-types")
async def get_form_types():
    """Lấy danh sách form types"""
    return {
        "form_types": [
            {
                "id": "registration",
                "name": "Đăng ký",
                "description": "Form đăng ký tài khoản, dịch vụ"
            },
            {
                "id": "survey",
                "name": "Khảo sát",
                "description": "Form thu thập ý kiến, phản hồi"
            },
            {
                "id": "application",
                "name": "Đơn ứng tuyển",
                "description": "Form nộp hồ sơ, ứng tuyển"
            },
            {
                "id": "consultation",
                "name": "Tư vấn",
                "description": "Form yêu cầu tư vấn, hỗ trợ"
            },
            {
                "id": "assessment",
                "name": "Đánh giá",
                "description": "Form đánh giá năng lực, kiến thức"
            }
        ]
    }

@app.delete("/api/forms/{form_id}")
async def delete_form(form_id: str):
    """Xóa form"""
    if form_id not in generated_forms:
        raise HTTPException(status_code=404, detail="Form không tồn tại")
    
    del generated_forms[form_id]
    
    # Also delete related submissions
    submission_ids_to_delete = [
        sid for sid, submission in form_submissions.items()
        if submission["form_id"] == form_id
    ]
    
    for sid in submission_ids_to_delete:
        del form_submissions[sid]
    
    return {
        "message": f"Form {form_id} đã được xóa",
        "deleted_submissions": len(submission_ids_to_delete)
    }

@app.get("/api/export-data")
async def export_data():
    """Export tất cả data để backup"""
    export_data = {
        "generated_forms": generated_forms,
        "form_submissions": form_submissions,
        "analytics_data": analytics_data,
        "exported_at": datetime.now().isoformat()
    }
    
    # Save to file
    export_filename = f"form_agent_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(export_filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
    
    return FileResponse(
        path=export_filename,
        filename=export_filename,
        media_type='application/json'
    )

# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Lỗi hệ thống"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
