#!/usr/bin/env python3
"""
Form Agent AI - FastAPI Backend Server
Professional backend to serve trained AI models for form generation
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import joblib
import json
import os
import logging
from datetime import datetime
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FormRequest(BaseModel):
    keyword: str
    domain: Optional[str] = None
    
class FormResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    message: str
    timestamp: str

class PredictionResponse(BaseModel):
    keyword: str
    category: str
    category_confidence: float
    form_type: str
    form_type_confidence: float
    complexity: str
    complexity_confidence: float
    estimated_fields: int
    estimated_time: str

class FormAgentAPI:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.models = {}
        self.vectorizers = {}
        self.metadata = {}
        self.load_models()
        
    def load_models(self):
        """Load trained models and vectorizers"""
        try:
            # Load models
            self.models['category'] = joblib.load(f"{self.models_dir}/category_model.pkl")
            self.models['form_type'] = joblib.load(f"{self.models_dir}/form_type_model.pkl") 
            self.models['complexity'] = joblib.load(f"{self.models_dir}/complexity_model.pkl")
            
            # Load vectorizers
            self.vectorizers['category'] = joblib.load(f"{self.models_dir}/category_vectorizer.pkl")
            self.vectorizers['form_type'] = joblib.load(f"{self.models_dir}/form_type_vectorizer.pkl")
            self.vectorizers['complexity'] = joblib.load(f"{self.models_dir}/complexity_vectorizer.pkl")
            
            # Load metadata
            with open(f"{self.models_dir}/model_metadata.json", 'r') as f:
                self.metadata = json.load(f)
                
            logger.info(" All models loaded successfully!")
            logger.info(f" Model accuracies: {self.metadata.get('accuracies', {})}")
            
        except Exception as e:
            logger.error(f" Error loading models: {e}")
            raise
    
    def predict(self, keyword: str) -> PredictionResponse:
        """Make predictions for a given keyword"""
        try:
            # Prepare keyword for prediction
            keyword_lower = keyword.lower().strip()
            
            # Category prediction
            cat_vec = self.vectorizers['category'].transform([keyword_lower])
            cat_pred = self.models['category'].predict(cat_vec)[0]
            cat_proba = max(self.models['category'].predict_proba(cat_vec)[0])
            
            # Form type prediction  
            ft_vec = self.vectorizers['form_type'].transform([keyword_lower])
            ft_pred = self.models['form_type'].predict(ft_vec)[0]
            ft_proba = max(self.models['form_type'].predict_proba(ft_vec)[0])
            
            # Complexity prediction
            comp_vec = self.vectorizers['complexity'].transform([keyword_lower])
            comp_pred = self.models['complexity'].predict(comp_vec)[0]
            comp_proba = max(self.models['complexity'].predict_proba(comp_vec)[0])
            
            # Estimate fields and time based on complexity
            field_mapping = {
                'Simple': (5, 8),
                'Moderate': (10, 15), 
                'Complex': (20, 30)
            }
            
            time_mapping = {
                'Simple': '5-10 minutes',
                'Moderate': '15-20 minutes',
                'Complex': '25-35 minutes'
            }
            
            import random
            min_fields, max_fields = field_mapping.get(comp_pred, (10, 15))
            estimated_fields = random.randint(min_fields, max_fields)
            estimated_time = time_mapping.get(comp_pred, '15-20 minutes')
            
            return PredictionResponse(
                keyword=keyword,
                category=cat_pred,
                category_confidence=round(cat_proba, 3),
                form_type=ft_pred,
                form_type_confidence=round(ft_proba, 3),
                complexity=comp_pred,
                complexity_confidence=round(comp_proba, 3),
                estimated_fields=estimated_fields,
                estimated_time=estimated_time
            )
            
        except Exception as e:
            logger.error(f" Prediction error for '{keyword}': {e}")
            raise

# Initialize FastAPI app
app = FastAPI(
    title="Form Agent AI API",
    description="Professional AI-powered form generation system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize AI engine
try:
    ai_engine = FormAgentAPI()
    logger.info(" Form Agent AI initialized successfully!")
except Exception as e:
    logger.error(f" Failed to initialize AI engine: {e}")
    ai_engine = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Form Agent AI - Professional Backend",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": ai_engine is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/models/info")
async def model_info():
    """Get model information and statistics"""
    if not ai_engine:
        raise HTTPException(status_code=503, detail="AI models not loaded")
    
    return {
        "metadata": ai_engine.metadata,
        "models_available": list(ai_engine.models.keys()),
        "vectorizers_available": list(ai_engine.vectorizers.keys()),
        "status": "ready"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_form(request: FormRequest):
    """Predict form characteristics based on keyword"""
    if not ai_engine:
        raise HTTPException(status_code=503, detail="AI models not loaded")
    
    if not request.keyword or len(request.keyword.strip()) < 2:
        raise HTTPException(status_code=400, detail="Keyword must be at least 2 characters")
    
    try:
        logger.info(f" Processing prediction for: '{request.keyword}'")
        prediction = ai_engine.predict(request.keyword)
        return prediction
        
    except Exception as e:
        logger.error(f" Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/generate", response_model=FormResponse)
async def generate_form(request: FormRequest):
    """Generate complete form structure based on keyword"""
    if not ai_engine:
        raise HTTPException(status_code=503, detail="AI models not loaded")
    
    try:
        # Get predictions
        prediction = ai_engine.predict(request.keyword)
        
        # Generate form structure based on predictions
        form_structure = {
            "form_id": f"form_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "title": f"{prediction.category.title()} {prediction.form_type.title()}: {request.keyword.title()}",
            "category": prediction.category,
            "type": prediction.form_type,
            "complexity": prediction.complexity,
            "estimated_time": prediction.estimated_time,
            "fields": generate_form_fields(prediction),
            "metadata": {
                "created": datetime.now().isoformat(),
                "keyword": request.keyword,
                "ai_confidence": {
                    "category": prediction.category_confidence,
                    "form_type": prediction.form_type_confidence,
                    "complexity": prediction.complexity_confidence
                }
            }
        }
        
        return FormResponse(
            success=True,
            data=form_structure,
            message="Form generated successfully",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f" Form generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Form generation failed: {str(e)}")

def generate_form_fields(prediction: PredictionResponse) -> List[Dict[str, Any]]:
    """Generate form fields based on prediction"""
    
    # Field templates by category and type
    field_templates = {
        'it': {
            'basic': ['name', 'email', 'company', 'role', 'experience_level'],
            'technical': ['programming_languages', 'frameworks', 'cloud_platforms', 'certifications'],
            'assessment': ['technical_skills', 'project_experience', 'problem_solving', 'code_review']
        },
        'economics': {
            'basic': ['name', 'email', 'organization', 'position', 'education'],
            'professional': ['industry_sector', 'years_experience', 'specialization', 'certifications'],
            'analysis': ['economic_indicators', 'market_knowledge', 'analytical_tools', 'research_methods']
        },
        'marketing': {
            'basic': ['name', 'email', 'company', 'department', 'role'],
            'strategy': ['target_audience', 'marketing_channels', 'budget_range', 'campaign_goals'],
            'digital': ['social_platforms', 'analytics_tools', 'content_types', 'conversion_metrics']
        }
    }
    
    # Generate fields based on category and complexity
    category = prediction.category
    complexity = prediction.complexity
    
    fields = []
    field_id = 1
    
    # Always include basic fields
    basic_fields = field_templates.get(category, {}).get('basic', ['name', 'email'])
    
    for field_name in basic_fields:
        fields.append({
            'id': field_id,
            'name': field_name,
            'label': field_name.replace('_', ' ').title(),
            'type': 'email' if 'email' in field_name else 'text',
            'required': True,
            'category': 'basic'
        })
        field_id += 1
    
    # Add category-specific fields based on complexity
    if complexity in ['Moderate', 'Complex']:
        professional_fields = field_templates.get(category, {}).get('professional', [])
        for field_name in professional_fields:
            fields.append({
                'id': field_id,
                'name': field_name,
                'label': field_name.replace('_', ' ').title(),
                'type': 'select',
                'required': False,
                'category': 'professional'
            })
            field_id += 1
    
    # Add advanced fields for complex forms
    if complexity == 'Complex':
        advanced_fields = field_templates.get(category, {}).get('analysis', [])
        for field_name in advanced_fields:
            fields.append({
                'id': field_id,
                'name': field_name,
                'label': field_name.replace('_', ' ').title(),
                'type': 'textarea',
                'required': False,
                'category': 'advanced'
            })
            field_id += 1
    
    return fields

@app.get("/static/{filename}")
async def serve_static(filename: str):
    """Serve static files"""
    static_path = f"static/{filename}"
    if os.path.exists(static_path):
        return FileResponse(static_path)
    raise HTTPException(status_code=404, detail="File not found")

if __name__ == "__main__":
    print(" Starting Form Agent AI Server...")
    print(" Loading AI models...")
    print(" Server will be available at: http://localhost:8000")
    print(" API Documentation: http://localhost:8000/docs")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
