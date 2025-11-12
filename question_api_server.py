#!/usr/bin/env python3
"""
Question Generation API Server
FastAPI server để serve question generation model
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pickle
import os
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Question Generation AI API",
    description="API để tự động tạo câu hỏi từ keywords cho 3 lĩnh vực: IT, Economics, Marketing",
    version="1.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
question_ai = None

# Request/Response models
class QuestionRequest(BaseModel):
    keyword: str = Field(..., min_length=1, max_length=200, description="Keyword để tạo câu hỏi")
    num_questions: int = Field(5, ge=1, le=20, description="Số câu hỏi cần tạo (1-20)")
    category_hint: Optional[str] = Field(None, pattern="^(it|economics|marketing)$", description="Gợi ý category")

class QuestionResponse(BaseModel):
    keyword: str
    category: str
    confidence: float
    questions: List[Dict[str, Any]]
    generated_at: str
    total_questions: int

class CategoryPrediction(BaseModel):
    keyword: str
    predicted_category: str
    confidence: float
    all_probabilities: Dict[str, float]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    api_version: str

# Load model on startup
@app.on_event("startup")
async def load_model():
    """Load AI model on startup"""
    global question_ai
    
    logger.info("🚀 Starting Question Generation API...")
    
    try:
        # Import after FastAPI starts to avoid issues
        from simple_question_trainer import SimpleQuestionAI
        from advanced_question_ai import AdvancedQuestionAI
        
        # Load advanced ML model first (preferred)
        try:
            question_ai = AdvancedQuestionAI()
            question_ai.load_advanced_model()
            logger.info("✅ Advanced ML model loaded successfully! (Pure ML - No Templates)")
        except Exception as e:
            logger.warning(f"⚠️ Advanced model failed: {e}")
            # Fallback to simple model
            question_ai = SimpleQuestionAI()
            if question_ai.load_model():
                logger.info("✅ Simple model loaded as fallback")
            else:
                logger.warning("⚠️ No models found. Using templates only.")
        
        logger.info("🎉 Question Generation API ready!")
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        question_ai = None

# API Endpoints

@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint"""
    return {
        "message": "Question Generation AI API",
        "version": "1.0.0",
        "description": "Tự động tạo câu hỏi từ keywords cho IT, Economics, Marketing",
        "endpoints": {
            "generate": "/api/generate-questions",
            "predict": "/api/predict-category", 
            "health": "/health",
            "docs": "/docs"
        },
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if question_ai else "unhealthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=question_ai is not None,
        api_version="1.0.0"
    )

@app.post("/api/generate-questions", response_model=QuestionResponse)
async def generate_questions(request: QuestionRequest):
    """Generate questions từ keyword"""
    
    if not question_ai:
        raise HTTPException(status_code=503, detail="AI model not available")
    
    try:
        logger.info(f"📝 Generating questions for: '{request.keyword}'")
        
        # Generate questions using advanced ML if available
        if hasattr(question_ai, 'generate_questions_ml'):
            questions = question_ai.generate_questions_ml(
                keyword=request.keyword,
                num_questions=request.num_questions
            )
        else:
            # Fallback to simple generation
            questions = question_ai.generate_questions(
                keyword=request.keyword,
                num_questions=request.num_questions
            )
        
        if not questions:
            raise HTTPException(status_code=400, detail="Could not generate questions for this keyword")
        
        # Extract info from first question
        first_q = questions[0]
        category = first_q['category']
        confidence = first_q['confidence']
        
        # Format response
        formatted_questions = []
        for i, q in enumerate(questions, 1):
            formatted_questions.append({
                "id": i,
                "question": q['question'],
                "category": q['category'],
                "confidence": q['confidence']
            })
        
        response = QuestionResponse(
            keyword=request.keyword,
            category=category,
            confidence=confidence,
            questions=formatted_questions,
            generated_at=datetime.now().isoformat(),
            total_questions=len(questions)
        )
        
        logger.info(f"✅ Generated {len(questions)} questions successfully")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error generating questions: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating questions: {str(e)}")

@app.post("/api/predict-category", response_model=CategoryPrediction)
async def predict_category(keyword: str):
    """Predict category từ keyword"""
    
    if not question_ai:
        raise HTTPException(status_code=503, detail="AI model not available")
    
    try:
        logger.info(f"🎯 Predicting category for: '{keyword}'")
        
        category, confidence = question_ai.predict_category(keyword)
        
        # Create mock probabilities (in real implementation, get from model)
        all_probs = {
            "it": 0.33,
            "economics": 0.33, 
            "marketing": 0.34
        }
        all_probs[category] = confidence
        
        response = CategoryPrediction(
            keyword=keyword,
            predicted_category=category,
            confidence=confidence,
            all_probabilities=all_probs
        )
        
        logger.info(f"✅ Predicted category: {category} ({confidence:.3f})")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error predicting category: {e}")
        raise HTTPException(status_code=500, detail=f"Error predicting category: {str(e)}")

@app.get("/api/categories")
async def get_categories():
    """Get available categories"""
    return {
        "categories": [
            {
                "id": "it", 
                "name": "Information Technology",
                "description": "Software development, cloud computing, AI, cybersecurity"
            },
            {
                "id": "economics",
                "name": "Economics & Finance", 
                "description": "Investment, financial planning, market analysis"
            },
            {
                "id": "marketing",
                "name": "Marketing & Advertising",
                "description": "Digital marketing, campaigns, brand management"
            }
        ]
    }

@app.get("/api/templates")
async def get_question_templates():
    """Get question templates for reference"""
    
    if not question_ai:
        raise HTTPException(status_code=503, detail="AI model not available")
    
    return {
        "templates": question_ai.question_templates,
        "total_templates": sum(len(templates) for templates in question_ai.question_templates.values()),
        "categories": list(question_ai.question_templates.keys())
    }

@app.get("/api/stats")
async def get_api_stats():
    """Get API statistics"""
    
    # In production, this would track real usage stats
    return {
        "total_requests": 0,
        "questions_generated": 0,
        "categories_predicted": 0,
        "uptime": "Just started",
        "model_status": "loaded" if question_ai else "not_loaded",
        "last_updated": datetime.now().isoformat()
    }

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"error": "Invalid input", "detail": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "An unexpected error occurred"}
    )

# Additional utility endpoints
@app.post("/api/batch-generate")
async def batch_generate_questions(keywords: List[str], num_questions: int = 3):
    """Generate questions cho nhiều keywords cùng lúc"""
    
    if not question_ai:
        raise HTTPException(status_code=503, detail="AI model not available")
    
    if len(keywords) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 keywords allowed")
    
    results = []
    
    for keyword in keywords:
        try:
            questions = question_ai.generate_questions(keyword, num_questions)
            
            if questions:
                results.append({
                    "keyword": keyword,
                    "category": questions[0]['category'],
                    "confidence": questions[0]['confidence'],
                    "questions": [q['question'] for q in questions],
                    "status": "success"
                })
            else:
                results.append({
                    "keyword": keyword,
                    "status": "failed",
                    "error": "Could not generate questions"
                })
                
        except Exception as e:
            results.append({
                "keyword": keyword,
                "status": "error",
                "error": str(e)
            })
    
    return {
        "results": results,
        "total_keywords": len(keywords),
        "successful": len([r for r in results if r.get("status") == "success"]),
        "failed": len([r for r in results if r.get("status") != "success"])
    }

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Question Generation API Server...")
    print("📖 API Documentation: http://localhost:9000/docs")
    print("🔍 Health Check: http://localhost:9000/health")
    
    uvicorn.run(
        "question_api_server:app",
        host="0.0.0.0",
        port=9000,
        reload=True,
        log_level="info"
    )
