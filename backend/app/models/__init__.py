from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class QuestionResponse:
    """Response model cho generated questions"""
    question: str
    category: str
    confidence: float
    method: str
    source_keyword: Optional[str] = None
    similarity_score: Optional[float] = None

@dataclass
class GenerateQuestionsRequest:
    """Request model cho generate questions"""
    keyword: str
    num_questions: int = 5
    category: Optional[str] = None

@dataclass
class CategoryPredictionRequest:
    """Request model cho category prediction"""
    keyword: str

@dataclass
class CategoryPredictionResponse:
    """Response model cho category prediction"""
    keyword: str
    predicted_category: str
    confidence: float
    all_probabilities: Dict[str, float]

@dataclass
class ModelInfoResponse:
    """Response model cho model information"""
    model_loaded: bool
    training_date: str
    total_keywords: int
    total_questions: int
    categories: List[str]
    model_accuracy: Optional[float] = None

@dataclass
class HealthResponse:
    """Response model cho health check"""
    status: str
    message: str
    model_loaded: bool
    timestamp: str