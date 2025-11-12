import pickle
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from collections import defaultdict
import sys

class QuestionGeneratorService:
    """Service class để load và sử dụng trained model"""
    
    _instance = None
    
    def __init__(self):
        self.model_loaded = False
        self.trainer = None
        self.model_data = None
        self.logger = logging.getLogger(__name__)
    
    @classmethod
    def get_instance(cls):
        """Singleton pattern để đảm bảo chỉ có 1 instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def initialize_model(self, model_path: str = None):
        """Initialize và load trained model"""
        try:
            if model_path is None:
                try:
                    from flask import current_app
                    model_path = current_app.config.get('MODEL_PATH', '../models/real_data_question_model.pkl')
                except RuntimeError:
                    # Working outside of application context
                    model_path = '../models/real_data_question_model.pkl'
            
            # Convert relative path to absolute
            if not os.path.isabs(model_path):
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                model_path = os.path.join(base_dir, model_path)
            
            self.logger.info(f"Loading model from: {model_path}")
            
            if not os.path.exists(model_path):
                self.logger.error(f"Model file not found: {model_path}")
                return False
            
            # Load model data
            with open(model_path, 'rb') as f:
                self.model_data = pickle.load(f)
            
            # Create trainer instance and load components
            from app.utils.trainer import BackendQuestionTrainer
            self.trainer = BackendQuestionTrainer()
            
            # Load all components from model data
            self.trainer.keyword_vectorizer = self.model_data['keyword_vectorizer']
            self.trainer.question_vectorizer = self.model_data['question_vectorizer']
            self.trainer.category_classifier = self.model_data['category_classifier']
            self.trainer.similarity_model = self.model_data['similarity_model']
            self.trainer.label_encoder = self.model_data['label_encoder']
            self.trainer.keyword_question_mapping = self.model_data['keyword_question_mapping']
            self.trainer.real_questions_db = self.model_data['real_questions_db']
            self.trainer.category_keywords = self.model_data['category_keywords']
            
            self.model_loaded = True
            self.logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            self.model_loaded = False
            return False
    
    def is_model_loaded(self) -> bool:
        """Kiểm tra xem model đã được load chưa"""
        return self.model_loaded and self.trainer is not None
    
    def generate_questions(self, keyword: str, num_questions: int = 5, category: str = None) -> List[Dict[str, Any]]:
        """Generate questions từ keyword"""
        if not self.is_model_loaded():
            raise RuntimeError("Model chưa được load. Hãy gọi initialize_model() trước.")
        
        try:
            # Limit số questions
            num_questions = min(num_questions, 20)  # Max 20 questions
            
            # Generate questions using trained model
            generated_questions = self.trainer.generate_questions_from_real_data(
                keyword, num_questions
            )
            
            return [
                {
                    'question': q['question'],
                    'category': q['category'],
                    'confidence': round(q['confidence'], 4),
                    'method': q['method'],
                    'source_keyword': q.get('source_keyword'),
                    'similarity_score': q.get('similarity_score')
                }
                for q in generated_questions
            ]
            
        except Exception as e:
            self.logger.error(f"Error generating questions for '{keyword}': {str(e)}")
            raise RuntimeError(f"Lỗi khi tạo câu hỏi: {str(e)}")
    
    def predict_category(self, keyword: str) -> Dict[str, Any]:
        """Predict category cho keyword"""
        if not self.is_model_loaded():
            raise RuntimeError("Model chưa được load. Hãy gọi initialize_model() trước.")
        
        try:
            category, confidence = self.trainer.predict_category(keyword)
            
            # Get all probabilities if possible
            all_probabilities = {}
            try:
                if self.trainer.question_vectorizer and self.trainer.category_classifier:
                    keyword_vec = self.trainer.question_vectorizer.transform([keyword])
                    probabilities = self.trainer.category_classifier.predict_proba(keyword_vec)[0]
                    
                    for i, prob in enumerate(probabilities):
                        cat_name = self.trainer.label_encoder.inverse_transform([i])[0]
                        all_probabilities[cat_name] = round(float(prob), 4)
            except Exception as e:
                self.logger.warning(f"Could not get all probabilities: {str(e)}")
            
            return {
                'keyword': keyword,
                'predicted_category': category,
                'confidence': round(confidence, 4),
                'all_probabilities': all_probabilities
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting category for '{keyword}': {str(e)}")
            raise RuntimeError(f"Lỗi khi dự đoán category: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Lấy thông tin về model đã load"""
        if not self.is_model_loaded():
            return {
                'model_loaded': False,
                'error': 'Model chưa được load'
            }
        
        try:
            model_info = self.model_data.get('model_info', {})
            training_date = self.model_data.get('training_date', 'Unknown')
            
            return {
                'model_loaded': True,
                'training_date': training_date,
                'total_keywords': model_info.get('total_keywords', 0),
                'total_questions': model_info.get('total_questions', 0),
                'categories': model_info.get('categories', []),
                'model_accuracy': None  # Could be added if stored in model
            }
            
        except Exception as e:
            self.logger.error(f"Error getting model info: {str(e)}")
            return {
                'model_loaded': True,
                'error': f'Lỗi khi lấy thông tin model: {str(e)}'
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Lấy trạng thái sức khỏe của service"""
        return {
            'status': 'healthy' if self.is_model_loaded() else 'unhealthy',
            'model_loaded': self.is_model_loaded(),
            'timestamp': datetime.now().isoformat()
        }