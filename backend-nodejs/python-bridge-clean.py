#!/usr/bin/env python3
"""
Python Bridge for Node.js Backend
Uses enhanced data-driven question generation
"""

import sys
import json
import warnings
import traceback
import pickle
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# Import enhanced question generator
try:
    from enhanced_question_generator import generate_enhanced_questions, data_driven_generator
except ImportError:
    print("Could not import enhanced_question_generator")
    sys.exit(1)

class PythonModelBridge:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.is_loaded = False
        
    def health_check(self):
        """Check if model and system are working"""
        try:
            model_loaded = data_driven_generator.is_loaded
            
            return {
                "success": True,
                "status": "ok",
                "model_loaded": model_loaded,
                "timestamp": datetime.now().isoformat(),
                "model_path": str(self.model_path)
            }
        except Exception as e:
            return {
                "success": False,
                "status": "error",
                "model_loaded": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_model_info(self):
        """Get model information and statistics"""
        try:
            if not data_driven_generator.is_loaded:
                return {
                    "success": False,
                    "error": "Model not loaded"
                }
            
            model_data = data_driven_generator.model_data
            if not model_data:
                return {
                    "success": False,
                    "error": "No model data available"
                }
            
            model_info = model_data.get('model_info', {})
            
            return {
                "success": True,
                "model_loaded": True,
                "training_date": model_data.get('training_date', 'Unknown'),
                "total_keywords": model_info.get('total_keywords', 0),
                "total_questions": model_info.get('total_questions', 0),
                "categories": model_info.get('categories', [])
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def predict_category(self, keyword):
        """Predict category for a keyword using the original model"""
        try:
            if not data_driven_generator.is_loaded:
                return {
                    "success": False,
                    "error": "Model not loaded"
                }
            
            model_data = data_driven_generator.model_data
            question_vectorizer = model_data.get('question_vectorizer')
            category_classifier = model_data.get('category_classifier')
            label_encoder = model_data.get('label_encoder')
            
            if not all([question_vectorizer, category_classifier, label_encoder]):
                return {
                    "success": False,
                    "error": "Model components missing"
                }
            
            # Vectorize keyword
            keyword_vec = question_vectorizer.transform([keyword])
            
            # Predict
            prediction = category_classifier.predict(keyword_vec)[0]
            probabilities = category_classifier.predict_proba(keyword_vec)[0]
            
            # Get category name
            predicted_category = label_encoder.inverse_transform([prediction])[0]
            confidence = float(max(probabilities))
            
            # Get all probabilities
            all_probabilities = {}
            for i, prob in enumerate(probabilities):
                category_name = label_encoder.inverse_transform([i])[0]
                all_probabilities[category_name] = float(prob)
            
            return {
                "success": True,
                "keyword": keyword,
                "predicted_category": predicted_category,
                "confidence": confidence,
                "all_probabilities": all_probabilities
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def generate_questions(self, keyword, num_questions=5, category=None):
        """Generate questions using enhanced data-driven approach"""
        try:
            # Use enhanced question generator with real data
            questions = generate_enhanced_questions(
                keyword=keyword,
                category=category or 'it', 
                num_questions=num_questions
            )
            
            # Format for API response
            formatted_questions = []
            for q in questions:
                formatted_questions.append({
                    'question': q.get('question', ''),
                    'category': q.get('category', 'general'),
                    'confidence': float(q.get('confidence', 0.5)),
                    'method': q.get('method', 'unknown'),
                    'source_keyword': q.get('source', ''),
                    'similarity_score': None
                })
            
            return {
                'success': True,
                'questions': formatted_questions,
                'keyword': keyword,
                'total_generated': len(formatted_questions),
                'execution_time': 0.1  # Placeholder
            }
            
        except Exception as e:
            print(f"Error generating questions: {e}")
            return {
                'success': False,
                'questions': [],
                'keyword': keyword,
                'error': str(e),
                'total_generated': 0,
                'execution_time': 0
            }

def main():
    """Main function to handle CLI requests"""
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No command provided"}))
        sys.exit(1)
    
    command = sys.argv[1]
    
    # Model path
    model_path = "../models/real_data_question_model.pkl"
    if len(sys.argv) > 2:
        model_path = sys.argv[2]
    
    bridge = PythonModelBridge(model_path)
    
    try:
        if command == "health_check":
            result = bridge.health_check()
            
        elif command == "model_info":
            result = bridge.get_model_info()
            
        elif command == "predict_category":
            if len(sys.argv) < 3:
                result = {"error": "Keyword required for predict_category"}
            else:
                keyword = sys.argv[2]
                result = bridge.predict_category(keyword)
                
        elif command == "generate_questions":
            if len(sys.argv) < 3:
                result = {"error": "Keyword required for generate_questions"}
            else:
                keyword = sys.argv[2]
                num_questions = int(sys.argv[3]) if len(sys.argv) > 3 else 5
                category = sys.argv[4] if len(sys.argv) > 4 else None
                result = bridge.generate_questions(keyword, num_questions, category)
                
        else:
            result = {"error": f"Unknown command: {command}"}
            
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        print(json.dumps(error_result))
        sys.exit(1)

if __name__ == "__main__":
    main()