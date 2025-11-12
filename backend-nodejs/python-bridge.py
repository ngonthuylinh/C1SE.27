#!/usr/bin/env python3
"""
Python Bridge for Node.js Backend
Uses enhanced data-driven question generation
"""

import sys
import json
import warnings
import traceback
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# Import enhanced question generator
try:
    from enhanced_question_generator import generate_enhanced_questions, data_driven_generator
except ImportError:
    print("❌ Could not import enhanced_question_generator")
    sys.exit(1)

class PythonModelBridge:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.model_data = None
        self.is_loaded = False
        
    def load_model(self):
        """Load the trained model"""
        try:
            if not self.model_path.exists():
                return {
                    "success": False,
                    "error": f"Model file not found: {self.model_path}"
                }
            
            with open(self.model_path, 'rb') as f:
                self.model_data = pickle.load(f)
                
            self.is_loaded = True
            
            return {
                "success": True,
                "message": "Model loaded successfully",
                "model_info": self.model_data.get('model_info', {})
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to load model: {str(e)}",
                "traceback": traceback.format_exc()
            }
    
    def get_health_status(self):
        """Get health status of the model"""
        return {
            "status": "ok",
            "model_loaded": self.is_loaded,
            "timestamp": datetime.now().isoformat(),
            "model_path": str(self.model_path)
        }
    
    def get_model_info(self):
        """Get detailed model information"""
        if not self.is_loaded:
            return {
                "success": False,
                "error": "Model not loaded"
            }
        
        try:
            model_info = self.model_data.get('model_info', {})
            return {
                "success": True,
                "model_loaded": True,
                "training_date": self.model_data.get('training_date'),
                "total_keywords": model_info.get('total_keywords', 0),
                "total_questions": model_info.get('total_questions', 0),
                "categories": model_info.get('categories', [])
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def predict_category(self, keyword):
        """Predict category for a keyword"""
        if not self.is_loaded:
            return {
                "success": False,
                "error": "Model not loaded"
            }
        
        try:
            # Get model components
            question_vectorizer = self.model_data.get('question_vectorizer')
            category_classifier = self.model_data.get('category_classifier')
            label_encoder = self.model_data.get('label_encoder')
            
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
            print(f"❌ Error generating questions: {e}")
            return {
                'success': False,
                'questions': [],
                'keyword': keyword,
                'error': str(e),
                'total_generated': 0,
                'execution_time': 0
            }
                            "source_keyword": cat_keyword,
                            "similarity_score": 0.6
                        })
            
            return {
                "success": True,
                "questions": generated_questions[:num_questions],
                "keyword": keyword,
                "total_generated": len(generated_questions[:num_questions]),
                "predicted_category": predicted_category,
                "confidence": confidence
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _find_similar_keywords(self, target_keyword, keyword_mapping, max_similar=5):
        """Find similar keywords using simple string matching"""
        similar_keywords = []
        target_lower = target_keyword.lower()
        target_words = target_lower.split()
        
        for keyword in keyword_mapping.keys():
            if keyword == target_lower:
                continue
            
            # Check for word overlap
            keyword_words = keyword.split()
            common_words = set(target_words) & set(keyword_words)
            
            if common_words:
                similarity_score = len(common_words) / max(len(target_words), len(keyword_words))
                if similarity_score > 0.3:  # Threshold for similarity
                    similar_keywords.append(keyword)
            
            # Check for substring match
            elif target_lower in keyword or keyword in target_lower:
                similar_keywords.append(keyword)
            
            if len(similar_keywords) >= max_similar:
                break
        
        return similar_keywords[:max_similar]
    
    def _adapt_question(self, original_question, source_keyword, target_keyword):
        """Adapt question from source keyword to target keyword"""
        adapted = original_question.lower()
        
        # Replace exact keyword matches
        adapted = adapted.replace(source_keyword.lower(), target_keyword.lower())
        
        # Handle partial matches
        source_words = source_keyword.lower().split()
        target_words = target_keyword.lower().split()
        
        for source_word in source_words:
            if len(source_word) > 3:  # Only replace meaningful words
                for target_word in target_words:
                    if len(target_word) > 3:
                        adapted = adapted.replace(source_word, target_word)
                        break
        
        # Capitalize first letter
        if adapted:
            adapted = adapted[0].upper() + adapted[1:] if len(adapted) > 1 else adapted.upper()
        
        return adapted

def main():
    """Main function to handle command line arguments"""
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "No command provided"}))
        sys.exit(1)
    
    command = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "../models/real_data_question_model.pkl"
    
    # Initialize bridge
    bridge = PythonModelBridge(model_path)
    
    # Load model if not health check
    if command != "health":
        load_result = bridge.load_model()
        if not load_result.get("success"):
            print(json.dumps(load_result))
            sys.exit(1)
    
    # Handle commands
    try:
        if command == "health":
            result = bridge.get_health_status()
        
        elif command == "info":
            result = bridge.get_model_info()
        
        elif command == "predict":
            if len(sys.argv) < 4:
                result = {"success": False, "error": "Keyword required for prediction"}
            else:
                keyword = sys.argv[3]
                result = bridge.predict_category(keyword)
        
        elif command == "generate":
            if len(sys.argv) < 4:
                result = {"success": False, "error": "Keyword required for generation"}
            else:
                keyword = sys.argv[3]
                num_questions = int(sys.argv[4]) if len(sys.argv) > 4 else 5
                category = sys.argv[5] if len(sys.argv) > 5 else None
                result = bridge.generate_questions(keyword, num_questions, category)
        
        else:
            result = {"success": False, "error": f"Unknown command: {command}"}
        
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