"""
Simplified Question Trainer for Backend
Chỉ chứa các methods cần thiết để load và sử dụng trained model
"""

import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

class BackendQuestionTrainer:
    """Simplified trainer class for backend model loading"""
    
    def __init__(self, datasets_path="datasets", questions_path="question_datasets", models_path="models"):
        self.datasets_path = Path(datasets_path)
        self.questions_path = Path(questions_path)
        self.models_path = Path(models_path)
        
        # AI components
        self.keyword_vectorizer = None
        self.question_vectorizer = None
        self.category_classifier = None
        self.similarity_model = None
        self.label_encoder = None
        
        # Data storage
        self.keyword_question_mapping = defaultdict(list)
        self.category_keywords = defaultdict(list)
        self.real_questions_db = []
    
    def find_similar_keywords(self, target_keyword, n_similar=5):
        """Tìm keywords tương tự từ training data"""
        if not self.keyword_vectorizer or not self.similarity_model:
            return []
        
        try:
            # Vectorize target keyword
            target_vector = self.keyword_vectorizer.transform([target_keyword])
            
            # Find similar keywords
            distances, indices = self.similarity_model.kneighbors(target_vector)
            
            # Get feature names (keywords)
            feature_names = self.keyword_vectorizer.get_feature_names_out()
            
            similar_keywords = []
            for i, idx in enumerate(indices[0]):
                if idx < len(feature_names):
                    similarity_score = 1 - distances[0][i]
                    similar_keywords.append({
                        'keyword': feature_names[idx],
                        'similarity': similarity_score
                    })
            
            return similar_keywords[:n_similar]
            
        except Exception as e:
            print(f"   ⚠️ Error finding similar keywords: {e}")
            return []
    
    def predict_category(self, keyword):
        """Predict category cho keyword"""
        if not self.category_classifier or not self.question_vectorizer:
            return 'it', 0.5
        
        try:
            # Vectorize keyword
            keyword_vec = self.question_vectorizer.transform([keyword])
            
            # Predict
            prediction = self.category_classifier.predict(keyword_vec)[0]
            probabilities = self.category_classifier.predict_proba(keyword_vec)[0]
            confidence = max(probabilities)
            
            # Decode category
            category = self.label_encoder.inverse_transform([prediction])[0]
            
            return category, confidence
            
        except Exception as e:
            print(f"   ⚠️ Error predicting category: {e}")
            return 'it', 0.5
    
    def adapt_question(self, original_question, source_keyword, target_keyword):
        """Thông minh adapt câu hỏi từ source keyword sang target keyword"""
        adapted = original_question.lower()
        
        # Replace exact keyword matches
        adapted = adapted.replace(source_keyword.lower(), target_keyword.lower())
        
        # Handle partial matches and related terms
        source_words = source_keyword.lower().split()
        target_words = target_keyword.lower().split()
        
        for source_word in source_words:
            if len(source_word) > 3:
                # Try to replace with most relevant target word
                if target_words:
                    best_target = target_words[0]  # Simple heuristic
                    adapted = adapted.replace(source_word, best_target)
        
        # Capitalize first letter
        if adapted:
            adapted = adapted[0].upper() + adapted[1:]
        
        return adapted
    
    def generate_questions_from_real_data(self, keyword, num_questions=5):
        """Generate câu hỏi dựa trên dữ liệu thực tế"""
        
        generated_questions = []
        
        # 1. Predict category
        category, confidence = self.predict_category(keyword)
        
        # 2. Tìm câu hỏi trực tiếp từ keyword
        direct_questions = self.keyword_question_mapping.get(keyword.lower(), [])
        
        for q_data in direct_questions[:num_questions]:
            generated_questions.append({
                'question': q_data['question'],
                'category': q_data['category'],
                'confidence': confidence,
                'method': 'direct_match',
                'source_keyword': keyword
            })
        
        # 3. Nếu chưa đủ câu hỏi, tìm từ similar keywords
        if len(generated_questions) < num_questions:
            similar_keywords = self.find_similar_keywords(keyword)
            
            for similar_kw in similar_keywords:
                similar_keyword = similar_kw['keyword']
                similarity_score = similar_kw['similarity']
                
                # Tìm câu hỏi của similar keyword
                similar_questions = self.keyword_question_mapping.get(similar_keyword, [])
                
                for q_data in similar_questions:
                    if len(generated_questions) >= num_questions:
                        break
                    
                    # Adapt câu hỏi
                    adapted_question = self.adapt_question(
                        q_data['question'], 
                        similar_keyword, 
                        keyword
                    )
                    
                    generated_questions.append({
                        'question': adapted_question,
                        'category': category,
                        'confidence': confidence * similarity_score,
                        'method': 'similarity_adaptation',
                        'source_keyword': similar_keyword,
                        'similarity_score': similarity_score
                    })
                
                if len(generated_questions) >= num_questions:
                    break
        
        # 4. Nếu vẫn chưa đủ, tìm từ cùng category
        if len(generated_questions) < num_questions:
            category_questions = [
                q for q in self.real_questions_db 
                if q['category'] == category and q['keyword'] != keyword.lower()
            ]
            
            for q_data in category_questions:
                if len(generated_questions) >= num_questions:
                    break
                
                adapted_question = self.adapt_question(
                    q_data['question'],
                    q_data['keyword'],
                    keyword
                )
                
                generated_questions.append({
                    'question': adapted_question,
                    'category': category,
                    'confidence': confidence * 0.7,
                    'method': 'category_adaptation',
                    'source_keyword': q_data['keyword']
                })
        
        return generated_questions[:num_questions]