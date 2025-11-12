#!/usr/bin/env python3
"""
Question Generation AI Trainer
Train AI model để từ keyword tự động generate câu hỏi phù hợp
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuestionGenerationTrainer:
    """AI Trainer cho question generation từ keywords"""
    
    def __init__(self, model_dir="models"):
        """Initialize trainer"""
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Models components
        self.category_model = None
        self.question_type_model = None
        self.question_pattern_model = None
        self.vectorizers = {}
        self.encoders = {}
        
        # Question templates cho generation
        self.setup_question_templates()
        
        print("🤖 Question Generation Trainer initialized!")
    
    def setup_question_templates(self):
        """Setup question templates cho từng category và type"""
        
        self.question_templates = {
            "it": {
                "how_to": [
                    "How to implement {keyword} in production environment?",
                    "How to optimize {keyword} for better performance?", 
                    "How to troubleshoot common {keyword} issues?",
                    "How to integrate {keyword} with existing systems?",
                    "How to scale {keyword} for large applications?",
                    "How to secure {keyword} implementations?",
                    "How to monitor and maintain {keyword} systems?",
                    "How to migrate from legacy systems to {keyword}?",
                    "How to test {keyword} functionality effectively?",
                    "How to deploy {keyword} using modern DevOps practices?"
                ],
                "what_is": [
                    "What is {keyword} and how does it work?",
                    "What are the key benefits of using {keyword}?",
                    "What are the main challenges with {keyword}?",
                    "What tools are best for {keyword} development?",
                    "What skills are needed to master {keyword}?",
                    "What are the latest trends in {keyword}?",
                    "What companies are leading in {keyword} innovation?",
                    "What are the alternatives to {keyword}?",
                    "What certifications are available for {keyword}?",
                    "What is the future of {keyword} technology?"
                ],
                "best_practices": [
                    "What are the best practices for {keyword} implementation?",
                    "What security considerations are important for {keyword}?",
                    "What performance optimization techniques work for {keyword}?",
                    "What are common mistakes to avoid with {keyword}?",
                    "What coding standards should be followed for {keyword}?",
                    "What testing strategies are recommended for {keyword}?",
                    "What documentation practices are essential for {keyword}?",
                    "What monitoring approaches work best for {keyword}?",
                    "What backup and recovery strategies suit {keyword}?",
                    "What team collaboration methods enhance {keyword} projects?"
                ]
            },
            "economics": {
                "analysis": [
                    "How to analyze {keyword} market trends effectively?",
                    "What factors influence {keyword} performance?",
                    "How to evaluate {keyword} investment opportunities?",
                    "What metrics are important for {keyword} assessment?",
                    "How to forecast {keyword} future performance?",
                    "What are the risks associated with {keyword}?",
                    "How to diversify {keyword} investment portfolio?",
                    "What economic indicators affect {keyword}?",
                    "How to calculate {keyword} return on investment?",
                    "What market conditions favor {keyword} growth?"
                ],
                "strategy": [
                    "What investment strategy works best for {keyword}?",
                    "How to develop a {keyword} financial plan?",
                    "What allocation percentage should {keyword} have in portfolio?",
                    "How to time {keyword} market entry and exit?",
                    "What hedging strategies protect {keyword} investments?",
                    "How to rebalance portfolio including {keyword}?",
                    "What tax implications exist for {keyword} investments?",
                    "How to research {keyword} before investing?",
                    "What professional advice is needed for {keyword}?",
                    "How to monitor {keyword} investment performance?"
                ],
                "fundamentals": [
                    "What are the fundamentals of {keyword}?",
                    "How does {keyword} impact overall economic growth?",
                    "What role does {keyword} play in financial markets?",
                    "How to understand {keyword} valuation methods?",
                    "What are the key drivers of {keyword} prices?",
                    "How does inflation affect {keyword} investments?",
                    "What are the liquidity considerations for {keyword}?",
                    "How to assess {keyword} market volatility?",
                    "What are the regulatory aspects of {keyword}?",
                    "How does global economy influence {keyword}?"
                ]
            },
            "marketing": {
                "campaign": [
                    "How to create effective {keyword} marketing campaigns?",
                    "What budget allocation works best for {keyword} marketing?",
                    "How to measure {keyword} campaign performance?",
                    "What channels are most effective for {keyword} promotion?",
                    "How to target the right audience for {keyword}?",
                    "What creative approaches work for {keyword} marketing?",
                    "How to optimize {keyword} campaign timing?",
                    "What A/B testing strategies improve {keyword} results?",
                    "How to scale successful {keyword} campaigns?",
                    "What ROI expectations are realistic for {keyword}?"
                ],
                "strategy": [
                    "What marketing strategy maximizes {keyword} success?",
                    "How to position {keyword} against competitors?",
                    "What brand messaging resonates with {keyword} audience?",
                    "How to integrate {keyword} across marketing channels?",
                    "What customer journey optimization improves {keyword}?",
                    "How to personalize {keyword} marketing experiences?",
                    "What content strategy supports {keyword} goals?",
                    "How to leverage social media for {keyword} marketing?",
                    "What partnerships enhance {keyword} marketing efforts?",
                    "How to adapt {keyword} strategy for different markets?"
                ],
                "analytics": [
                    "How to track {keyword} marketing performance?",
                    "What KPIs are most important for {keyword} campaigns?",
                    "How to analyze {keyword} customer behavior data?",
                    "What attribution models work for {keyword} marketing?",
                    "How to optimize {keyword} conversion funnels?",
                    "What tools provide best {keyword} marketing insights?",
                    "How to predict {keyword} campaign success?",
                    "What segmentation strategies improve {keyword} targeting?",
                    "How to measure {keyword} brand awareness impact?",
                    "What competitive analysis methods suit {keyword}?"
                ]
            }
        }
        
        print("📝 Question templates loaded for all categories and types")

    def load_training_data(self, question_datasets_dir="question_datasets"):
        """Load question datasets cho training"""
        
        print("📊 Loading question datasets for training...")
        
        all_data = []
        
        # Load từ batch files nếu có
        if os.path.exists(question_datasets_dir):
            # Load sample files trước
            sample_files = [f for f in os.listdir(question_datasets_dir) 
                          if f.startswith("question_sample") and f.endswith(".csv")]
            
            for file in sample_files:
                file_path = os.path.join(question_datasets_dir, file)
                try:
                    df = pd.read_csv(file_path)
                    all_data.append(df)
                    print(f"   ✅ Loaded: {file} ({len(df)} records)")
                except Exception as e:
                    print(f"   ❌ Error loading {file}: {e}")
            
            # Nếu có batch files, load một số batch
            batch_files = sorted([f for f in os.listdir(question_datasets_dir) 
                                if f.startswith("question_batch") and f.endswith(".csv")])
            
            if batch_files:
                # Load first 5 batches cho training
                for batch_file in batch_files[:5]:
                    file_path = os.path.join(question_datasets_dir, batch_file)
                    try:
                        df = pd.read_csv(file_path)
                        all_data.append(df)
                        print(f"   ✅ Loaded: {batch_file} ({len(df)} records)")
                    except Exception as e:
                        print(f"   ❌ Error loading {batch_file}: {e}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"📊 Total training data: {len(combined_df)} questions")
            return combined_df
        else:
            print("⚠️  No question data found. Creating sample data...")
            return self._create_sample_training_data()
    
    def _create_sample_training_data(self):
        """Tạo sample training data nếu chưa có"""
        
        sample_data = []
        
        # Tạo sample questions từ templates
        categories = ["it", "economics", "marketing"]
        
        for category in categories:
            for question_type in self.question_templates[category]:
                templates = self.question_templates[category][question_type]
                
                # Sample keywords cho category
                if category == "it":
                    keywords = ["cloud computing", "machine learning", "web development", "cybersecurity"]
                elif category == "economics":
                    keywords = ["investment portfolio", "market analysis", "financial planning", "risk management"] 
                else:
                    keywords = ["digital marketing", "social media", "content marketing", "brand management"]
                
                for keyword in keywords:
                    for template in templates[:3]:  # Take first 3 templates
                        question = template.format(keyword=keyword)
                        
                        sample_data.append({
                            "keyword": keyword,
                            "category": category,
                            "question_type": question_type,
                            "question": question,
                            "difficulty": np.random.choice(["Beginner", "Intermediate", "Advanced"]),
                            "question_length": len(question),
                            "word_count": len(question.split())
                        })
        
        df = pd.DataFrame(sample_data)
        print(f"📊 Created sample training data: {len(df)} questions")
        return df

    def train_models(self, df):
        """Train các AI models"""
        
        print("\n🎯 Training AI Models...")
        
        # Prepare features
        X_text = df['keyword'].fillna('') + ' ' + df['question'].fillna('')
        
        # Train Category Classification Model
        print("\n1️⃣ Training Category Classification Model...")
        y_category = df['category']
        self._train_category_model(X_text, y_category)
        
        # Train Question Type Classification Model  
        print("\n2️⃣ Training Question Type Model...")
        y_question_type = df['question_type']
        self._train_question_type_model(X_text, y_question_type)
        
        # Train Question Pattern Model (based on structure)
        print("\n3️⃣ Training Question Pattern Model...")
        y_difficulty = df['difficulty'] 
        self._train_pattern_model(X_text, y_difficulty)
        
        # Save all models
        self.save_models()
        
        print("\n✅ All models trained and saved!")
    
    def _train_category_model(self, X, y):
        """Train category classification model"""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create pipeline
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Train
        X_train_vec = vectorizer.fit_transform(X_train)
        classifier.fit(X_train_vec, y_train)
        
        # Evaluate
        X_test_vec = vectorizer.transform(X_test)
        y_pred = classifier.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"   📊 Category Model Accuracy: {accuracy:.3f}")
        print("   📋 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save components
        self.category_model = classifier
        self.vectorizers['category'] = vectorizer
    
    def _train_question_type_model(self, X, y):
        """Train question type model"""
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        vectorizer = CountVectorizer(max_features=3000, ngram_range=(1, 3))
        classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
        
        X_train_vec = vectorizer.fit_transform(X_train)
        classifier.fit(X_train_vec, y_train)
        
        X_test_vec = vectorizer.transform(X_test)
        y_pred = classifier.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"   📊 Question Type Model Accuracy: {accuracy:.3f}")
        
        self.question_type_model = classifier
        self.vectorizers['question_type'] = vectorizer
    
    def _train_pattern_model(self, X, y):
        """Train question pattern/difficulty model"""
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
        classifier = LogisticRegression(random_state=42, max_iter=1000)
        
        X_train_vec = vectorizer.fit_transform(X_train)
        classifier.fit(X_train_vec, y_train)
        
        X_test_vec = vectorizer.transform(X_test)
        y_pred = classifier.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"   📊 Question Pattern Model Accuracy: {accuracy:.3f}")
        
        self.question_pattern_model = classifier
        self.vectorizers['pattern'] = vectorizer

    def generate_questions_from_keyword(self, keyword, num_questions=10):
        """Generate questions từ keyword sử dụng trained models"""
        
        if not self.category_model:
            print("❌ Models not trained yet!")
            return []
        
        print(f"\n🎯 Generating {num_questions} questions for: '{keyword}'")
        
        # Predict category
        keyword_vec = self.vectorizers['category'].transform([keyword])
        predicted_category = self.category_model.predict(keyword_vec)[0]
        category_proba = self.category_model.predict_proba(keyword_vec)[0]
        
        print(f"   📂 Predicted Category: {predicted_category}")
        
        # Predict question type
        question_type_vec = self.vectorizers['question_type'].transform([keyword])
        predicted_type = self.question_type_model.predict(question_type_vec)[0]
        
        print(f"   🎯 Predicted Type: {predicted_type}")
        
        # Generate questions using templates
        generated_questions = []
        
        if predicted_category in self.question_templates:
            if predicted_type in self.question_templates[predicted_category]:
                templates = self.question_templates[predicted_category][predicted_type]
            else:
                # Use all templates if type not found
                all_templates = []
                for qtype in self.question_templates[predicted_category]:
                    all_templates.extend(self.question_templates[predicted_category][qtype])
                templates = all_templates
            
            # Generate questions
            import random
            selected_templates = random.sample(templates, min(num_questions, len(templates)))
            
            for template in selected_templates:
                question = template.format(keyword=keyword)
                
                # Predict difficulty
                q_vec = self.vectorizers['pattern'].transform([keyword + ' ' + question])
                difficulty = self.question_pattern_model.predict(q_vec)[0]
                
                generated_questions.append({
                    "keyword": keyword,
                    "question": question,
                    "category": predicted_category,
                    "question_type": predicted_type,
                    "difficulty": difficulty,
                    "confidence": max(category_proba)
                })
        
        print(f"   ✅ Generated {len(generated_questions)} questions")
        return generated_questions

    def save_models(self):
        """Save trained models"""
        
        print("\n💾 Saving trained models...")
        
        models_data = {
            'category_model': self.category_model,
            'question_type_model': self.question_type_model,
            'question_pattern_model': self.question_pattern_model,
            'vectorizers': self.vectorizers,
            'question_templates': self.question_templates,
            'training_date': datetime.now().isoformat(),
            'model_info': {
                'framework': 'scikit-learn',
                'purpose': 'Question Generation from Keywords',
                'categories': ['it', 'economics', 'marketing']
            }
        }
        
        # Save to pickle
        model_path = os.path.join(self.model_dir, "question_generation_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(models_data, f)
        
        print(f"   ✅ Models saved to: {model_path}")
        
        # Save model metadata
        metadata = {
            'model_type': 'Question Generation AI',
            'categories': ['it', 'economics', 'marketing'],
            'question_types': {
                'it': list(self.question_templates['it'].keys()),
                'economics': list(self.question_templates['economics'].keys()),
                'marketing': list(self.question_templates['marketing'].keys())
            },
            'training_date': datetime.now().isoformat(),
            'model_files': ['question_generation_model.pkl'],
            'usage': 'Load model and call generate_questions_from_keyword(keyword, num_questions)'
        }
        
        metadata_path = os.path.join(self.model_dir, "question_model_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✅ Metadata saved to: {metadata_path}")

    def load_models(self):
        """Load pre-trained models"""
        
        model_path = os.path.join(self.model_dir, "question_generation_model.pkl")
        
        if os.path.exists(model_path):
            print(f"📥 Loading models from: {model_path}")
            
            with open(model_path, 'rb') as f:
                models_data = pickle.load(f)
            
            self.category_model = models_data['category_model']
            self.question_type_model = models_data['question_type_model'] 
            self.question_pattern_model = models_data['question_pattern_model']
            self.vectorizers = models_data['vectorizers']
            self.question_templates = models_data['question_templates']
            
            print("   ✅ Models loaded successfully!")
            return True
        else:
            print(f"   ❌ No trained models found at: {model_path}")
            return False


def main():
    """Main training function"""
    
    print("🤖 Question Generation AI Trainer")
    print("=" * 50)
    
    # Initialize trainer
    trainer = QuestionGenerationTrainer()
    
    # Load training data
    df = trainer.load_training_data()
    
    if len(df) > 0:
        print(f"\n📊 Dataset Summary:")
        print(f"   Total Questions: {len(df):,}")
        print(f"   Categories: {df['category'].value_counts().to_dict()}")
        print(f"   Question Types: {len(df['question_type'].unique())}")
        
        # Train models
        trainer.train_models(df)
        
        # Test with sample keywords
        print("\n🧪 Testing Question Generation:")
        test_keywords = [
            "machine learning algorithms",
            "cryptocurrency investment", 
            "social media advertising"
        ]
        
        for keyword in test_keywords:
            questions = trainer.generate_questions_from_keyword(keyword, num_questions=3)
            
            print(f"\n🔍 Keyword: '{keyword}'")
            for i, q in enumerate(questions, 1):
                print(f"   {i}. {q['question']}")
                print(f"      Category: {q['category']}, Type: {q['question_type']}, Difficulty: {q['difficulty']}")
    
    else:
        print("❌ No training data available!")
    
    print("\n✅ Training completed!")


if __name__ == "__main__":
    main()
