#!/usr/bin/env python3
"""
Simple Question AI Trainer
Train lightweight model để từ keyword generate câu hỏi
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleQuestionAI:
    """Simple AI để generate câu hỏi từ keyword"""
    
    def __init__(self):
        self.category_model = None
        self.vectorizer = None
        self.trained_questions_db = None  # Database of trained questions
        self.keyword_to_questions = {}    # Map keywords to questions
        print("🤖 Simple Question AI initialized!")
    


    def load_training_data(self, max_samples=1000000):
        """Load training data từ datasets folder - sử dụng dữ liệu thật đã train"""
        
        print(f"📊 Loading training data from datasets (max {max_samples:,} samples)...")
        
        # Load từ datasets folder (50 triệu dữ liệu)
        datasets_dir = "datasets"
        if not os.path.exists(datasets_dir):
            print("❌ No datasets folder found!")
            return None
        
        # Load các batch files
        batch_files = [f for f in os.listdir(datasets_dir) 
                      if f.startswith("batch_") and f.endswith(".csv")]
        batch_files.sort()  # Đảm bảo thứ tự
        
        all_data = []
        total_loaded = 0
        
        print(f"📁 Found {len(batch_files)} batch files")
        
        for file in batch_files:
            if total_loaded >= max_samples:
                break
                
            file_path = os.path.join(datasets_dir, file)
            try:
                df = pd.read_csv(file_path)
                
                # Kiểm tra columns - adapt to dataset structure
                if 'keyword' in df.columns and 'category' in df.columns and 'form_title' in df.columns:
                    # Convert form data to question format
                    df['question'] = df['form_title'].apply(lambda x: self._form_title_to_question(x))
                    remaining = max_samples - total_loaded
                    if len(df) > remaining:
                        df = df.head(remaining)
                    
                    all_data.append(df)
                    total_loaded += len(df)
                    print(f"   ✅ {file}: {len(df):,} records")
                else:
                    print(f"   ⚠️ {file}: Missing required columns")
                
            except Exception as e:
                print(f"   ❌ Error loading {file}: {e}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"📊 Total loaded: {len(combined_df):,} questions")
            
            # Tạo keyword mapping cho generate_questions
            self._build_keyword_mapping(combined_df)
            return combined_df
        
        return None

    def _build_keyword_mapping(self, df):
        """Xây dựng mapping từ keyword đến questions"""
        
        print("🗺️ Building keyword to questions mapping...")
        
        # Group by keyword
        grouped = df.groupby('keyword')
        
        for keyword, group in grouped:
            questions_list = []
            for _, row in group.iterrows():
                questions_list.append({
                    'question': row['question'],
                    'category': row['category'],
                    'keyword': keyword
                })
            
            self.keyword_to_questions[keyword.lower()] = questions_list
        
        print(f"   📊 Mapped {len(self.keyword_to_questions):,} unique keywords")
        
        # Store full dataset for similarity search
        self.trained_questions_db = df

    def train_category_model(self, df):
        """Train simple category classification"""
        
        print("🎯 Training category classifier...")
        
        # Prepare data
        X = df['keyword'].fillna('') + ' ' + df['question'].fillna('')
        y = df['category']
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train
        self.vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
        X_train_vec = self.vectorizer.fit_transform(X_train)
        
        self.category_model = MultinomialNB()
        self.category_model.fit(X_train_vec, y_train)
        
        # Test
        X_test_vec = self.vectorizer.transform(X_test)
        y_pred = self.category_model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"   📊 Accuracy: {accuracy:.3f}")
        print("   📋 Report:")
        print(classification_report(y_test, y_pred))
        
        return accuracy

    def predict_category(self, keyword):
        """Predict category từ keyword"""
        
        if not self.category_model:
            return "it", 0.33
        
        keyword_vec = self.vectorizer.transform([keyword])
        category = self.category_model.predict(keyword_vec)[0]
        probabilities = self.category_model.predict_proba(keyword_vec)[0]
        confidence = max(probabilities)
        
        return category, confidence

    def generate_questions(self, keyword, num_questions=5):
        """Generate questions từ trained dataset - KHÔNG dùng templates"""
        
        print(f"🎯 Generating questions for: '{keyword}' from trained data")
        
        # Predict category
        category, confidence = self.predict_category(keyword)
        print(f"   📂 Predicted category: {category} (confidence: {confidence:.3f})")
        
        questions = []
        keyword_lower = keyword.lower()
        
        # Method 1: Exact keyword match
        if keyword_lower in self.keyword_to_questions:
            exact_matches = self.keyword_to_questions[keyword_lower]
            if len(exact_matches) >= num_questions:
                import random
                selected = random.sample(exact_matches, num_questions)
                questions.extend(selected)
                print(f"   ✅ Found {len(selected)} exact matches")
                return questions
            else:
                questions.extend(exact_matches)
                num_questions -= len(exact_matches)
                print(f"   ✅ Found {len(exact_matches)} exact matches, need {num_questions} more")
        
        # Method 2: Partial keyword matching
        if num_questions > 0:
            partial_matches = self._find_partial_matches(keyword, category, num_questions)
            questions.extend(partial_matches)
            print(f"   ✅ Found {len(partial_matches)} partial matches")
        
        # Method 3: Category-based fallback if still not enough
        if len(questions) < num_questions:
            category_matches = self._find_category_matches(category, num_questions - len(questions))
            questions.extend(category_matches)
            print(f"   ✅ Added {len(category_matches)} category matches")
        
        # Ensure unique questions and proper format
        unique_questions = []
        seen_questions = set()
        
        for q in questions:
            question_text = q['question'] if isinstance(q, dict) else str(q)
            if question_text not in seen_questions:
                seen_questions.add(question_text)
                unique_questions.append({
                    "keyword": keyword,
                    "question": question_text,
                    "category": q.get('category', category) if isinstance(q, dict) else category,
                    "confidence": confidence
                })
        
        # Limit to requested number
        final_questions = unique_questions[:num_questions]
        print(f"   🎯 Returning {len(final_questions)} unique questions")
        
        return final_questions

    def _find_partial_matches(self, keyword, category, num_needed):
        """Tìm questions có keyword tương tự"""
        
        if not self.trained_questions_db is not None:
            return []
        
        import random
        matches = []
        keyword_words = keyword.lower().split()
        
        # Search trong database
        for _, row in self.trained_questions_db.iterrows():
            if len(matches) >= num_needed * 3:  # Lấy nhiều để có thể chọn
                break
                
            row_keyword = str(row['keyword']).lower()
            
            # Check if any word in keyword matches
            if any(word in row_keyword for word in keyword_words):
                matches.append({
                    'question': row['question'],
                    'category': row['category'],
                    'keyword': row['keyword']
                })
        
        # Shuffle and return
        random.shuffle(matches)
        return matches[:num_needed]

    def _find_category_matches(self, category, num_needed):
        """Tìm questions theo category"""
        
        if not hasattr(self, 'trained_questions_db') or self.trained_questions_db is None:
            return []
        
        import random
        
        # Filter by category
        category_df = self.trained_questions_db[self.trained_questions_db['category'] == category]
        
        if len(category_df) == 0:
            return []
        
        # Sample random questions
        sample_size = min(num_needed, len(category_df))
        sampled = category_df.sample(n=sample_size)
        
        matches = []
        for _, row in sampled.iterrows():
            matches.append({
                'question': row['question'],
                'category': row['category'],
                'keyword': row['keyword']
            })
        
        return matches
    
    def _create_varied_question(self, template, keyword):
        """Create varied question text"""
        import random
        
        # Basic substitution
        question = template.format(keyword=keyword)
        
        # Add some variation based on keyword type
        keyword_lower = keyword.lower()
        
        # If keyword is plural, sometimes use singular
        if keyword_lower.endswith('s') and len(keyword_lower) > 3:
            if random.random() < 0.3:  # 30% chance
                singular_keyword = keyword[:-1]
                question = template.format(keyword=singular_keyword)
        
        # If keyword has multiple words, sometimes use just the main word
        words = keyword.split()
        if len(words) > 1 and random.random() < 0.2:  # 20% chance
            main_word = words[-1] if len(words[-1]) > 3 else words[0]
            question = template.format(keyword=main_word)
        
        return question

    def _form_title_to_question(self, form_title):
        """Convert form title to question format"""
        
        import random
        
        # Templates để convert form title thành questions
        question_templates = [
            f"What is the purpose of {form_title.lower()}?",
            f"How do you complete {form_title.lower()}?", 
            f"What information is required for {form_title.lower()}?",
            f"What are the benefits of {form_title.lower()}?",
            f"How long does {form_title.lower()} take to complete?",
            f"What are the steps involved in {form_title.lower()}?",
            f"Who should fill out {form_title.lower()}?",
            f"What are the requirements for {form_title.lower()}?",
            f"How do you submit {form_title.lower()}?",
            f"What happens after completing {form_title.lower()}?"
        ]
        
        return random.choice(question_templates)

    def save_model(self):
        """Save trained model with question database"""
        
        os.makedirs("models", exist_ok=True)
        
        model_data = {
            'category_model': self.category_model,
            'vectorizer': self.vectorizer,
            'keyword_to_questions': self.keyword_to_questions,
            'trained_questions_db': self.trained_questions_db,
            'training_date': datetime.now().isoformat(),
            'model_type': 'Dataset-based Question AI',
            'total_keywords': len(self.keyword_to_questions),
            'total_questions': len(self.trained_questions_db) if self.trained_questions_db is not None else 0
        }
        
        model_path = "models/simple_question_ai.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model saved: {model_path}")
        print(f"   📊 Keywords: {len(self.keyword_to_questions):,}")
        print(f"   📊 Questions: {len(self.trained_questions_db) if self.trained_questions_db is not None else 0:,}")

    def load_model(self):
        """Load trained model with question database"""
        
        model_path = "models/simple_question_ai.pkl"
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.category_model = model_data.get('category_model')
                self.vectorizer = model_data.get('vectorizer')
                self.keyword_to_questions = model_data.get('keyword_to_questions', {})
                self.trained_questions_db = model_data.get('trained_questions_db')
                
                print(f"📥 Model loaded: {model_path}")
                print(f"   📊 Keywords: {len(self.keyword_to_questions):,}")
                if self.trained_questions_db is not None:
                    print(f"   📊 Questions: {len(self.trained_questions_db):,}")
                
                return True
                
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                return False
        
        print(f"❌ No model found: {model_path}")
        return False


def main():
    """Main training function"""
    
    print("🤖 Simple Question AI Trainer")
    print("=" * 40)
    
    # Initialize
    ai = SimpleQuestionAI()
    
    # Load training data
    df = ai.load_training_data(max_samples=20000)  # Limit to 20k samples
    
    if df is not None and len(df) > 100:
        print(f"\n📊 Dataset info:")
        print(f"   Records: {len(df):,}")
        print(f"   Categories: {df['category'].value_counts().to_dict()}")
        
        # Train
        accuracy = ai.train_category_model(df)
        
        if accuracy > 0.7:  # Only save if decent accuracy
            ai.save_model()
            
            # Test generation
            print(f"\n🧪 Testing question generation:")
            test_keywords = [
                "machine learning algorithms",
                "cryptocurrency investment",
                "social media marketing"
            ]
            
            for keyword in test_keywords:
                questions = ai.generate_questions(keyword, num_questions=3)
                
                print(f"\n🔍 '{keyword}':")
                for i, q in enumerate(questions, 1):
                    print(f"   {i}. {q['question']}")
        else:
            print(f"❌ Accuracy too low: {accuracy:.3f}")
    
    else:
        print("❌ Insufficient training data")
    
    print("\n✅ Training completed!")


if __name__ == "__main__":
    main()
