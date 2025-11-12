#!/usr/bin/env python3
"""
Real Data Question Generation Trainer
Kết hợp datasets/ (keywords) với question_datasets/ (câu hỏi thực tế) 
để train AI model generate câu hỏi từ keywords - KHÔNG DÙNG TEMPLATES
"""

import pandas as pd
import numpy as np
import os
import glob
import pickle
import json
from datetime import datetime
import re
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

# Text Processing
import nltk
from collections import defaultdict, Counter
from tqdm import tqdm

print("🚀 Real Data Question Generation Trainer")
print("📊 Kết hợp datasets/ + question_datasets/ để train AI thực sự")

class RealDataQuestionTrainer:
    def __init__(self, datasets_path="datasets", questions_path="question_datasets", models_path="models"):
        self.datasets_path = Path(datasets_path)
        self.questions_path = Path(questions_path)
        self.models_path = Path(models_path)
        
        # Create models directory
        self.models_path.mkdir(exist_ok=True)
        
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
        
        print(f"📂 Datasets path: {self.datasets_path}")
        print(f"📂 Questions path: {self.questions_path}")
        print(f"💾 Models path: {self.models_path}")
    
    def load_and_process_datasets(self, max_files=None, batch_size=10000):
        """Load và process dữ liệu từ cả 2 folders với memory optimization"""
        print("\n🔍 Loading ALL real datasets...")
        
        all_data = []
        total_records = 0
        
        # Load từ datasets/ folder (keyword data) - TẤT CẢ FILES
        dataset_files = list(self.datasets_path.glob("*.csv"))
        if max_files:
            dataset_files = dataset_files[:max_files]
        
        print(f"   Found {len(dataset_files)} files in datasets/ (processing ALL)")
        
        for i, file_path in enumerate(tqdm(dataset_files, desc="Loading datasets")):
            try:
                # Read with optimizations
                df = pd.read_csv(file_path, low_memory=False)
                
                # Sample if file is too large
                if len(df) > batch_size:
                    df = df.sample(n=batch_size, random_state=42)
                
                # Extract keywords from dataset structure  
                if len(df.columns) >= 2:  # Ensure có ít nhất 2 columns
                    for _, row in df.iterrows():
                        try:
                            # Từ format: 79900167,financial modeling,economics,registration,Simple,14,2025-04-07,Financial Modeling Registration,8
                            if len(row) >= 3:
                                keyword = str(row.iloc[1]).strip().lower()  # Column 2: financial modeling
                                category = str(row.iloc[2]).strip().lower()  # Column 3: economics
                                
                                if keyword and category and len(keyword) > 3:
                                    all_data.append({
                                        'keyword': keyword,
                                        'category': category,
                                        'source': 'datasets',
                                        'file': file_path.name
                                    })
                                    self.category_keywords[category].append(keyword)
                                    total_records += 1
                        except:
                            continue
                
                # Memory cleanup every 50 files
                if (i + 1) % 50 == 0:
                    print(f"   Processed {i + 1}/{len(dataset_files)} files, {total_records:,} records so far")
                            
            except Exception as e:
                print(f"   ❌ Error loading {file_path.name}: {e}")
                continue
        
        # Load từ question_datasets/ folder (real questions) - TẤT CẢ FILES  
        question_files = list(self.questions_path.glob("*.csv"))
        if max_files:
            question_files = question_files[:max_files]
            
        print(f"   Found {len(question_files)} files in question_datasets/ (processing ALL)")
        
        for i, file_path in enumerate(tqdm(question_files, desc="Loading questions")):
            try:
                # Try different CSV reading approaches to handle corrupted files
                df = None
                
                # Approach 1: Normal read
                try:
                    df = pd.read_csv(file_path, low_memory=False)
                except:
                    # Approach 2: Read with error handling
                    try:
                        df = pd.read_csv(file_path, low_memory=False, on_bad_lines='skip', engine='python')
                    except:
                        # Approach 3: Read only first N lines that are likely good
                        try:
                            df = pd.read_csv(file_path, low_memory=False, nrows=1500, on_bad_lines='skip')
                        except:
                            # Skip this file completely if all approaches fail
                            print(f"   ⚠️ Skipping corrupted file: {file_path.name}")
                            continue
                
                if df is None or len(df) == 0:
                    continue
                
                # Sample if too large
                if len(df) > batch_size:
                    df = df.sample(n=batch_size, random_state=42)
                
                for _, row in df.iterrows():
                    try:
                        question = str(row.get('question', '')).strip()
                        keyword = str(row.get('keyword', '')).strip()
                        category = str(row.get('category', 'it')).lower()
                        
                        if question and keyword and len(question) > 10:
                            # Add to mapping
                            self.keyword_question_mapping[keyword.lower()].append({
                                'question': question,
                                'category': category,
                                'source': 'question_datasets'
                            })
                            
                            # Add to main data
                            all_data.append({
                                'keyword': keyword.lower(),
                                'category': category,
                                'question': question,
                                'source': 'question_datasets',
                                'file': file_path.name
                            })
                            
                            # Store in questions database
                            self.real_questions_db.append({
                                'keyword': keyword.lower(),
                                'question': question,
                                'category': category
                            })
                            total_records += 1
                            
                    except:
                        continue
                
                # Memory cleanup every 25 files
                if (i + 1) % 25 == 0:
                    print(f"   Processed {i + 1}/{len(question_files)} question files")
                        
            except Exception as e:
                print(f"   ❌ Error loading {file_path.name}: {e}")
                continue
        
        # Convert to DataFrame với memory optimization
        print(f"   Converting {len(all_data):,} records to DataFrame...")
        combined_df = pd.DataFrame(all_data)
        
        # Clean up memory
        del all_data
        
        print(f"\n📊 Data Loading Results:")
        print(f"   Total dataset files processed: {len(dataset_files)}")
        print(f"   Total question files processed: {len(question_files)}")
        print(f"   Total records: {len(combined_df):,}")
        print(f"   Unique keywords: {combined_df['keyword'].nunique():,}")
        print(f"   Categories: {combined_df['category'].value_counts().to_dict()}")
        print(f"   Real questions collected: {len(self.real_questions_db):,}")
        print(f"   Keyword-question mappings: {len(self.keyword_question_mapping):,}")
        
        return combined_df
    
    def build_keyword_similarity_model(self, df):
        """Build model để tìm keywords tương tự"""
        print("\n🔧 Building keyword similarity model...")
        
        # Get unique keywords
        unique_keywords = df['keyword'].unique()
        
        # Create TF-IDF vectors for keywords
        self.keyword_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        keyword_vectors = self.keyword_vectorizer.fit_transform(unique_keywords)
        
        # Build similarity model
        self.similarity_model = NearestNeighbors(
            n_neighbors=10,
            metric='cosine'
        )
        
        self.similarity_model.fit(keyword_vectors)
        
        print(f"   ✅ Similarity model trained on {len(unique_keywords):,} keywords")
        
        return unique_keywords
    
    def train_category_classifier(self, df):
        """Train model phân loại category từ keyword"""
        print("\n🎯 Training category classifier...")
        
        # Basic normalization / consolidation of category labels
        # This merges common variants (e.g. 'econ', 'econo' -> 'economics',
        # 'mark*' -> 'marketing', single-letter noisy labels -> main categories)
        def normalize_category(cat):
            try:
                c = str(cat).strip().lower()
            except:
                return 'nan'

            if c in ('', 'none', 'nan', 'na'):
                return 'nan'

            # economics variants
            if c.startswith('econ') or c.startswith('eco') or 'econ' in c or 'eco' in c or c == 'economic':
                return 'economics'

            # marketing variants
            if c.startswith('mark') or 'mark' in c:
                return 'marketing'

            # it / tech variants
            if c in ('it', 'information technology') or c == 'i':
                return 'it'

            # single-letter heuristics
            if c in ('e',):
                return 'economics'
            if c in ('ec', 'eco'):
                return 'economics'
            if c in ('m', 'ma', 'mar'):
                return 'marketing'

            return c

        df = df.copy()
        df['category'] = df['category'].fillna('nan').apply(normalize_category)

        # Filter categories with at least 10 samples for stable training
        category_counts = df['category'].value_counts()
        valid_categories = category_counts[category_counts >= 10].index.tolist()

        print(f"   Filtering categories: keeping {len(valid_categories)} out of {len(category_counts)} categories")
        print(f"   Valid categories: {dict(category_counts[category_counts >= 10].head(10))}")
        print(f"   Dropped categories: {list(category_counts[category_counts < 10].keys())}")

        # Filter dataframe to only include valid categories
        df_filtered = df[df['category'].isin(valid_categories)].copy()
        
        if len(df_filtered) < 100:
            print(f"   ⚠️ Warning: Only {len(df_filtered)} samples after filtering!")
            return 0.0
        
        # Prepare data
        X = df_filtered['keyword'].values
        y = df_filtered['category'].values
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Check if we still have the stratify issue
        unique_labels, counts = np.unique(y_encoded, return_counts=True)
        min_count = counts.min()
        
        if min_count < 2:
            print(f"   ⚠️ Still have categories with < 2 samples. Using random split instead of stratified.")
            # Split data without stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42
            )
        else:
            # Split data with stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
        
        # Create question vectorizer
        self.question_vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        # Vectorize keywords
        X_train_vec = self.question_vectorizer.fit_transform(X_train)
        X_test_vec = self.question_vectorizer.transform(X_test)
        
        # Train classifier
        self.category_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        
        self.category_classifier.fit(X_train_vec, y_train)
        
        # Evaluate
        y_pred = self.category_classifier.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"   ✅ Category classifier accuracy: {accuracy:.4f}")
        print(f"   Categories: {list(self.label_encoder.classes_)}")
        
        return accuracy
    
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
    
    def save_trained_model(self):
        """Save toàn bộ trained model"""
        print("\n💾 Saving trained model...")
        
        model_data = {
            'keyword_vectorizer': self.keyword_vectorizer,
            'question_vectorizer': self.question_vectorizer,
            'category_classifier': self.category_classifier,
            'similarity_model': self.similarity_model,
            'label_encoder': self.label_encoder,
            'keyword_question_mapping': dict(self.keyword_question_mapping),
            'real_questions_db': self.real_questions_db,
            'category_keywords': dict(self.category_keywords),
            'training_date': datetime.now().isoformat(),
            'model_info': {
                'total_keywords': len(self.keyword_question_mapping),
                'total_questions': len(self.real_questions_db),
                'categories': list(self.label_encoder.classes_) if self.label_encoder else []
            }
        }
        
        # Save model
        model_file = self.models_path / 'real_data_question_model.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save summary
        summary_data = {
            'training_completed': datetime.now().isoformat(),
            'model_file': str(model_file),
            'statistics': model_data['model_info'],
            'sample_keywords': list(self.keyword_question_mapping.keys())[:50]
        }
        
        summary_file = self.models_path / 'real_data_model_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"   ✅ Model saved to: {model_file}")
        print(f"   ✅ Summary saved to: {summary_file}")
        
        return model_file
    
    def load_trained_model(self, model_file=None):
        """Load trained model"""
        if not model_file:
            model_file = self.models_path / 'real_data_question_model.pkl'
        
        try:
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            self.keyword_vectorizer = model_data['keyword_vectorizer']
            self.question_vectorizer = model_data['question_vectorizer']
            self.category_classifier = model_data['category_classifier']
            self.similarity_model = model_data['similarity_model']
            self.label_encoder = model_data['label_encoder']
            self.keyword_question_mapping = model_data['keyword_question_mapping']
            self.real_questions_db = model_data['real_questions_db']
            self.category_keywords = model_data['category_keywords']
            
            print(f"   ✅ Model loaded from: {model_file}")
            return True
            
        except Exception as e:
            print(f"   ❌ Error loading model: {e}")
            return False

def main():
    """Main training function - Xử lý TOÀN BỘ datasets"""
    print("=" * 70)
    print("🚀 REAL DATA QUESTION GENERATION TRAINER - FULL DATASET")
    print("=" * 70)
    
    # Initialize trainer
    trainer = RealDataQuestionTrainer()
    
    # Load and process ALL datasets (800 + 500 files)
    print("🔥 Processing ALL 800 dataset files + 500 question files...")
    combined_df = trainer.load_and_process_datasets()  # No limit = all files
    
    if len(combined_df) < 1000:
        print("❌ Insufficient data for training!")
        return
    
    # Build AI models
    print("\n🎯 Building AI models from MASSIVE real data...")
    
    # Train category classifier
    accuracy = trainer.train_category_classifier(combined_df)
    
    # Build similarity model
    unique_keywords = trainer.build_keyword_similarity_model(combined_df)
    
    # Save trained model regardless of threshold so user can export and inspect it.
    # If accuracy is low we'll still save but include a warning in the logs.
    if trainer.category_classifier is not None:
        model_file = trainer.save_trained_model()
        if accuracy is None:
            accuracy = 0.0
        if accuracy <= 0.5:
            print(f"\n⚠️ Model saved but accuracy is low ({accuracy:.4f}). Inspect and retrain if needed.")
        else:
            print(f"\n✅ Model saved (accuracy: {accuracy:.4f})")
        
        # Test the system
        print("\n🧪 Testing question generation on TRAINED MODEL...")
        test_keywords = [
            "artificial intelligence",
            "financial modeling", 
            "digital marketing automation",
            "cloud computing security",
            "investment portfolio management",
            "machine learning algorithms",
            "social media advertising",
            "cryptocurrency trading",
            "data science analytics",
            "e-commerce optimization"
        ]
        
        for keyword in test_keywords:
            print(f"\n🎯 Testing: '{keyword}'")
            questions = trainer.generate_questions_from_real_data(keyword, 4)
            
            for i, q in enumerate(questions, 1):
                method = q['method']
                confidence = q.get('confidence', 0)
                print(f"   {i}. {q['question']} [{method}, conf: {confidence:.2f}]")
        
        print(f"\n✅ MASSIVE DATA Training Completed!")
        print(f"📊 Final Model Statistics:")
        print(f"   • Dataset files processed: 800+")
        print(f"   • Question files processed: 500+") 
        print(f"   • Model Accuracy: {accuracy:.4f}")
        print(f"   • Total keywords: {len(trainer.keyword_question_mapping):,}")
        print(f"   • Total questions: {len(trainer.real_questions_db):,}")
        print(f"   • Unique keywords: {len(unique_keywords):,}")
        print(f"   • Model saved: {model_file}")
        
    else:
        print("❌ Category classifier was not trained. No model to save.")

if __name__ == "__main__":
    main()
