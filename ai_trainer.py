#!/usr/bin/env python3
"""
Form Agent AI Training Script
Train models cho classification và form generation
"""

import pandas as pd
import numpy as np
import os
import pickle
import json
import logging
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import joblib
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FormAgentTrainer:
    """Class để train các models cho Form Agent AI"""
    
    def __init__(self, model_dir="models"):
        """Initialize trainer"""
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Model components
        self.category_model = None
        self.form_type_model = None  
        self.complexity_model = None
        self.vectorizers = {}
        self.label_encoders = {}
        self.trained_models = {}
        
        logger.info(f"FormAgentTrainer initialized. Models will be saved to: {model_dir}")
        
    def load_dataset(self, dataset_path):
        """Load training dataset"""
        logger.info(f"Loading dataset from: {dataset_path}")
        
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset not found: {dataset_path}")
            return None
            
        try:
            df = pd.read_csv(dataset_path)
            logger.info(f"Dataset loaded successfully: {len(df):,} records")
            logger.info(f"Columns: {list(df.columns)}")
            
            # Show dataset statistics
            logger.info("\nDataset Statistics:")
            logger.info(f"Categories: {df['category'].value_counts().to_dict()}")
            logger.info(f"Form Types: {df['form_type'].value_counts().to_dict()}")
            logger.info(f"Complexity: {df['complexity'].value_counts().to_dict()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return None
    
    def preprocess_data(self, df):
        """Preprocess data for training"""
        logger.info("Preprocessing data...")
        
        # Clean text data
        df['keyword_clean'] = df['keyword'].str.lower().str.strip()
        
        # Remove any null values
        df = df.dropna(subset=['keyword_clean', 'category', 'form_type', 'complexity'])
        
        logger.info(f"Data preprocessed. Final dataset size: {len(df):,} records")
        return df
    
    def train_category_classifier(self, df):
        """Train category classification model"""
        logger.info("Training category classification model...")
        
        X = df['keyword_clean']
        y = df['category']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set size: {len(X_train):,}")
        logger.info(f"Test set size: {len(X_test):,}")
        
        # Try multiple models
        models_to_try = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
            'MultinomialNB': MultinomialNB(),
            'GradientBoosting': GradientBoostingClassifier(random_state=42)
        }
        
        best_model = None
        best_score = 0
        best_name = ""
        
        # Vectorizer
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Test models
        for name, model in models_to_try.items():
            logger.info(f"Testing {name}...")
            
            model.fit(X_train_vec, y_train)
            y_pred = model.predict(X_test_vec)
            
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            logger.info(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
            if accuracy > best_score:
                best_score = accuracy
                best_model = model
                best_name = name
        
        logger.info(f"Best model: {best_name} with accuracy: {best_score:.4f}")
        
        # Save best model
        self.category_model = best_model
        self.vectorizers['category'] = vectorizer
        
        # Detailed evaluation
        y_pred_best = best_model.predict(X_test_vec)
        logger.info(f"\nCategory Classification Report:")
        logger.info(f"\n{classification_report(y_test, y_pred_best)}")
        
        return best_score
    
    def train_form_type_classifier(self, df):
        """Train form type classification model"""
        logger.info("Training form type classification model...")
        
        X = df['keyword_clean']
        y = df['form_type']
        
        # Split data  
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Models to try
        models_to_try = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
            'MultinomialNB': MultinomialNB()
        }
        
        best_model = None
        best_score = 0
        best_name = ""
        
        # Vectorizer
        vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Test models
        for name, model in models_to_try.items():
            logger.info(f"Testing {name} for form type...")
            
            model.fit(X_train_vec, y_train)
            y_pred = model.predict(X_test_vec)
            
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            logger.info(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
            if accuracy > best_score:
                best_score = accuracy
                best_model = model
                best_name = name
        
        logger.info(f"Best form type model: {best_name} with accuracy: {best_score:.4f}")
        
        # Save best model
        self.form_type_model = best_model
        self.vectorizers['form_type'] = vectorizer
        
        # Detailed evaluation
        y_pred_best = best_model.predict(X_test_vec)
        logger.info(f"\nForm Type Classification Report:")
        logger.info(f"\n{classification_report(y_test, y_pred_best)}")
        
        return best_score
    
    def train_complexity_classifier(self, df):
        """Train complexity classification model"""
        logger.info("Training complexity classification model...")
        
        # Use both keyword and num_fields for complexity prediction
        X_text = df['keyword_clean']
        X_numeric = df[['num_fields', 'estimated_time']].fillna(0)
        y = df['complexity']
        
        # Split data
        X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
            X_text, X_numeric, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Vectorize text features
        vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
        X_text_train_vec = vectorizer.fit_transform(X_text_train)
        X_text_test_vec = vectorizer.transform(X_text_test)
        
        # Combine text and numeric features
        from scipy.sparse import hstack
        X_train_combined = hstack([X_text_train_vec, X_num_train.values])
        X_test_combined = hstack([X_text_test_vec, X_num_test.values])
        
        # Try models
        models_to_try = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42)
        }
        
        best_model = None
        best_score = 0
        best_name = ""
        
        for name, model in models_to_try.items():
            logger.info(f"Testing {name} for complexity...")
            
            model.fit(X_train_combined, y_train)
            y_pred = model.predict(X_test_combined)
            
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            logger.info(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
            if accuracy > best_score:
                best_score = accuracy
                best_model = model
                best_name = name
        
        logger.info(f"Best complexity model: {best_name} with accuracy: {best_score:.4f}")
        
        # Save best model
        self.complexity_model = best_model
        self.vectorizers['complexity'] = vectorizer
        
        # Detailed evaluation
        y_pred_best = best_model.predict(X_test_combined)
        logger.info(f"\nComplexity Classification Report:")
        logger.info(f"\n{classification_report(y_test, y_pred_best)}")
        
        return best_score
    
    def save_models(self):
        """Save all trained models"""
        logger.info(f"Saving models to {self.model_dir}...")
        
        # Save models
        model_files = {
            'category_model.pkl': self.category_model,
            'form_type_model.pkl': self.form_type_model, 
            'complexity_model.pkl': self.complexity_model
        }
        
        for filename, model in model_files.items():
            if model is not None:
                filepath = os.path.join(self.model_dir, filename)
                joblib.dump(model, filepath)
                logger.info(f"Saved: {filepath}")
        
        # Save vectorizers
        vectorizer_files = {
            'category_vectorizer.pkl': self.vectorizers.get('category'),
            'form_type_vectorizer.pkl': self.vectorizers.get('form_type'),
            'complexity_vectorizer.pkl': self.vectorizers.get('complexity')
        }
        
        for filename, vectorizer in vectorizer_files.items():
            if vectorizer is not None:
                filepath = os.path.join(self.model_dir, filename)
                joblib.dump(vectorizer, filepath)
                logger.info(f"Saved: {filepath}")
        
        # Save metadata
        metadata = {
            'trained_date': datetime.now().isoformat(),
            'models': {
                'category': type(self.category_model).__name__ if self.category_model else None,
                'form_type': type(self.form_type_model).__name__ if self.form_type_model else None,
                'complexity': type(self.complexity_model).__name__ if self.complexity_model else None
            }
        }
        
        metadata_path = os.path.join(self.model_dir, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model metadata saved: {metadata_path}")
    
    def load_models(self):
        """Load pre-trained models"""
        logger.info(f"Loading models from {self.model_dir}...")
        
        try:
            # Load models
            self.category_model = joblib.load(os.path.join(self.model_dir, 'category_model.pkl'))
            self.form_type_model = joblib.load(os.path.join(self.model_dir, 'form_type_model.pkl'))
            self.complexity_model = joblib.load(os.path.join(self.model_dir, 'complexity_model.pkl'))
            
            # Load vectorizers
            self.vectorizers['category'] = joblib.load(os.path.join(self.model_dir, 'category_vectorizer.pkl'))
            self.vectorizers['form_type'] = joblib.load(os.path.join(self.model_dir, 'form_type_vectorizer.pkl'))
            self.vectorizers['complexity'] = joblib.load(os.path.join(self.model_dir, 'complexity_vectorizer.pkl'))
            
            logger.info("Models loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def predict_category(self, keyword):
        """Predict category for keyword"""
        if self.category_model is None:
            return "it", 0.5
        
        keyword_clean = keyword.lower().strip()
        keyword_vec = self.vectorizers['category'].transform([keyword_clean])
        
        category = self.category_model.predict(keyword_vec)[0]
        confidence = np.max(self.category_model.predict_proba(keyword_vec)[0])
        
        return category, confidence
    
    def predict_form_type(self, keyword):
        """Predict form type for keyword"""
        if self.form_type_model is None:
            return "survey", 0.5
        
        keyword_clean = keyword.lower().strip()
        keyword_vec = self.vectorizers['form_type'].transform([keyword_clean])
        
        form_type = self.form_type_model.predict(keyword_vec)[0]
        confidence = np.max(self.form_type_model.predict_proba(keyword_vec)[0])
        
        return form_type, confidence
    
    def predict_complexity(self, keyword, num_fields=8, estimated_time=5):
        """Predict complexity for keyword"""
        if self.complexity_model is None:
            return "Moderate", 0.5
        
        keyword_clean = keyword.lower().strip()
        keyword_vec = self.vectorizers['complexity'].transform([keyword_clean])
        
        # Combine with numeric features
        from scipy.sparse import hstack
        numeric_features = np.array([[num_fields, estimated_time]])
        combined_features = hstack([keyword_vec, numeric_features])
        
        complexity = self.complexity_model.predict(combined_features)[0]
        confidence = np.max(self.complexity_model.predict_proba(combined_features)[0])
        
        return complexity, confidence
    
    def train_all_models(self, dataset_path):
        """Train all models using the dataset"""
        logger.info("Starting complete model training pipeline...")
        
        # Load dataset
        df = self.load_dataset(dataset_path)
        if df is None:
            logger.error("Failed to load dataset. Training aborted.")
            return False
        
        # Preprocess data
        df = self.preprocess_data(df)
        
        # Train models
        results = {}
        
        logger.info("\n" + "="*50)
        logger.info("TRAINING CATEGORY CLASSIFIER")
        logger.info("="*50)
        results['category_accuracy'] = self.train_category_classifier(df)
        
        logger.info("\n" + "="*50)
        logger.info("TRAINING FORM TYPE CLASSIFIER") 
        logger.info("="*50)
        results['form_type_accuracy'] = self.train_form_type_classifier(df)
        
        logger.info("\n" + "="*50)
        logger.info("TRAINING COMPLEXITY CLASSIFIER")
        logger.info("="*50) 
        results['complexity_accuracy'] = self.train_complexity_classifier(df)
        
        # Save models
        logger.info("\n" + "="*50)
        logger.info("SAVING MODELS")
        logger.info("="*50)
        self.save_models()
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info("TRAINING COMPLETE - SUMMARY")
        logger.info("="*50)
        for model_type, accuracy in results.items():
            logger.info(f"{model_type}: {accuracy:.4f}")
        
        return True

def main():
    """Main training function"""
    print(" Form Agent AI - Model Training")
    
    # Initialize trainer
    trainer = FormAgentTrainer()
    
    # Look for dataset files
    dataset_dir = "datasets"
    sample_files = []
    
    if os.path.exists(dataset_dir):
        files = os.listdir(dataset_dir)
        sample_files = [f for f in files if f.startswith("sample_") and f.endswith(".csv")]
        batch_files = [f for f in files if f.startswith("batch_") and f.endswith(".csv")]
        
        print(f" Found {len(sample_files)} sample files and {len(batch_files)} batch files")
    
    # Choose training dataset
    if sample_files:
        # Use largest sample file for training
        sample_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]) if x.split("_")[1].split(".")[0].isdigit() else 0)
        largest_sample = sample_files[-1]
        dataset_path = os.path.join(dataset_dir, largest_sample)
        print(f"Using dataset: {dataset_path}")
    else:
        print(" No dataset files found. Please run dataset generator first.")
        return
    
    # Train models
    try:
        success = trainer.train_all_models(dataset_path)
        
        if success:
            print("\n Training completed successfully!")
            
            # Test predictions
            print("\n Testing predictions:")
            test_keywords = [
                "cloud security management",
                "investment portfolio analysis", 
                "digital marketing campaign"
            ]
            
            for keyword in test_keywords:
                category, cat_conf = trainer.predict_category(keyword)
                form_type, type_conf = trainer.predict_form_type(keyword)
                complexity, comp_conf = trainer.predict_complexity(keyword)
                
                print(f"\nKeyword: '{keyword}'")
                print(f"  Category: {category} (confidence: {cat_conf:.3f})")
                print(f"  Form Type: {form_type} (confidence: {type_conf:.3f})")
                print(f"  Complexity: {complexity} (confidence: {comp_conf:.3f})")
        else:
            print(" Training failed!")
            
    except Exception as e:
        print(f" Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
