#!/usr/bin/env python3
"""
Form Agent AI Model
Mô hình AI để phân loại keyword và tạo form structure
"""

import pandas as pd
import numpy as np
import pickle
import json
import re
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import joblib
import logging
from typing import Dict, List, Tuple, Any

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FormAgentAI:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.category_model = None
        self.form_type_model = None
        self.complexity_model = None
        self.vectorizer = None
        self.label_encoders = {}
        
        # Form generation templates
        self.setup_form_templates()
        
    def setup_form_templates(self):
        """Thiết lập templates để tạo form"""
        
        self.field_templates = {
            "it": {
                "registration": {
                    "fields": [
                        {"name": "full_name", "type": "text", "label": "Họ và tên", "required": True},
                        {"name": "email", "type": "email", "label": "Email", "required": True},
                        {"name": "phone", "type": "tel", "label": "Số điện thoại", "required": True},
                        {"name": "company", "type": "text", "label": "Công ty", "required": False},
                        {"name": "position", "type": "select", "label": "Vị trí", "required": True,
                         "options": ["Developer", "Senior Developer", "Tech Lead", "CTO", "DevOps", "QA Engineer"]},
                        {"name": "experience_years", "type": "number", "label": "Năm kinh nghiệm", "required": True,
                         "validation": {"min": 0, "max": 50}},
                        {"name": "skills", "type": "checkbox", "label": "Kỹ năng", "required": True,
                         "options": ["Python", "JavaScript", "Java", "React", "Node.js", "Docker", "AWS", "Machine Learning"]},
                        {"name": "github_profile", "type": "url", "label": "GitHub Profile", "required": False},
                    ]
                },
                "survey": {
                    "fields": [
                        {"name": "satisfaction_rating", "type": "range", "label": "Mức độ hài lòng", "required": True,
                         "validation": {"min": 1, "max": 10}},
                        {"name": "usage_frequency", "type": "select", "label": "Tần suất sử dụng", "required": True,
                         "options": ["Daily", "Weekly", "Monthly", "Occasionally", "First time"]},
                        {"name": "preferred_tools", "type": "checkbox", "label": "Công cụ ưa thích", "required": False,
                         "options": ["VS Code", "IntelliJ", "Sublime", "Vim", "Eclipse"]},
                        {"name": "challenges", "type": "textarea", "label": "Thách thức gặp phải", "required": False,
                         "validation": {"maxLength": 500}},
                        {"name": "suggestions", "type": "textarea", "label": "Đề xuất cải thiện", "required": False,
                         "validation": {"maxLength": 500}}
                    ]
                }
            },
            "economics": {
                "financial_assessment": {
                    "fields": [
                        {"name": "monthly_income", "type": "number", "label": "Thu nhập hàng tháng", "required": True,
                         "validation": {"min": 0, "max": 1000000000}},
                        {"name": "monthly_expenses", "type": "number", "label": "Chi phí hàng tháng", "required": True,
                         "validation": {"min": 0, "max": 1000000000}},
                        {"name": "current_savings", "type": "number", "label": "Tiết kiệm hiện tại", "required": True,
                         "validation": {"min": 0}},
                        {"name": "investment_experience", "type": "select", "label": "Kinh nghiệm đầu tư", "required": True,
                         "options": ["No experience", "Beginner", "Intermediate", "Advanced", "Professional"]},
                        {"name": "risk_tolerance", "type": "select", "label": "Khả năng chấp nhận rủi ro", "required": True,
                         "options": ["Very Low", "Low", "Medium", "High", "Very High"]},
                        {"name": "investment_goals", "type": "checkbox", "label": "Mục tiêu đầu tư", "required": True,
                         "options": ["Retirement", "House purchase", "Education", "Emergency fund", "Wealth building"]},
                        {"name": "time_horizon", "type": "select", "label": "Thời gian đầu tư", "required": True,
                         "options": ["< 1 year", "1-3 years", "3-5 years", "5-10 years", "> 10 years"]}
                    ]
                },
                "market_survey": {
                    "fields": [
                        {"name": "industry", "type": "select", "label": "Ngành nghề", "required": True,
                         "options": ["Technology", "Finance", "Healthcare", "Education", "Manufacturing", "Retail", "Services"]},
                        {"name": "company_size", "type": "select", "label": "Quy mô công ty", "required": True,
                         "options": ["1-10", "11-50", "51-200", "201-1000", "> 1000"]},
                        {"name": "market_position", "type": "select", "label": "Vị trí thị trường", "required": True,
                         "options": ["Market leader", "Strong competitor", "Growing player", "Niche player", "New entrant"]},
                        {"name": "main_challenges", "type": "checkbox", "label": "Thách thức chính", "required": True,
                         "options": ["Competition", "Pricing", "Technology", "Regulations", "Talent", "Funding"]},
                        {"name": "growth_strategy", "type": "textarea", "label": "Chiến lược tăng trưởng", "required": False,
                         "validation": {"maxLength": 1000}}
                    ]
                }
            },
            "marketing": {
                "campaign_brief": {
                    "fields": [
                        {"name": "campaign_name", "type": "text", "label": "Tên chiến dịch", "required": True,
                         "validation": {"minLength": 3, "maxLength": 100}},
                        {"name": "campaign_objective", "type": "select", "label": "Mục tiêu chiến dịch", "required": True,
                         "options": ["Brand awareness", "Lead generation", "Sales conversion", "Customer retention", "Market expansion"]},
                        {"name": "target_audience", "type": "text", "label": "Đối tượng mục tiêu", "required": True},
                        {"name": "budget_range", "type": "select", "label": "Ngân sách", "required": True,
                         "options": ["< $1,000", "$1,000 - $5,000", "$5,000 - $10,000", "$10,000 - $50,000", "> $50,000"]},
                        {"name": "channels", "type": "checkbox", "label": "Kênh marketing", "required": True,
                         "options": ["Social Media", "Email", "PPC", "SEO", "Content Marketing", "Influencer", "Traditional Media"]},
                        {"name": "timeline", "type": "select", "label": "Thời gian thực hiện", "required": True,
                         "options": ["< 1 month", "1-3 months", "3-6 months", "6-12 months", "> 1 year"]},
                        {"name": "success_metrics", "type": "checkbox", "label": "Metrics đo lường", "required": True,
                         "options": ["CTR", "Conversion rate", "ROI", "Brand recall", "Engagement rate", "Lead quality"]}
                    ]
                },
                "customer_survey": {
                    "fields": [
                        {"name": "age_group", "type": "select", "label": "Độ tuổi", "required": True,
                         "options": ["18-25", "26-35", "36-45", "46-55", "56-65", "> 65"]},
                        {"name": "gender", "type": "select", "label": "Giới tính", "required": False,
                         "options": ["Male", "Female", "Other", "Prefer not to say"]},
                        {"name": "income_level", "type": "select", "label": "Thu nhập", "required": False,
                         "options": ["< $25k", "$25k-$50k", "$50k-$75k", "$75k-$100k", "> $100k"]},
                        {"name": "purchase_frequency", "type": "select", "label": "Tần suất mua sắm", "required": True,
                         "options": ["Daily", "Weekly", "Monthly", "Quarterly", "Annually", "Rarely"]},
                        {"name": "preferred_channels", "type": "checkbox", "label": "Kênh mua sắm ưa thích", "required": True,
                         "options": ["Online", "In-store", "Mobile app", "Social media", "Phone", "Email"]},
                        {"name": "brand_loyalty", "type": "range", "label": "Độ trung thành thương hiệu", "required": True,
                         "validation": {"min": 1, "max": 10}},
                        {"name": "feedback", "type": "textarea", "label": "Phản hồi", "required": False,
                         "validation": {"maxLength": 500}}
                    ]
                }
            }
        }
        
        # Complexity scoring weights
        self.complexity_weights = {
            "field_count": 0.3,
            "validation_rules": 0.2,
            "field_types": 0.2,
            "options_count": 0.15,
            "required_fields": 0.15
        }

    def train_models(self, dataset_path: str):
        """Huấn luyện các models"""
        logger.info(f"Loading dataset from {dataset_path}")
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        logger.info(f"Dataset loaded: {len(df)} records")
        
        # Prepare features
        X = df['keyword'].fillna('')
        
        # Train category classification model
        self._train_category_model(X, df['category'])
        
        # Train form type model for each category
        self._train_form_type_models(df)
        
        # Train complexity model
        self._train_complexity_model(X, df['complexity'])
        
        # Save all models
        self.save_models()
        
        logger.info("Training completed successfully!")
        
    def _train_category_model(self, X, y):
        """Huấn luyện model phân loại category"""
        logger.info("Training category classification model...")
        
        # Create pipeline
        self.category_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 3), stop_words='english')),
            ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
        ])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train
        self.category_pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.category_pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Category model accuracy: {accuracy:.4f}")
        logger.info("Category classification report:")
        logger.info("\n" + classification_report(y_test, y_pred))
        
    def _train_form_type_models(self, df):
        """Huấn luyện models phân loại form type cho từng category"""
        logger.info("Training form type models...")
        
        self.form_type_models = {}
        
        for category in df['category'].unique():
            category_df = df[df['category'] == category]
            
            if len(category_df) < 10:  # Skip if not enough data
                continue
                
            X_cat = category_df['keyword']
            y_cat = category_df['form_type']
            
            # Skip if only one form type
            if len(y_cat.unique()) < 2:
                continue
            
            # Create pipeline for this category
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
                ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
            ])
            
            try:
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_cat, y_cat, test_size=0.2, random_state=42, stratify=y_cat
                )
                
                # Train
                pipeline.fit(X_train, y_train)
                
                # Evaluate
                y_pred = pipeline.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                self.form_type_models[category] = pipeline
                
                logger.info(f"Form type model for {category}: accuracy = {accuracy:.4f}")
                
            except Exception as e:
                logger.warning(f"Could not train form type model for {category}: {str(e)}")
                
    def _train_complexity_model(self, X, y):
        """Huấn luyện model dự đoán complexity"""
        logger.info("Training complexity prediction model...")
        
        # Create pipeline
        self.complexity_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train
        self.complexity_pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.complexity_pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Complexity model accuracy: {accuracy:.4f}")

    def predict_category(self, keyword: str) -> Tuple[str, float]:
        """Dự đoán category từ keyword"""
        if not self.category_pipeline:
            return "it", 0.5  # default
            
        proba = self.category_pipeline.predict_proba([keyword])[0]
        category = self.category_pipeline.predict([keyword])[0]
        confidence = max(proba)
        
        return category, confidence
    
    def predict_form_type(self, keyword: str, category: str) -> Tuple[str, float]:
        """Dự đoán form type từ keyword và category"""
        if category not in self.form_type_models:
            # Return default form type for each category
            defaults = {
                "it": "registration",
                "economics": "financial_assessment",
                "marketing": "campaign_brief"
            }
            return defaults.get(category, "registration"), 0.5
            
        model = self.form_type_models[category]
        proba = model.predict_proba([keyword])[0]
        form_type = model.predict([keyword])[0]
        confidence = max(proba)
        
        return form_type, confidence
    
    def predict_complexity(self, keyword: str) -> Tuple[str, float]:
        """Dự đoán complexity từ keyword"""
        if not self.complexity_pipeline:
            return "Moderate", 0.5
            
        proba = self.complexity_pipeline.predict_proba([keyword])[0]
        complexity = self.complexity_pipeline.predict([keyword])[0]
        confidence = max(proba)
        
        return complexity, confidence

    def generate_form(self, keyword: str) -> Dict[str, Any]:
        """Tạo form structure từ keyword"""
        # Predict category, form type, and complexity
        category, cat_confidence = self.predict_category(keyword)
        form_type, type_confidence = self.predict_form_type(keyword, category)
        complexity, comp_confidence = self.predict_complexity(keyword)
        
        # Get base template
        base_fields = self._get_base_fields(category, form_type)
        
        # Customize fields based on keyword
        customized_fields = self._customize_fields(base_fields, keyword, category, complexity)
        
        # Generate form metadata
        form_structure = {
            "form_id": self._generate_form_id(),
            "title": f"{keyword.title()} Form",
            "description": f"Form generated for {keyword} in {category} domain",
            "category": category,
            "form_type": form_type,
            "complexity": complexity,
            "keyword": keyword,
            "fields": customized_fields,
            "metadata": {
                "category_confidence": cat_confidence,
                "type_confidence": type_confidence,
                "complexity_confidence": comp_confidence,
                "estimated_completion_time": self._estimate_completion_time(customized_fields, complexity),
                "created_at": datetime.now().isoformat(),
                "field_count": len(customized_fields),
                "required_fields": len([f for f in customized_fields if f.get("required", False)])
            },
            "validation_rules": self._generate_validation_rules(customized_fields),
            "styling": self._generate_styling(category, complexity)
        }
        
        return form_structure
    
    def _get_base_fields(self, category: str, form_type: str) -> List[Dict]:
        """Lấy base fields từ template"""
        try:
            return self.field_templates[category][form_type]["fields"].copy()
        except KeyError:
            # Return default fields if template not found
            return [
                {"name": "name", "type": "text", "label": "Họ và tên", "required": True},
                {"name": "email", "type": "email", "label": "Email", "required": True},
                {"name": "message", "type": "textarea", "label": "Tin nhắn", "required": False}
            ]
    
    def _customize_fields(self, base_fields: List[Dict], keyword: str, category: str, complexity: str) -> List[Dict]:
        """Tùy chỉnh fields dựa trên keyword và context"""
        customized = base_fields.copy()
        
        # Add keyword-specific fields
        keyword_lower = keyword.lower()
        
        # Add specific fields based on keyword content
        if "security" in keyword_lower and category == "it":
            customized.append({
                "name": "security_concerns",
                "type": "checkbox",
                "label": "Vấn đề bảo mật quan tâm",
                "required": False,
                "options": ["Data encryption", "Access control", "Vulnerability assessment", "Compliance", "Monitoring"]
            })
        
        if "investment" in keyword_lower and category == "economics":
            customized.append({
                "name": "investment_amount",
                "type": "number",
                "label": "Số tiền đầu tư",
                "required": True,
                "validation": {"min": 1000, "max": 10000000}
            })
        
        if "campaign" in keyword_lower and category == "marketing":
            customized.append({
                "name": "campaign_duration",
                "type": "select",
                "label": "Thời gian chiến dịch",
                "required": True,
                "options": ["1 week", "2 weeks", "1 month", "3 months", "6 months", "1 year"]
            })
        
        # Adjust complexity
        if complexity == "Complex":
            # Add more detailed fields for complex forms
            customized.append({
                "name": "additional_requirements",
                "type": "textarea",
                "label": "Yêu cầu bổ sung",
                "required": False,
                "validation": {"maxLength": 1000}
            })
        elif complexity == "Simple":
            # Remove some optional fields for simple forms
            customized = [f for f in customized if f.get("required", True) or len(customized) <= 5]
        
        return customized
    
    def _generate_form_id(self) -> str:
        """Tạo unique form ID"""
        return f"form_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
    
    def _estimate_completion_time(self, fields: List[Dict], complexity: str) -> int:
        """Ước tính thời gian hoàn thành form (phút)"""
        base_time = len(fields) * 0.5  # 30 seconds per field
        
        complexity_multiplier = {
            "Simple": 0.8,
            "Moderate": 1.0,
            "Complex": 1.5
        }
        
        estimated = base_time * complexity_multiplier.get(complexity, 1.0)
        return max(1, int(estimated))  # At least 1 minute
    
    def _generate_validation_rules(self, fields: List[Dict]) -> Dict[str, Any]:
        """Tạo validation rules tổng thể cho form"""
        rules = {
            "required_fields": [f["name"] for f in fields if f.get("required", False)],
            "field_validations": {},
            "form_level_rules": []
        }
        
        for field in fields:
            if "validation" in field:
                rules["field_validations"][field["name"]] = field["validation"]
        
        return rules
    
    def _generate_styling(self, category: str, complexity: str) -> Dict[str, Any]:
        """Tạo styling configuration cho form"""
        color_schemes = {
            "it": {"primary": "#007ACC", "secondary": "#F0F8FF", "accent": "#32CD32"},
            "economics": {"primary": "#2E8B57", "secondary": "#F0FFF0", "accent": "#FFD700"},
            "marketing": {"primary": "#FF6347", "secondary": "#FFF5EE", "accent": "#FF69B4"}
        }
        
        layout_configs = {
            "Simple": {"columns": 1, "spacing": "normal", "size": "medium"},
            "Moderate": {"columns": 2, "spacing": "comfortable", "size": "medium"},
            "Complex": {"columns": 2, "spacing": "comfortable", "size": "large"}
        }
        
        return {
            "colors": color_schemes.get(category, color_schemes["it"]),
            "layout": layout_configs.get(complexity, layout_configs["Moderate"]),
            "theme": "professional",
            "animations": complexity != "Simple"
        }

    def save_models(self):
        """Lưu tất cả models"""
        logger.info("Saving models...")
        
        # Save category model
        if self.category_pipeline:
            joblib.dump(self.category_pipeline, os.path.join(self.model_dir, "category_model.pkl"))
        
        # Save form type models
        if hasattr(self, 'form_type_models'):
            joblib.dump(self.form_type_models, os.path.join(self.model_dir, "form_type_models.pkl"))
        
        # Save complexity model
        if self.complexity_pipeline:
            joblib.dump(self.complexity_pipeline, os.path.join(self.model_dir, "complexity_model.pkl"))
        
        # Save field templates
        with open(os.path.join(self.model_dir, "field_templates.json"), "w") as f:
            json.dump(self.field_templates, f, indent=2)
        
        logger.info(f"Models saved to {self.model_dir}")
    
    def load_models(self):
        """Load đã lưu models"""
        logger.info("Loading models...")
        
        try:
            # Load category model
            category_model_path = os.path.join(self.model_dir, "category_model.pkl")
            if os.path.exists(category_model_path):
                self.category_pipeline = joblib.load(category_model_path)
            
            # Load form type models
            form_type_models_path = os.path.join(self.model_dir, "form_type_models.pkl")
            if os.path.exists(form_type_models_path):
                self.form_type_models = joblib.load(form_type_models_path)
            
            # Load complexity model
            complexity_model_path = os.path.join(self.model_dir, "complexity_model.pkl")
            if os.path.exists(complexity_model_path):
                self.complexity_pipeline = joblib.load(complexity_model_path)
            
            # Load field templates
            templates_path = os.path.join(self.model_dir, "field_templates.json")
            if os.path.exists(templates_path):
                with open(templates_path, "r") as f:
                    self.field_templates = json.load(f)
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")

def main():
    """Demo function"""
    print("🤖 Form Agent AI Model Training")
    
    # Initialize model
    ai_model = FormAgentAI()
    
    # Check if dataset exists
    dataset_path = "datasets/form_agent_dataset_sample_10000.csv"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        print("Please run dataset_generator.py first to create the dataset")
        return
    
    # Train models
    print("🎯 Training AI models...")
    ai_model.train_models(dataset_path)
    
    # Test form generation
    print("\n🧪 Testing form generation...")
    test_keywords = [
        "cloud security assessment",
        "investment portfolio analysis", 
        "digital marketing campaign planning"
    ]
    
    for keyword in test_keywords:
        print(f"\n📝 Generating form for: '{keyword}'")
        form = ai_model.generate_form(keyword)
        
        print(f"   Category: {form['category']} (confidence: {form['metadata']['category_confidence']:.2f})")
        print(f"   Form Type: {form['form_type']}")
        print(f"   Complexity: {form['complexity']}")
        print(f"   Fields: {form['metadata']['field_count']}")
        print(f"   Estimated time: {form['metadata']['estimated_completion_time']} minutes")
    
    print("\n✅ Form Agent AI setup completed!")

if __name__ == "__main__":
    main()
