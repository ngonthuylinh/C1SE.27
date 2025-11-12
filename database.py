#!/usr/bin/env python3
"""
Database Models cho Form Agent AI
Sử dụng SQLAlchemy ORM để quản lý database
"""

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, Float, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime, timezone
import uuid
import json
from typing import Dict, List, Any, Optional
import os
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///form_agent.db")

# Create database engine
engine = create_engine(
    DATABASE_URL,
    echo=os.getenv("DB_ECHO", "False").lower() == "true"
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for all models
Base = declarative_base()

class TimestampMixin:
    """Mixin for created_at and updated_at timestamps"""
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

class User(Base, TimestampMixin):
    """User model"""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, index=True, nullable=False)
    full_name = Column(String(255))
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    preferences = Column(JSON)  # User preferences as JSON
    
    # Relationships
    generated_forms = relationship("GeneratedForm", back_populates="creator")
    form_submissions = relationship("FormSubmission", back_populates="user")
    
    def __repr__(self):
        return f"<User(id={self.id}, email={self.email})>"

class Category(Base, TimestampMixin):
    """Category model for form categorization"""
    __tablename__ = "categories"
    
    id = Column(String, primary_key=True)  # 'it', 'economics', 'marketing'
    name = Column(String(100), nullable=False)
    description = Column(Text)
    is_active = Column(Boolean, default=True)
    color_scheme = Column(JSON)  # Color configuration for UI
    
    # Relationships
    generated_forms = relationship("GeneratedForm", back_populates="category_obj")
    keywords = relationship("Keyword", back_populates="category_obj")
    
    def __repr__(self):
        return f"<Category(id={self.id}, name={self.name})>"

class Keyword(Base, TimestampMixin):
    """Keyword model to track and analyze keywords"""
    __tablename__ = "keywords"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    keyword_text = Column(String(500), nullable=False, index=True)
    category_id = Column(String, ForeignKey("categories.id"))
    usage_count = Column(Integer, default=1)
    confidence_score = Column(Float)  # AI prediction confidence
    
    # Analysis results
    complexity_score = Column(Float)
    predicted_form_type = Column(String(100))
    analysis_metadata = Column(JSON)  # Additional analysis data
    
    # Relationships
    category_obj = relationship("Category", back_populates="keywords")
    generated_forms = relationship("GeneratedForm", back_populates="keyword_obj")
    
    def __repr__(self):
        return f"<Keyword(id={self.id}, text={self.keyword_text})>"

class FormTemplate(Base, TimestampMixin):
    """Form template model for reusable form structures"""
    __tablename__ = "form_templates"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(200), nullable=False)
    description = Column(Text)
    category_id = Column(String, ForeignKey("categories.id"))
    form_type = Column(String(100))  # 'registration', 'survey', etc.
    complexity = Column(String(50))  # 'Simple', 'Moderate', 'Complex'
    
    # Template structure
    fields_schema = Column(JSON, nullable=False)  # Field definitions
    validation_rules = Column(JSON)
    styling_config = Column(JSON)
    behavior_config = Column(JSON)
    
    # Usage statistics
    usage_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    generated_forms = relationship("GeneratedForm", back_populates="template")
    
    def __repr__(self):
        return f"<FormTemplate(id={self.id}, name={self.name})>"

class GeneratedForm(Base, TimestampMixin):
    """Generated form model"""
    __tablename__ = "generated_forms"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String(300), nullable=False)
    description = Column(Text)
    
    # Form classification
    keyword_id = Column(String, ForeignKey("keywords.id"))
    category_id = Column(String, ForeignKey("categories.id"))
    form_type = Column(String(100))
    complexity = Column(String(50))
    
    # Form structure
    fields_data = Column(JSON, nullable=False)  # Complete field definitions
    sections_data = Column(JSON)  # Form sections
    validation_rules = Column(JSON)
    styling_config = Column(JSON)
    behavior_config = Column(JSON)
    
    # Generation metadata
    ai_confidence_scores = Column(JSON)  # AI prediction confidence scores
    generation_parameters = Column(JSON)  # Parameters used for generation
    template_id = Column(String, ForeignKey("form_templates.id"), nullable=True)
    
    # Usage and status
    view_count = Column(Integer, default=0)
    submission_count = Column(Integer, default=0)
    is_published = Column(Boolean, default=True)
    is_archived = Column(Boolean, default=False)
    
    # User and timestamps
    creator_id = Column(String, ForeignKey("users.id"))
    
    # Relationships
    creator = relationship("User", back_populates="generated_forms")
    keyword_obj = relationship("Keyword", back_populates="generated_forms")
    category_obj = relationship("Category", back_populates="generated_forms")
    template = relationship("FormTemplate", back_populates="generated_forms")
    submissions = relationship("FormSubmission", back_populates="form")
    analytics = relationship("FormAnalytics", back_populates="form")
    
    def __repr__(self):
        return f"<GeneratedForm(id={self.id}, title={self.title})>"

class FormSubmission(Base, TimestampMixin):
    """Form submission model"""
    __tablename__ = "form_submissions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    form_id = Column(String, ForeignKey("generated_forms.id"), nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)  # Anonymous submissions allowed
    
    # Submission data
    form_data = Column(JSON, nullable=False)  # User-provided form data
    validation_status = Column(String(20), default="pending")  # 'valid', 'invalid', 'pending'
    validation_errors = Column(JSON)  # Validation error details
    
    # Metadata
    submission_ip = Column(String(45))  # IPv4/IPv6 address
    user_agent = Column(Text)
    referrer_url = Column(Text)
    completion_time = Column(Integer)  # Time taken to complete form (seconds)
    
    # Processing status
    is_processed = Column(Boolean, default=False)
    processed_at = Column(DateTime)
    processing_notes = Column(Text)
    
    # Relationships
    form = relationship("GeneratedForm", back_populates="submissions")
    user = relationship("User", back_populates="form_submissions")
    
    def __repr__(self):
        return f"<FormSubmission(id={self.id}, form_id={self.form_id})>"

class FormAnalytics(Base, TimestampMixin):
    """Form analytics and metrics"""
    __tablename__ = "form_analytics"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    form_id = Column(String, ForeignKey("generated_forms.id"), nullable=False)
    
    # Date for daily analytics
    analytics_date = Column(DateTime, nullable=False)
    
    # Metrics
    views = Column(Integer, default=0)
    submissions = Column(Integer, default=0)
    completion_rate = Column(Float, default=0.0)  # Percentage
    avg_completion_time = Column(Float, default=0.0)  # Average time in seconds
    bounce_rate = Column(Float, default=0.0)  # Percentage who left without submitting
    
    # Field-level analytics
    field_analytics = Column(JSON)  # Analytics for individual fields
    
    # Relationships
    form = relationship("GeneratedForm", back_populates="analytics")
    
    def __repr__(self):
        return f"<FormAnalytics(id={self.id}, form_id={self.form_id}, date={self.analytics_date})>"

class AIModelMetrics(Base, TimestampMixin):
    """AI model performance metrics"""
    __tablename__ = "ai_model_metrics"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    model_name = Column(String(100), nullable=False)  # 'category_classifier', 'complexity_predictor', etc.
    model_version = Column(String(50))
    
    # Performance metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    
    # Training metadata
    training_date = Column(DateTime)
    training_dataset_size = Column(Integer)
    training_parameters = Column(JSON)
    
    # Validation data
    validation_results = Column(JSON)
    confusion_matrix = Column(JSON)
    
    def __repr__(self):
        return f"<AIModelMetrics(id={self.id}, model={self.model_name})>"

class SystemLog(Base, TimestampMixin):
    """System logs for monitoring and debugging"""
    __tablename__ = "system_logs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    log_level = Column(String(20), nullable=False)  # 'INFO', 'WARNING', 'ERROR', etc.
    component = Column(String(100))  # 'api', 'ai_model', 'form_engine', etc.
    action = Column(String(200))  # 'form_generated', 'model_trained', etc.
    
    # Log details
    message = Column(Text, nullable=False)
    details = Column(JSON)  # Additional structured data
    
    # Context
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    request_id = Column(String(100))  # For tracing requests
    session_id = Column(String(100))
    
    def __repr__(self):
        return f"<SystemLog(id={self.id}, level={self.log_level}, action={self.action})>"

# Database utility functions

def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")

def get_db_session():
    """Get database session"""
    db = SessionLocal()
    try:
        return db
    finally:
        pass  # Don't close here, let caller handle it

@contextmanager
def get_db():
    """Database session context manager"""
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

def init_default_data():
    """Initialize default data"""
    with get_db() as db:
        # Create default categories
        categories_data = [
            {
                "id": "it",
                "name": "Công nghệ thông tin",
                "description": "Lĩnh vực công nghệ thông tin, phần mềm, hệ thống",
                "color_scheme": {
                    "primary": "#0066cc",
                    "secondary": "#f0f8ff",
                    "accent": "#00cc66"
                }
            },
            {
                "id": "economics",
                "name": "Kinh tế - Tài chính",
                "description": "Lĩnh vực tài chính, đầu tư, kinh doanh",
                "color_scheme": {
                    "primary": "#2e8b57",
                    "secondary": "#f0fff0",
                    "accent": "#ffd700"
                }
            },
            {
                "id": "marketing",
                "name": "Marketing",
                "description": "Lĩnh vực marketing, quảng cáo, bán hàng",
                "color_scheme": {
                    "primary": "#ff6347",
                    "secondary": "#fff5ee",
                    "accent": "#ff69b4"
                }
            }
        ]
        
        for cat_data in categories_data:
            existing = db.query(Category).filter(Category.id == cat_data["id"]).first()
            if not existing:
                category = Category(**cat_data)
                db.add(category)
        
        db.commit()
        logger.info("Default categories created")

# Repository classes for data access

class UserRepository:
    """Repository for User operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_user(self, email: str, full_name: str = None, is_admin: bool = False) -> User:
        """Create new user"""
        user = User(
            email=email,
            full_name=full_name,
            is_admin=is_admin
        )
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        return self.db.query(User).filter(User.email == email).first()
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.db.query(User).filter(User.id == user_id).first()

class FormRepository:
    """Repository for Form operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_form(self, form_data: Dict[str, Any], creator_id: str = None) -> GeneratedForm:
        """Create new generated form"""
        # Handle keyword
        keyword_obj = self._get_or_create_keyword(
            form_data.get("keyword", ""),
            form_data.get("category", "it")
        )
        
        form = GeneratedForm(
            title=form_data.get("title", ""),
            description=form_data.get("description", ""),
            keyword_id=keyword_obj.id,
            category_id=form_data.get("category", "it"),
            form_type=form_data.get("form_type", "general"),
            complexity=form_data.get("complexity", "Moderate"),
            fields_data=form_data.get("fields", []),
            sections_data=form_data.get("sections", []),
            validation_rules=form_data.get("validation_rules", {}),
            styling_config=form_data.get("styling", {}),
            behavior_config=form_data.get("behavior", {}),
            ai_confidence_scores=form_data.get("metadata", {}).get("ai_confidence", {}),
            generation_parameters=form_data.get("metadata", {}),
            creator_id=creator_id
        )
        
        self.db.add(form)
        self.db.commit()
        self.db.refresh(form)
        return form
    
    def get_form_by_id(self, form_id: str) -> Optional[GeneratedForm]:
        """Get form by ID"""
        return self.db.query(GeneratedForm).filter(GeneratedForm.id == form_id).first()
    
    def get_user_forms(self, user_id: str, limit: int = 50) -> List[GeneratedForm]:
        """Get forms created by user"""
        return (self.db.query(GeneratedForm)
                .filter(GeneratedForm.creator_id == user_id)
                .order_by(GeneratedForm.created_at.desc())
                .limit(limit)
                .all())
    
    def increment_view_count(self, form_id: str):
        """Increment form view count"""
        form = self.get_form_by_id(form_id)
        if form:
            form.view_count = (form.view_count or 0) + 1
            self.db.commit()
    
    def _get_or_create_keyword(self, keyword_text: str, category_id: str) -> Keyword:
        """Get existing keyword or create new one"""
        keyword = (self.db.query(Keyword)
                  .filter(Keyword.keyword_text == keyword_text)
                  .filter(Keyword.category_id == category_id)
                  .first())
        
        if not keyword:
            keyword = Keyword(
                keyword_text=keyword_text,
                category_id=category_id
            )
            self.db.add(keyword)
            self.db.commit()
            self.db.refresh(keyword)
        else:
            # Increment usage count
            keyword.usage_count = (keyword.usage_count or 0) + 1
            self.db.commit()
        
        return keyword

class SubmissionRepository:
    """Repository for Form Submission operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_submission(self, submission_data: Dict[str, Any]) -> FormSubmission:
        """Create new form submission"""
        submission = FormSubmission(
            form_id=submission_data["form_id"],
            user_id=submission_data.get("user_id"),
            form_data=submission_data["form_data"],
            validation_status=submission_data.get("validation_status", "pending"),
            validation_errors=submission_data.get("validation_errors"),
            submission_ip=submission_data.get("ip_address"),
            user_agent=submission_data.get("user_agent"),
            referrer_url=submission_data.get("referrer"),
            completion_time=submission_data.get("completion_time")
        )
        
        self.db.add(submission)
        
        # Update form submission count
        form = self.db.query(GeneratedForm).filter(GeneratedForm.id == submission_data["form_id"]).first()
        if form:
            form.submission_count = (form.submission_count or 0) + 1
        
        self.db.commit()
        self.db.refresh(submission)
        return submission
    
    def get_submission_by_id(self, submission_id: str) -> Optional[FormSubmission]:
        """Get submission by ID"""
        return self.db.query(FormSubmission).filter(FormSubmission.id == submission_id).first()
    
    def get_form_submissions(self, form_id: str, limit: int = 100) -> List[FormSubmission]:
        """Get all submissions for a form"""
        return (self.db.query(FormSubmission)
                .filter(FormSubmission.form_id == form_id)
                .order_by(FormSubmission.created_at.desc())
                .limit(limit)
                .all())

class AnalyticsRepository:
    """Repository for Analytics operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_popular_keywords(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most popular keywords"""
        keywords = (self.db.query(Keyword)
                   .order_by(Keyword.usage_count.desc())
                   .limit(limit)
                   .all())
        
        return [
            {
                "keyword": k.keyword_text,
                "category": k.category_id,
                "usage_count": k.usage_count,
                "confidence_score": k.confidence_score
            }
            for k in keywords
        ]
    
    def get_category_distribution(self) -> Dict[str, int]:
        """Get form distribution by category"""
        from sqlalchemy import func
        
        result = (self.db.query(GeneratedForm.category_id, func.count(GeneratedForm.id))
                 .group_by(GeneratedForm.category_id)
                 .all())
        
        return {category: count for category, count in result}
    
    def get_form_stats(self) -> Dict[str, Any]:
        """Get overall form statistics"""
        total_forms = self.db.query(GeneratedForm).count()
        total_submissions = self.db.query(FormSubmission).count()
        total_users = self.db.query(User).count()
        
        return {
            "total_forms": total_forms,
            "total_submissions": total_submissions,
            "total_users": total_users,
            "avg_submissions_per_form": total_submissions / max(total_forms, 1)
        }

def main():
    """Initialize database"""
    print("  Initializing Form Agent AI Database")
    
    # Create tables
    create_tables()
    
    # Initialize default data
    init_default_data()
    
    print(" Database initialized successfully!")
    
    # Test basic operations
    with get_db() as db:
        user_repo = UserRepository(db)
        form_repo = FormRepository(db)
        
        # Create test user
        test_user = user_repo.create_user(
            email="test@formagen.ai",
            full_name="Test User"
        )
        print(f"Created test user: {test_user.email}")
        
        # Create test form
        test_form_data = {
            "title": "Test Form",
            "description": "This is a test form",
            "keyword": "test form",
            "category": "it",
            "form_type": "registration",
            "complexity": "Simple",
            "fields": [
                {
                    "name": "name",
                    "type": "text",
                    "label": "Name",
                    "required": True
                }
            ]
        }
        
        test_form = form_repo.create_form(test_form_data, test_user.id)
        print(f"Created test form: {test_form.title}")
        
        # Analytics
        analytics_repo = AnalyticsRepository(db)
        stats = analytics_repo.get_form_stats()
        print(f"Database stats: {stats}")

if __name__ == "__main__":
    main()
