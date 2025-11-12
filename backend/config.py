import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Cấu hình cơ bản cho ứng dụng"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    MODEL_PATH = os.environ.get('MODEL_PATH') or '../models/real_data_question_model.pkl'
    HOST = os.environ.get('HOST') or '0.0.0.0'
    PORT = int(os.environ.get('PORT') or 8000)
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    # CORS settings
    CORS_ORIGINS = ['http://localhost:3000', 'http://127.0.0.1:3000']
    
    # API settings
    API_VERSION = 'v1'
    API_PREFIX = '/api'
    
    # Model settings
    DEFAULT_NUM_QUESTIONS = 5
    MAX_NUM_QUESTIONS = 20

class DevelopmentConfig(Config):
    """Cấu hình cho development"""
    DEBUG = True

class ProductionConfig(Config):
    """Cấu hình cho production"""
    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY')
    if not SECRET_KEY:
        raise ValueError("SECRET_KEY environment variable must be set for production")

class TestingConfig(Config):
    """Cấu hình cho testing"""
    TESTING = True
    DEBUG = True

# Cấu hình mặc định
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}