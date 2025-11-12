from flask import Flask
from flask_cors import CORS
from flask_restx import Api
import os
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config

def create_app(config_name=None):
    """Factory function để tạo Flask app"""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')
    
    app = Flask(__name__)
    app.config.from_object(config.get(config_name, config['default']))
    
    # Initialize extensions
    CORS(app, origins=app.config['CORS_ORIGINS'])
    
    # Initialize Flask-RESTX
    api = Api(
        app,
        version='1.0',
        title='Form Agent AI API',
        description='API cho hệ thống tạo câu hỏi thông minh từ keywords',
        doc='/docs/',
        prefix=app.config['API_PREFIX']
    )
    
    # Register namespaces
    from app.api.questions import api as questions_ns
    from app.api.predict import api as predict_ns
    from app.api.model import api as model_ns
    from app.api.health import api as health_ns
    
    api.add_namespace(health_ns, path='/health')
    api.add_namespace(questions_ns, path='/questions')
    api.add_namespace(predict_ns, path='/predict')
    api.add_namespace(model_ns, path='/model')
    
    # Initialize model service with app context
    with app.app_context():
        from app.services.question_generator import QuestionGeneratorService
        QuestionGeneratorService.get_instance().initialize_model()
    
    return app