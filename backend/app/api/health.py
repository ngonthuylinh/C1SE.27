from flask_restx import Namespace, Resource
from datetime import datetime

api = Namespace('health', description='Health check endpoints')

@api.route('')
class HealthCheck(Resource):
    """Health check endpoint"""
    
    def get(self):
        """Kiểm tra trạng thái server"""
        from app.services.question_generator import QuestionGeneratorService
        
        service = QuestionGeneratorService.get_instance()
        health_status = service.get_health_status()
        
        return {
            'status': 'ok',
            'message': 'Server đang hoạt động',
            'model_loaded': health_status['model_loaded'],
            'timestamp': datetime.now().isoformat()
        }, 200