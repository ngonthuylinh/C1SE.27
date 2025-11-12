from flask_restx import Namespace, Resource, fields

api = Namespace('model', description='Model information endpoints')

# Response models
model_info_response_model = api.model('ModelInfoResponse', {
    'model_loaded': fields.Boolean(description='Model đã được load chưa'),
    'training_date': fields.String(description='Ngày train model'),
    'total_keywords': fields.Integer(description='Tổng số keywords trong model'),
    'total_questions': fields.Integer(description='Tổng số questions trong model'),
    'categories': fields.List(fields.String, description='Danh sách categories'),
    'model_accuracy': fields.Float(description='Độ chính xác của model')
})

@api.route('/info')
class ModelInfo(Resource):
    """Thông tin về model đã load"""
    
    @api.marshal_with(model_info_response_model)
    def get(self):
        """Lấy thông tin chi tiết về model"""
        from app.services.question_generator import QuestionGeneratorService
        
        try:
            service = QuestionGeneratorService.get_instance()
            model_info = service.get_model_info()
            
            return model_info, 200
            
        except Exception as e:
            api.abort(500, f'Internal server error: {str(e)}')

@api.route('/reload')
class ReloadModel(Resource):
    """Reload model"""
    
    def post(self):
        """Reload model từ file"""
        from app.services.question_generator import QuestionGeneratorService
        
        try:
            service = QuestionGeneratorService.get_instance()
            success = service.initialize_model()
            
            if success:
                return {
                    'status': 'success',
                    'message': 'Model đã được reload thành công'
                }, 200
            else:
                return {
                    'status': 'error',
                    'message': 'Không thể reload model'
                }, 500
                
        except Exception as e:
            api.abort(500, f'Internal server error: {str(e)}')

@api.route('/status')
class ModelStatus(Resource):
    """Trạng thái model"""
    
    def get(self):
        """Kiểm tra trạng thái model"""
        from app.services.question_generator import QuestionGeneratorService
        
        try:
            service = QuestionGeneratorService.get_instance()
            
            return {
                'model_loaded': service.is_model_loaded(),
                'status': 'ready' if service.is_model_loaded() else 'not_loaded',
                'message': 'Model sẵn sàng' if service.is_model_loaded() else 'Model chưa được load'
            }, 200
            
        except Exception as e:
            api.abort(500, f'Internal server error: {str(e)}')