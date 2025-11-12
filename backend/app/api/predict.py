from flask_restx import Namespace, Resource, fields
from flask import request

api = Namespace('predict', description='Prediction endpoints')

# Request models
category_request_model = api.model('CategoryPredictionRequest', {
    'keyword': fields.String(required=True, description='Keyword để dự đoán category')
})

# Response models
category_response_model = api.model('CategoryPredictionResponse', {
    'keyword': fields.String(description='Keyword đầu vào'),
    'predicted_category': fields.String(description='Category được dự đoán'),
    'confidence': fields.Float(description='Độ tin cậy'),
    'all_probabilities': fields.Raw(description='Xác suất tất cả categories')
})

@api.route('/category')
class PredictCategory(Resource):
    """Dự đoán category từ keyword"""
    
    @api.expect(category_request_model)
    @api.marshal_with(category_response_model)
    def post(self):
        """Dự đoán category cho keyword"""
        from app.services.question_generator import QuestionGeneratorService
        
        data = request.get_json()
        if not data:
            api.abort(400, 'Request body is required')
        
        keyword = data.get('keyword')
        if not keyword:
            api.abort(400, 'Keyword is required')
        
        try:
            service = QuestionGeneratorService.get_instance()
            
            if not service.is_model_loaded():
                api.abort(503, 'Model chưa được load. Hãy thử lại sau.')
            
            prediction = service.predict_category(keyword.strip())
            
            return prediction, 200
            
        except RuntimeError as e:
            api.abort(500, str(e))
        except Exception as e:
            api.abort(500, f'Internal server error: {str(e)}')

@api.route('/batch-category')
class BatchPredictCategory(Resource):
    """Dự đoán category cho nhiều keywords"""
    
    batch_category_request_model = api.model('BatchCategoryRequest', {
        'keywords': fields.List(fields.String, required=True, description='Danh sách keywords')
    })
    
    batch_category_response_model = api.model('BatchCategoryResponse', {
        'results': fields.Raw(description='Kết quả dự đoán theo keyword'),
        'total_keywords': fields.Integer(description='Tổng số keywords')
    })
    
    @api.expect(batch_category_request_model)
    @api.marshal_with(batch_category_response_model)
    def post(self):
        """Dự đoán category cho nhiều keywords"""
        from app.services.question_generator import QuestionGeneratorService
        
        data = request.get_json()
        if not data:
            api.abort(400, 'Request body is required')
        
        keywords = data.get('keywords')
        if not keywords or not isinstance(keywords, list):
            api.abort(400, 'Keywords list is required')
        
        if len(keywords) > 50:
            api.abort(400, 'Maximum 50 keywords per batch request')
        
        try:
            service = QuestionGeneratorService.get_instance()
            
            if not service.is_model_loaded():
                api.abort(503, 'Model chưa được load. Hãy thử lại sau.')
            
            results = {}
            
            for keyword in keywords:
                if isinstance(keyword, str) and keyword.strip():
                    prediction = service.predict_category(keyword.strip())
                    results[keyword] = prediction
            
            return {
                'results': results,
                'total_keywords': len(results)
            }, 200
            
        except RuntimeError as e:
            api.abort(500, str(e))
        except Exception as e:
            api.abort(500, f'Internal server error: {str(e)}')