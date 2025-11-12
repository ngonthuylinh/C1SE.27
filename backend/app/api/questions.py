from flask_restx import Namespace, Resource, fields
from flask import request

api = Namespace('questions', description='Question generation endpoints')

# Request models
generate_request_model = api.model('GenerateQuestionsRequest', {
    'keyword': fields.String(required=True, description='Keyword để tạo câu hỏi'),
    'num_questions': fields.Integer(default=5, description='Số câu hỏi cần tạo (tối đa 20)'),
    'category': fields.String(description='Category mong muốn (optional)')
})

# Response models
question_response_model = api.model('QuestionResponse', {
    'question': fields.String(description='Câu hỏi được tạo'),
    'category': fields.String(description='Category của câu hỏi'),
    'confidence': fields.Float(description='Độ tin cậy'),
    'method': fields.String(description='Phương pháp tạo câu hỏi'),
    'source_keyword': fields.String(description='Keyword nguồn'),
    'similarity_score': fields.Float(description='Điểm tương tự (nếu có)')
})

@api.route('/generate')
class GenerateQuestions(Resource):
    """Generate questions từ keyword"""
    
    @api.expect(generate_request_model)
    @api.marshal_list_with(question_response_model)
    def post(self):
        """Tạo câu hỏi từ keyword"""
        from app.services.question_generator import QuestionGeneratorService
        
        data = request.get_json()
        if not data:
            api.abort(400, 'Request body is required')
        
        keyword = data.get('keyword')
        if not keyword:
            api.abort(400, 'Keyword is required')
        
        num_questions = data.get('num_questions', 5)
        category = data.get('category')
        
        # Validate num_questions
        if num_questions < 1:
            api.abort(400, 'num_questions must be at least 1')
        if num_questions > 20:
            api.abort(400, 'num_questions cannot exceed 20')
        
        try:
            service = QuestionGeneratorService.get_instance()
            
            if not service.is_model_loaded():
                api.abort(503, 'Model chưa được load. Hãy thử lại sau.')
            
            questions = service.generate_questions(
                keyword=keyword.strip(),
                num_questions=num_questions,
                category=category
            )
            
            return questions, 200
            
        except RuntimeError as e:
            api.abort(500, str(e))
        except Exception as e:
            api.abort(500, f'Internal server error: {str(e)}')

@api.route('/batch')
class BatchGenerateQuestions(Resource):
    """Generate questions cho nhiều keywords"""
    
    batch_request_model = api.model('BatchGenerateRequest', {
        'keywords': fields.List(fields.String, required=True, description='Danh sách keywords'),
        'num_questions_per_keyword': fields.Integer(default=3, description='Số câu hỏi cho mỗi keyword')
    })
    
    batch_response_model = api.model('BatchGenerateResponse', {
        'results': fields.Raw(description='Kết quả theo keyword'),
        'total_keywords': fields.Integer(description='Tổng số keywords'),
        'total_questions': fields.Integer(description='Tổng số câu hỏi tạo được')
    })
    
    @api.expect(batch_request_model)
    @api.marshal_with(batch_response_model)
    def post(self):
        """Tạo câu hỏi cho nhiều keywords"""
        from app.services.question_generator import QuestionGeneratorService
        
        data = request.get_json()
        if not data:
            api.abort(400, 'Request body is required')
        
        keywords = data.get('keywords')
        if not keywords or not isinstance(keywords, list):
            api.abort(400, 'Keywords list is required')
        
        if len(keywords) > 10:
            api.abort(400, 'Maximum 10 keywords per batch request')
        
        num_questions_per_keyword = data.get('num_questions_per_keyword', 3)
        if num_questions_per_keyword > 10:
            api.abort(400, 'num_questions_per_keyword cannot exceed 10')
        
        try:
            service = QuestionGeneratorService.get_instance()
            
            if not service.is_model_loaded():
                api.abort(503, 'Model chưa được load. Hãy thử lại sau.')
            
            results = {}
            total_questions = 0
            
            for keyword in keywords:
                if isinstance(keyword, str) and keyword.strip():
                    questions = service.generate_questions(
                        keyword=keyword.strip(),
                        num_questions=num_questions_per_keyword
                    )
                    results[keyword] = questions
                    total_questions += len(questions)
            
            return {
                'results': results,
                'total_keywords': len(results),
                'total_questions': total_questions
            }, 200
            
        except RuntimeError as e:
            api.abort(500, str(e))
        except Exception as e:
            api.abort(500, f'Internal server error: {str(e)}')