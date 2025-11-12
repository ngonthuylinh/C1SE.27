import logging
from datetime import datetime

def setup_logging():
    """Cấu hình logging cho ứng dụng"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('app.log')
        ]
    )

def validate_keyword(keyword: str) -> bool:
    """Validate keyword input"""
    if not keyword or not isinstance(keyword, str):
        return False
    
    keyword = keyword.strip()
    if len(keyword) < 2 or len(keyword) > 200:
        return False
    
    return True

def sanitize_keyword(keyword: str) -> str:
    """Làm sạch keyword input"""
    if not keyword:
        return ""
    
    # Remove extra whitespace
    keyword = " ".join(keyword.split())
    
    # Remove special characters (keep alphanumeric, spaces, hyphens)
    import re
    keyword = re.sub(r'[^a-zA-Z0-9\s\-]', '', keyword)
    
    return keyword.strip().lower()

def format_response(data, status_code=200, message=None):
    """Format API response"""
    response = {
        'success': status_code < 400,
        'data': data,
        'timestamp': datetime.now().isoformat()
    }
    
    if message:
        response['message'] = message
    
    return response, status_code