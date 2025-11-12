#!/usr/bin/env python3
"""
Form Generation Engine
Engine tự động tạo form fields, validation rules, và structure từ keyword
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class FieldType(Enum):
    TEXT = "text"
    EMAIL = "email"
    PASSWORD = "password"
    NUMBER = "number"
    TEL = "tel"
    URL = "url"
    DATE = "date"
    TIME = "time"
    DATETIME = "datetime-local"
    TEXTAREA = "textarea"
    SELECT = "select"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    RANGE = "range"
    FILE = "file"
    HIDDEN = "hidden"

class ComplexityLevel(Enum):
    SIMPLE = "Simple"
    MODERATE = "Moderate"
    COMPLEX = "Complex"

@dataclass
class ValidationRule:
    """Validation rule for form field"""
    required: bool = False
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    pattern: Optional[str] = None
    custom_message: Optional[str] = None
    
    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}

@dataclass
class FormField:
    """Form field definition"""
    name: str
    field_type: FieldType
    label: str
    placeholder: Optional[str] = None
    description: Optional[str] = None
    required: bool = False
    default_value: Optional[Any] = None
    options: Optional[List[str]] = None
    validation: Optional[ValidationRule] = None
    conditional_logic: Optional[Dict] = None
    
    def to_dict(self):
        result = {
            'name': self.name,
            'type': self.field_type.value,
            'label': self.label,
            'required': self.required
        }
        
        if self.placeholder:
            result['placeholder'] = self.placeholder
        if self.description:
            result['description'] = self.description
        if self.default_value is not None:
            result['default_value'] = self.default_value
        if self.options:
            result['options'] = self.options
        if self.validation:
            result['validation'] = self.validation.to_dict()
        if self.conditional_logic:
            result['conditional_logic'] = self.conditional_logic
            
        return result

class FormGenerationEngine:
    """Engine tạo form tự động từ keyword và context"""
    
    def __init__(self):
        self.setup_keyword_mappings()
        self.setup_industry_templates()
        self.setup_validation_patterns()
    
    def setup_keyword_mappings(self):
        """Thiết lập mapping từ keywords đến field types"""
        self.keyword_to_field_type = {
            # Contact information
            'name': FieldType.TEXT,
            'email': FieldType.EMAIL,
            'phone': FieldType.TEL,
            'address': FieldType.TEXTAREA,
            'website': FieldType.URL,
            
            # Numbers and dates
            'age': FieldType.NUMBER,
            'salary': FieldType.NUMBER,
            'budget': FieldType.NUMBER,
            'price': FieldType.NUMBER,
            'quantity': FieldType.NUMBER,
            'date': FieldType.DATE,
            'time': FieldType.TIME,
            'birthday': FieldType.DATE,
            
            # Text areas
            'description': FieldType.TEXTAREA,
            'comment': FieldType.TEXTAREA,
            'feedback': FieldType.TEXTAREA,
            'message': FieldType.TEXTAREA,
            'note': FieldType.TEXTAREA,
            'requirement': FieldType.TEXTAREA,
            
            # Selections
            'gender': FieldType.RADIO,
            'country': FieldType.SELECT,
            'state': FieldType.SELECT,
            'category': FieldType.SELECT,
            'priority': FieldType.SELECT,
            'status': FieldType.SELECT,
            
            # Multiple choice
            'skills': FieldType.CHECKBOX,
            'interests': FieldType.CHECKBOX,
            'preferences': FieldType.CHECKBOX,
            'features': FieldType.CHECKBOX,
            
            # Ratings
            'rating': FieldType.RANGE,
            'satisfaction': FieldType.RANGE,
            'score': FieldType.RANGE,
            
            # Files
            'resume': FieldType.FILE,
            'portfolio': FieldType.FILE,
            'document': FieldType.FILE,
            'image': FieldType.FILE
        }
    
    def setup_industry_templates(self):
        """Thiết lập templates cho từng ngành"""
        self.industry_templates = {
            'it': {
                'common_fields': [
                    'programming_languages', 'experience_years', 'github_profile',
                    'certifications', 'project_examples', 'preferred_technologies'
                ],
                'field_options': {
                    'programming_languages': [
                        'Python', 'JavaScript', 'Java', 'C++', 'C#', 'PHP', 'Ruby',
                        'Go', 'Rust', 'TypeScript', 'Kotlin', 'Swift'
                    ],
                    'experience_level': ['Entry', 'Junior', 'Mid', 'Senior', 'Lead', 'Architect'],
                    'work_environment': ['Remote', 'Hybrid', 'On-site', 'Flexible'],
                    'project_type': ['Web App', 'Mobile App', 'Desktop App', 'API', 'Database', 'DevOps']
                }
            },
            'economics': {
                'common_fields': [
                    'income_range', 'investment_experience', 'risk_tolerance',
                    'financial_goals', 'time_horizon', 'current_portfolio'
                ],
                'field_options': {
                    'income_range': ['< $25k', '$25k-$50k', '$50k-$100k', '$100k-$200k', '> $200k'],
                    'investment_experience': ['None', 'Beginner', 'Intermediate', 'Advanced', 'Professional'],
                    'risk_tolerance': ['Very Low', 'Low', 'Moderate', 'High', 'Very High'],
                    'investment_type': ['Stocks', 'Bonds', 'Real Estate', 'Crypto', 'Commodities', 'Mutual Funds'],
                    'time_horizon': ['< 1 year', '1-3 years', '3-5 years', '5-10 years', '> 10 years']
                }
            },
            'marketing': {
                'common_fields': [
                    'target_audience', 'marketing_channels', 'campaign_budget',
                    'campaign_duration', 'success_metrics', 'brand_guidelines'
                ],
                'field_options': {
                    'marketing_channels': [
                        'Social Media', 'Email Marketing', 'Content Marketing', 'PPC',
                        'SEO', 'Influencer Marketing', 'Traditional Media', 'Events'
                    ],
                    'budget_range': ['< $1k', '$1k-$5k', '$5k-$25k', '$25k-$100k', '> $100k'],
                    'target_audience': ['B2B', 'B2C', 'Students', 'Professionals', 'Seniors', 'Millennials'],
                    'campaign_objective': [
                        'Brand Awareness', 'Lead Generation', 'Sales', 'Engagement', 'Retention'
                    ]
                }
            }
        }
    
    def setup_validation_patterns(self):
        """Thiết lập validation patterns"""
        self.validation_patterns = {
            'email': r'^[^\s@]+@[^\s@]+\.[^\s@]+$',
            'phone': r'^\+?[\d\s\-\(\)]{10,}$',
            'url': r'^https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$',
            'password': r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[a-zA-Z\d@$!%*?&]{8,}$',
            'zip_code': r'^\d{5}(-\d{4})?$',
            'credit_card': r'^\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}$'
        }
    
    def analyze_keyword(self, keyword: str, category: str) -> Dict[str, Any]:
        """Phân tích keyword để xác định form structure"""
        keyword_lower = keyword.lower()
        words = re.findall(r'\b\w+\b', keyword_lower)
        
        analysis = {
            'primary_intent': self._detect_primary_intent(words),
            'data_types': self._detect_data_types(words),
            'complexity_indicators': self._detect_complexity_indicators(words),
            'industry_specifics': self._detect_industry_specifics(words, category),
            'form_flow': self._suggest_form_flow(words, category)
        }
        
        return analysis
    
    def _detect_primary_intent(self, words: List[str]) -> str:
        """Xác định mục đích chính của form"""
        intent_keywords = {
            'registration': ['register', 'signup', 'join', 'create', 'account'],
            'survey': ['survey', 'poll', 'feedback', 'opinion', 'rate', 'evaluate'],
            'application': ['apply', 'application', 'job', 'position', 'hire'],
            'assessment': ['test', 'quiz', 'assessment', 'evaluation', 'skill'],
            'consultation': ['consult', 'consultation', 'advice', 'help', 'support'],
            'booking': ['book', 'reserve', 'schedule', 'appointment'],
            'quote': ['quote', 'estimate', 'price', 'cost', 'budget'],
            'contact': ['contact', 'inquire', 'question', 'reach']
        }
        
        for intent, keywords in intent_keywords.items():
            if any(word in words for word in keywords):
                return intent
        
        return 'general'
    
    def _detect_data_types(self, words: List[str]) -> List[str]:
        """Xác định các loại dữ liệu cần thu thập"""
        data_type_keywords = {
            'personal': ['name', 'personal', 'profile', 'individual'],
            'contact': ['email', 'phone', 'contact', 'address'],
            'financial': ['budget', 'price', 'cost', 'salary', 'income', 'investment'],
            'technical': ['skill', 'technology', 'tool', 'software', 'programming'],
            'preferences': ['prefer', 'like', 'choice', 'option', 'select'],
            'experience': ['experience', 'background', 'history', 'previous'],
            'goals': ['goal', 'objective', 'target', 'aim', 'purpose'],
            'timeline': ['time', 'duration', 'deadline', 'schedule', 'when']
        }
        
        detected_types = []
        for data_type, keywords in data_type_keywords.items():
            if any(word in words for word in keywords):
                detected_types.append(data_type)
        
        return detected_types or ['general']
    
    def _detect_complexity_indicators(self, words: List[str]) -> ComplexityLevel:
        """Xác định mức độ phức tạp của form"""
        complex_indicators = [
            'advanced', 'detailed', 'comprehensive', 'complete', 'thorough',
            'analysis', 'assessment', 'evaluation', 'optimization', 'strategy'
        ]
        
        simple_indicators = [
            'basic', 'simple', 'quick', 'easy', 'brief', 'short',
            'contact', 'info', 'signup'
        ]
        
        complex_count = sum(1 for word in words if word in complex_indicators)
        simple_count = sum(1 for word in words if word in simple_indicators)
        
        if complex_count > simple_count and complex_count >= 2:
            return ComplexityLevel.COMPLEX
        elif simple_count > complex_count and simple_count >= 2:
            return ComplexityLevel.SIMPLE
        else:
            return ComplexityLevel.MODERATE
    
    def _detect_industry_specifics(self, words: List[str], category: str) -> List[str]:
        """Xác định các yếu tố đặc thù của ngành"""
        industry_keywords = {
            'it': [
                'software', 'development', 'programming', 'code', 'system',
                'database', 'security', 'cloud', 'api', 'framework',
                'algorithm', 'architecture', 'devops', 'testing'
            ],
            'economics': [
                'investment', 'portfolio', 'financial', 'market', 'trading',
                'risk', 'return', 'asset', 'fund', 'stock', 'bond',
                'economics', 'finance', 'banking', 'insurance'
            ],
            'marketing': [
                'campaign', 'brand', 'advertising', 'promotion', 'customer',
                'market', 'sales', 'lead', 'conversion', 'engagement',
                'social', 'digital', 'content', 'seo', 'ppc'
            ]
        }
        
        relevant_keywords = []
        category_words = industry_keywords.get(category, [])
        
        for word in words:
            if word in category_words:
                relevant_keywords.append(word)
        
        return relevant_keywords
    
    def _suggest_form_flow(self, words: List[str], category: str) -> List[str]:
        """Đề xuất flow của form"""
        base_flow = ['basic_info', 'specific_requirements', 'preferences', 'confirmation']
        
        # Adjust flow based on category and keywords
        if category == 'it':
            if any(word in words for word in ['job', 'position', 'career']):
                return ['personal_info', 'technical_skills', 'experience', 'portfolio', 'availability']
            elif any(word in words for word in ['project', 'development']):
                return ['project_overview', 'technical_requirements', 'timeline', 'budget', 'contact']
        
        elif category == 'economics':
            if any(word in words for word in ['investment', 'portfolio']):
                return ['personal_info', 'financial_situation', 'investment_goals', 'risk_assessment', 'preferences']
            elif any(word in words for word in ['loan', 'credit']):
                return ['personal_info', 'financial_details', 'loan_requirements', 'documentation', 'terms']
        
        elif category == 'marketing':
            if any(word in words for word in ['campaign', 'advertising']):
                return ['campaign_overview', 'target_audience', 'budget_timeline', 'channels', 'success_metrics']
            elif any(word in words for word in ['brand', 'design']):
                return ['brand_overview', 'design_requirements', 'target_market', 'guidelines', 'deliverables']
        
        return base_flow
    
    def generate_fields(self, keyword: str, category: str, form_type: str, complexity: ComplexityLevel) -> List[FormField]:
        """Tạo danh sách fields cho form"""
        analysis = self.analyze_keyword(keyword, category)
        fields = []
        
        # Always start with basic contact info
        fields.extend(self._generate_basic_fields())
        
        # Add category-specific fields
        fields.extend(self._generate_category_fields(category, analysis['industry_specifics']))
        
        # Add intent-specific fields
        fields.extend(self._generate_intent_fields(analysis['primary_intent'], category))
        
        # Add complexity-based fields
        if complexity == ComplexityLevel.COMPLEX:
            fields.extend(self._generate_advanced_fields(category, analysis))
        
        # Add conditional fields based on keyword
        fields.extend(self._generate_conditional_fields(keyword, category, analysis))
        
        # Apply complexity adjustments
        fields = self._adjust_for_complexity(fields, complexity)
        
        return fields
    
    def _generate_basic_fields(self) -> List[FormField]:
        """Tạo các trường cơ bản"""
        return [
            FormField(
                name="full_name",
                field_type=FieldType.TEXT,
                label="Họ và tên",
                placeholder="Nhập họ và tên đầy đủ",
                required=True,
                validation=ValidationRule(required=True, min_length=2, max_length=100)
            ),
            FormField(
                name="email",
                field_type=FieldType.EMAIL,
                label="Email",
                placeholder="example@email.com",
                required=True,
                validation=ValidationRule(required=True, pattern=self.validation_patterns['email'])
            )
        ]
    
    def _generate_category_fields(self, category: str, industry_keywords: List[str]) -> List[FormField]:
        """Tạo các trường đặc thù cho category"""
        fields = []
        template = self.industry_templates.get(category, {})
        
        if category == 'it':
            if any(word in industry_keywords for word in ['programming', 'development', 'software']):
                fields.append(FormField(
                    name="programming_languages",
                    field_type=FieldType.CHECKBOX,
                    label="Ngôn ngữ lập trình",
                    options=template.get('field_options', {}).get('programming_languages', []),
                    required=True
                ))
                
                fields.append(FormField(
                    name="experience_years",
                    field_type=FieldType.NUMBER,
                    label="Số năm kinh nghiệm",
                    validation=ValidationRule(min_value=0, max_value=50),
                    required=True
                ))
        
        elif category == 'economics':
            if any(word in industry_keywords for word in ['investment', 'financial', 'portfolio']):
                fields.append(FormField(
                    name="investment_experience",
                    field_type=FieldType.SELECT,
                    label="Kinh nghiệm đầu tư",
                    options=template.get('field_options', {}).get('investment_experience', []),
                    required=True
                ))
                
                fields.append(FormField(
                    name="risk_tolerance",
                    field_type=FieldType.SELECT,
                    label="Khả năng chấp nhận rủi ro",
                    options=template.get('field_options', {}).get('risk_tolerance', []),
                    required=True
                ))
        
        elif category == 'marketing':
            if any(word in industry_keywords for word in ['campaign', 'marketing', 'advertising']):
                fields.append(FormField(
                    name="marketing_channels",
                    field_type=FieldType.CHECKBOX,
                    label="Kênh marketing",
                    options=template.get('field_options', {}).get('marketing_channels', []),
                    required=True
                ))
                
                fields.append(FormField(
                    name="budget_range",
                    field_type=FieldType.SELECT,
                    label="Ngân sách dự kiến",
                    options=template.get('field_options', {}).get('budget_range', []),
                    required=True
                ))
        
        return fields
    
    def _generate_intent_fields(self, intent: str, category: str) -> List[FormField]:
        """Tạo các trường dựa trên mục đích"""
        fields = []
        
        if intent == 'consultation':
            fields.append(FormField(
                name="consultation_topic",
                field_type=FieldType.TEXTAREA,
                label="Chủ đề tư vấn",
                placeholder="Mô tả vấn đề bạn muốn tư vấn",
                required=True,
                validation=ValidationRule(required=True, min_length=10, max_length=1000)
            ))
            
            fields.append(FormField(
                name="preferred_contact_method",
                field_type=FieldType.RADIO,
                label="Phương thức liên hệ ưa thích",
                options=["Email", "Điện thoại", "Video call", "Trực tiếp"],
                required=True
            ))
        
        elif intent == 'survey':
            fields.append(FormField(
                name="overall_satisfaction",
                field_type=FieldType.RANGE,
                label="Mức độ hài lòng tổng thể",
                validation=ValidationRule(min_value=1, max_value=10),
                required=True
            ))
            
            fields.append(FormField(
                name="feedback_comments",
                field_type=FieldType.TEXTAREA,
                label="Nhận xét và góp ý",
                placeholder="Chia sẻ ý kiến của bạn",
                validation=ValidationRule(max_length=1000)
            ))
        
        elif intent == 'application':
            fields.append(FormField(
                name="position_applied",
                field_type=FieldType.TEXT,
                label="Vị trí ứng tuyển",
                required=True
            ))
            
            fields.append(FormField(
                name="resume_file",
                field_type=FieldType.FILE,
                label="CV/Resume",
                description="Tải lên file CV (PDF, DOC, DOCX)",
                required=True
            ))
        
        return fields
    
    def _generate_advanced_fields(self, category: str, analysis: Dict) -> List[FormField]:
        """Tạo các trường nâng cao cho form phức tạp"""
        fields = []
        
        # Add detailed requirements field
        fields.append(FormField(
            name="detailed_requirements",
            field_type=FieldType.TEXTAREA,
            label="Yêu cầu chi tiết",
            placeholder="Mô tả chi tiết các yêu cầu và mong đợi",
            validation=ValidationRule(min_length=50, max_length=2000)
        ))
        
        # Add priority field
        fields.append(FormField(
            name="priority_level",
            field_type=FieldType.SELECT,
            label="Mức độ ưu tiên",
            options=["Thấp", "Trung bình", "Cao", "Khẩn cấp"],
            required=True
        ))
        
        # Add timeline field
        fields.append(FormField(
            name="expected_timeline",
            field_type=FieldType.SELECT,
            label="Thời gian mong muốn",
            options=["Ngay lập tức", "1-2 tuần", "1 tháng", "2-3 tháng", "Linh hoạt"],
            required=True
        ))
        
        return fields
    
    def _generate_conditional_fields(self, keyword: str, category: str, analysis: Dict) -> List[FormField]:
        """Tạo các trường có điều kiện dựa trên từ khóa cụ thể"""
        fields = []
        keyword_lower = keyword.lower()
        
        # Security-related fields
        if 'security' in keyword_lower:
            fields.append(FormField(
                name="security_concerns",
                field_type=FieldType.CHECKBOX,
                label="Vấn đề bảo mật quan tâm",
                options=[
                    "Mã hóa dữ liệu", "Kiểm soát truy cập", "Đánh giá lỗ hổng",
                    "Tuân thủ quy định", "Giám sát hệ thống", "Backup & Recovery"
                ]
            ))
        
        # Budget-related fields
        if any(word in keyword_lower for word in ['budget', 'cost', 'price', 'investment']):
            fields.append(FormField(
                name="budget_details",
                field_type=FieldType.NUMBER,
                label="Ngân sách cụ thể (VNĐ)",
                placeholder="Nhập số tiền",
                validation=ValidationRule(min_value=0)
            ))
        
        # Timeline-related fields
        if any(word in keyword_lower for word in ['urgent', 'asap', 'deadline', 'schedule']):
            fields.append(FormField(
                name="deadline_date",
                field_type=FieldType.DATE,
                label="Thời hạn hoàn thành",
                required=True
            ))
        
        return fields
    
    def _adjust_for_complexity(self, fields: List[FormField], complexity: ComplexityLevel) -> List[FormField]:
        """Điều chỉnh fields dựa trên độ phức tạp"""
        if complexity == ComplexityLevel.SIMPLE:
            # Keep only essential fields
            essential_types = [FieldType.TEXT, FieldType.EMAIL, FieldType.SELECT, FieldType.RADIO]
            fields = [f for f in fields if f.field_type in essential_types or f.required]
            
            # Limit to maximum 6 fields for simple forms
            if len(fields) > 6:
                # Keep required fields and most important optional ones
                required_fields = [f for f in fields if f.required]
                optional_fields = [f for f in fields if not f.required]
                fields = required_fields + optional_fields[:6-len(required_fields)]
        
        elif complexity == ComplexityLevel.COMPLEX:
            # Add more validation rules and descriptions
            for field in fields:
                if not field.description and field.field_type in [FieldType.TEXTAREA, FieldType.CHECKBOX]:
                    field.description = f"Vui lòng cung cấp thông tin chi tiết về {field.label.lower()}"
                
                # Add more strict validation for complex forms
                if field.validation and field.field_type == FieldType.TEXT:
                    if not field.validation.min_length:
                        field.validation.min_length = 3
        
        return fields
    
    def generate_form_structure(self, keyword: str, category: str, form_type: str = None, complexity: ComplexityLevel = None) -> Dict[str, Any]:
        """Tạo cấu trúc form hoàn chỉnh"""
        # Auto-detect complexity if not provided
        if not complexity:
            analysis = self.analyze_keyword(keyword, category)
            complexity = analysis['complexity_indicators']
        
        # Generate fields
        fields = self.generate_fields(keyword, category, form_type or 'general', complexity)
        
        # Convert fields to dict format
        fields_dict = [field.to_dict() for field in fields]
        
        # Generate form structure
        form_structure = {
            'form_id': self._generate_form_id(),
            'title': self._generate_form_title(keyword),
            'description': self._generate_form_description(keyword, category),
            'category': category,
            'form_type': form_type or 'general',
            'complexity': complexity.value,
            'keyword': keyword,
            'fields': fields_dict,
            'sections': self._organize_fields_into_sections(fields_dict),
            'validation_rules': self._generate_form_validation_rules(fields_dict),
            'styling': self._generate_form_styling(category, complexity),
            'behavior': self._generate_form_behavior(complexity),
            'metadata': {
                'field_count': len(fields_dict),
                'required_fields_count': len([f for f in fields_dict if f.get('required', False)]),
                'estimated_completion_time': self._estimate_completion_time(fields_dict, complexity),
                'created_at': datetime.now().isoformat(),
                'generated_by': 'FormGenerationEngine'
            }
        }
        
        return form_structure
    
    def _generate_form_id(self) -> str:
        """Tạo unique form ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"form_{timestamp}_{hash(timestamp) % 10000:04d}"
    
    def _generate_form_title(self, keyword: str) -> str:
        """Tạo tiêu đề form"""
        return f"Form: {keyword.title()}"
    
    def _generate_form_description(self, keyword: str, category: str) -> str:
        """Tạo mô tả form"""
        category_names = {
            'it': 'Công nghệ thông tin',
            'economics': 'Kinh tế - Tài chính',
            'marketing': 'Marketing'
        }
        
        category_name = category_names.get(category, category.title())
        return f"Form thu thập thông tin về {keyword} trong lĩnh vực {category_name}"
    
    def _organize_fields_into_sections(self, fields: List[Dict]) -> List[Dict]:
        """Tổ chức fields thành các sections"""
        sections = []
        
        # Basic Information Section
        basic_fields = []
        for field in fields:
            if field['name'] in ['full_name', 'email', 'phone', 'company', 'position']:
                basic_fields.append(field)
        
        if basic_fields:
            sections.append({
                'id': 'basic_info',
                'title': 'Thông tin cơ bản',
                'description': 'Vui lòng cung cấp thông tin liên hệ cơ bản',
                'fields': basic_fields
            })
        
        # Requirements/Preferences Section
        requirement_fields = []
        for field in fields:
            if field not in basic_fields and field['type'] in ['select', 'checkbox', 'radio', 'range']:
                requirement_fields.append(field)
        
        if requirement_fields:
            sections.append({
                'id': 'requirements',
                'title': 'Yêu cầu và Sở thích',
                'description': 'Cho chúng tôi biết về nhu cầu và sở thích của bạn',
                'fields': requirement_fields
            })
        
        # Detailed Information Section
        detailed_fields = []
        for field in fields:
            if field not in basic_fields and field not in requirement_fields:
                detailed_fields.append(field)
        
        if detailed_fields:
            sections.append({
                'id': 'detailed_info',
                'title': 'Thông tin chi tiết',
                'description': 'Cung cấp thêm thông tin chi tiết',
                'fields': detailed_fields
            })
        
        return sections
    
    def _generate_form_validation_rules(self, fields: List[Dict]) -> Dict[str, Any]:
        """Tạo validation rules cho toàn form"""
        return {
            'required_fields': [f['name'] for f in fields if f.get('required', False)],
            'field_dependencies': self._detect_field_dependencies(fields),
            'cross_field_validation': self._generate_cross_field_validation(fields)
        }
    
    def _detect_field_dependencies(self, fields: List[Dict]) -> Dict[str, List[str]]:
        """Phát hiện dependencies giữa các fields"""
        dependencies = {}
        
        # Example: if has budget field, might need budget_details
        field_names = [f['name'] for f in fields]
        
        if 'budget_range' in field_names and 'budget_details' in field_names:
            dependencies['budget_details'] = ['budget_range']
        
        if 'other_option' in [opt for f in fields for opt in f.get('options', [])]:
            # Find field with "Other" option
            for field in fields:
                if 'Other' in field.get('options', []):
                    dependencies[f"{field['name']}_other"] = [field['name']]
        
        return dependencies
    
    def _generate_cross_field_validation(self, fields: List[Dict]) -> List[Dict]:
        """Tạo cross-field validation rules"""
        rules = []
        
        # Example: email confirmation
        field_names = [f['name'] for f in fields]
        if 'email' in field_names and 'email_confirm' in field_names:
            rules.append({
                'type': 'field_match',
                'fields': ['email', 'email_confirm'],
                'message': 'Email và xác nhận email phải giống nhau'
            })
        
        return rules
    
    def _generate_form_styling(self, category: str, complexity: ComplexityLevel) -> Dict[str, Any]:
        """Tạo styling configuration"""
        color_schemes = {
            'it': {
                'primary': '#0066cc',
                'secondary': '#f0f8ff',
                'accent': '#00cc66',
                'text': '#333333',
                'border': '#cccccc'
            },
            'economics': {
                'primary': '#2e8b57',
                'secondary': '#f0fff0',
                'accent': '#ffd700',
                'text': '#2f4f4f',
                'border': '#90ee90'
            },
            'marketing': {
                'primary': '#ff6347',
                'secondary': '#fff5ee',
                'accent': '#ff69b4',
                'text': '#8b0000',
                'border': '#ffb6c1'
            }
        }
        
        layout_configs = {
            ComplexityLevel.SIMPLE: {
                'columns': 1,
                'spacing': 'compact',
                'field_size': 'medium'
            },
            ComplexityLevel.MODERATE: {
                'columns': 2,
                'spacing': 'normal',
                'field_size': 'medium'
            },
            ComplexityLevel.COMPLEX: {
                'columns': 2,
                'spacing': 'comfortable',
                'field_size': 'large'
            }
        }
        
        return {
            'colors': color_schemes.get(category, color_schemes['it']),
            'layout': layout_configs.get(complexity, layout_configs[ComplexityLevel.MODERATE]),
            'typography': {
                'font_family': 'Arial, sans-serif',
                'font_size_base': '16px',
                'line_height': '1.5'
            },
            'components': {
                'border_radius': '8px',
                'shadow': '0 2px 4px rgba(0,0,0,0.1)',
                'focus_outline': '2px solid #0066cc'
            }
        }
    
    def _generate_form_behavior(self, complexity: ComplexityLevel) -> Dict[str, Any]:
        """Tạo behavior configuration"""
        return {
            'auto_save': complexity != ComplexityLevel.SIMPLE,
            'progress_indicator': complexity == ComplexityLevel.COMPLEX,
            'field_validation': 'on_blur',  # validate when user leaves field
            'submit_confirmation': True,
            'error_display': 'inline',
            'animations': {
                'enabled': complexity != ComplexityLevel.SIMPLE,
                'duration': '0.3s',
                'easing': 'ease-in-out'
            },
            'accessibility': {
                'keyboard_navigation': True,
                'screen_reader_support': True,
                'high_contrast_mode': True
            }
        }
    
    def _estimate_completion_time(self, fields: List[Dict], complexity: ComplexityLevel) -> int:
        """Ước tính thời gian hoàn thành (phút)"""
        base_time_per_field = {
            'text': 0.5,
            'email': 0.5,
            'tel': 0.5,
            'number': 0.5,
            'select': 0.3,
            'radio': 0.3,
            'checkbox': 0.5,
            'textarea': 1.5,
            'file': 1.0,
            'date': 0.3,
            'range': 0.2
        }
        
        total_time = 0
        for field in fields:
            field_type = field.get('type', 'text')
            base_time = base_time_per_field.get(field_type, 0.5)
            
            # Adjust for field complexity
            if field.get('required', False):
                base_time *= 1.2  # Required fields take more time
            
            if field.get('options') and len(field['options']) > 5:
                base_time *= 1.3  # Many options take more time
            
            total_time += base_time
        
        # Complexity adjustment
        complexity_multiplier = {
            ComplexityLevel.SIMPLE: 0.8,
            ComplexityLevel.MODERATE: 1.0,
            ComplexityLevel.COMPLEX: 1.5
        }
        
        total_time *= complexity_multiplier.get(complexity, 1.0)
        
        return max(1, int(total_time))  # At least 1 minute

def main():
    """Demo function"""
    print("🔧 Form Generation Engine Demo")
    
    engine = FormGenerationEngine()
    
    test_cases = [
        ("cloud security assessment", "it"),
        ("investment portfolio planning", "economics"),
        ("digital marketing campaign", "marketing")
    ]
    
    for keyword, category in test_cases:
        print(f"\n📝 Generating form for: '{keyword}' in {category}")
        
        form_structure = engine.generate_form_structure(keyword, category)
        
        print(f"   Title: {form_structure['title']}")
        print(f"   Complexity: {form_structure['complexity']}")
        print(f"   Fields: {form_structure['metadata']['field_count']}")
        print(f"   Sections: {len(form_structure['sections'])}")
        print(f"   Estimated time: {form_structure['metadata']['estimated_completion_time']} minutes")
        
        # Save to file for inspection
        filename = f"form_{keyword.replace(' ', '_')}_{category}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(form_structure, f, indent=2, ensure_ascii=False)
        print(f"   Saved to: {filename}")

if __name__ == "__main__":
    main()
