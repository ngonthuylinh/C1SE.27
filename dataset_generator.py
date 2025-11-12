#!/usr/bin/env python3
"""
Dataset Generator cho Form Agent AI
Tạo 500,000,000 mẫu dữ liệu cho 3 lĩnh vực: IT, Kinh tế, Marketing
"""

import pandas as pd
import numpy as np
import random
import os
import re
import json
import uuid
from itertools import combinations, product
from datetime import datetime, timedelta
from tqdm import tqdm
import logging
import gc
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetGenerator:
    def __init__(self):
        """Initialize dataset generator with comprehensive data sources"""
        self.setup_data_sources()
        self.setup_form_structures()
        
    def setup_data_sources(self):
        """Thiết lập các nguồn dữ liệu cho từng lĩnh vực"""
        
        # IT Keywords - Mở rộng danh sách lớn
        self.it_keywords = [
            # Core Technologies
            "cloud computing", "machine learning", "artificial intelligence", "data analytics",
            "cybersecurity", "blockchain", "IoT", "big data", "software development",
            "web development", "mobile app", "database design", "API development",
            "DevOps", "microservices", "serverless", "containerization", "kubernetes",
            
            # Programming Languages
            "python programming", "java development", "javascript coding", "typescript",
            "c++ programming", "c# development", "ruby coding", "php development",
            "go programming", "rust development", "kotlin mobile", "swift ios",
            "dart flutter", "scala development", "r analytics", "matlab engineering",
            
            # Frameworks & Libraries
            "react framework", "angular development", "vue frontend", "nodejs backend",
            "django python", "flask api", "spring boot", "laravel php", "express js",
            "tensorflow ml", "pytorch deep learning", "scikit-learn", "opencv computer vision",
            "pandas data analysis", "numpy scientific computing", "fastapi", "nestjs",
            
            # Cloud & Infrastructure
            "aws cloud services", "azure microsoft cloud", "google cloud platform",
            "docker containers", "jenkins ci cd", "terraform infrastructure",
            "ansible automation", "mongodb nosql", "postgresql database", "mysql rdbms",
            "redis caching", "elasticsearch search", "cassandra database", "kafka messaging",
            
            # Development Practices
            "agile methodology", "scrum framework", "continuous integration", "continuous deployment",
            "test driven development", "automation testing", "performance monitoring",
            "git version control", "code review process", "pair programming",
            "clean code practices", "software architecture", "design patterns",
            
            # Security & Compliance
            "network security", "data encryption", "user authentication", "authorization systems",
            "penetration testing", "vulnerability assessment", "GDPR compliance", "HIPAA security",
            "oauth implementation", "jwt tokens", "ssl certificates", "firewall configuration",
            
            # Performance & Scalability
            "load balancing", "horizontal scalability", "performance optimization", "debugging techniques",
            "memory management", "caching strategies", "CDN implementation", "database optimization",
            "microservices architecture", "distributed systems", "event driven architecture",
            
            # Emerging Technologies
            "quantum computing", "edge computing", "5G technology", "augmented reality development",
            "virtual reality applications", "computer vision", "natural language processing",
            "robotics programming", "autonomous systems", "smart contracts", "web3 development"
        ]
        
        # Economics Keywords - Mở rộng danh sách lớn
        self.economics_keywords = [
            # Financial Analysis & Planning
            "market analysis", "financial planning", "investment strategy", "portfolio management",
            "risk assessment", "economic indicators", "inflation analysis", "GDP forecasting",
            "unemployment trends", "fiscal policy analysis", "monetary policy", "trade balance",
            "exchange rate analysis", "currency hedging", "financial modeling", "valuation methods",
            
            # Investment & Trading
            "stock market analysis", "bond investment", "derivatives trading", "commodities investment",
            "real estate investment", "forex trading", "cryptocurrency trading", "options strategy",
            "futures contracts", "hedge fund management", "mutual fund analysis", "ETF investment",
            "REIT analysis", "private equity", "venture capital", "IPO evaluation",
            
            # Banking & Financial Services
            "commercial banking", "investment banking", "insurance analysis", "fintech innovation",
            "digital payments", "blockchain finance", "credit analysis", "loan assessment",
            "mortgage evaluation", "personal finance", "retirement planning", "wealth management",
            "corporate finance", "mergers acquisitions", "financial restructuring", "debt management",
            
            # Business Economics & Strategy
            "budget analysis", "cost accounting", "management accounting", "financial reporting",
            "cash flow analysis", "working capital management", "capital budgeting", "investment appraisal",
            "business valuation", "company analysis", "industry analysis", "competitive analysis",
            "market research", "consumer behavior", "pricing strategy", "revenue optimization",
            
            # Economic Theory & Policy
            "supply demand analysis", "price elasticity", "market equilibrium", "economic growth",
            "business cycle analysis", "recession indicators", "economic forecasting", "policy analysis",
            "international trade", "global economics", "emerging markets", "economic development",
            "behavioral economics", "game theory", "econometrics", "statistical analysis",
            
            # Specialized Finance Areas
            "environmental economics", "health economics", "labor economics", "public economics",
            "development economics", "agricultural economics", "energy economics", "transport economics",
            "digital economics", "sharing economy", "platform economics", "network effects"
        ]
        
        # Marketing Keywords - Mở rộng danh sách lớn
        self.marketing_keywords = [
            # Digital Marketing Fundamentals
            "digital marketing strategy", "online marketing", "internet marketing", "digital advertising",
            "marketing automation", "customer journey mapping", "conversion optimization", "lead generation",
            "customer acquisition", "customer retention", "lifetime value", "marketing funnel",
            
            # Content Marketing
            "content marketing strategy", "content creation", "storytelling marketing", "brand storytelling",
            "video marketing", "podcast marketing", "webinar marketing", "blog marketing",
            "visual content", "infographic design", "copywriting", "content calendar",
            "editorial calendar", "content distribution", "content amplification", "viral content",
            
            # Social Media Marketing
            "social media strategy", "facebook marketing", "instagram marketing", "twitter marketing",
            "linkedin marketing", "youtube marketing", "tiktok marketing", "pinterest marketing",
            "social media advertising", "community management", "influencer marketing", "micro influencers",
            "social media analytics", "engagement strategy", "social listening", "user generated content",
            
            # Search Engine Marketing
            "search engine optimization", "keyword research", "on page seo", "technical seo",
            "link building", "local seo", "voice search optimization", "search engine marketing",
            "pay per click advertising", "google ads", "facebook ads", "display advertising",
            "remarketing campaigns", "programmatic advertising", "native advertising",
            
            # Email & Direct Marketing
            "email marketing campaigns", "email automation", "newsletter marketing", "drip campaigns",
            "personalized email", "email deliverability", "list building", "segmentation strategy",
            "direct mail marketing", "telemarketing", "sms marketing", "push notifications",
            
            # Analytics & Performance
            "marketing analytics", "web analytics", "conversion tracking", "attribution modeling",
            "customer analytics", "behavioral analytics", "predictive analytics", "marketing roi",
            "kpi tracking", "dashboard reporting", "ab testing", "multivariate testing",
            "statistical analysis", "data visualization", "marketing intelligence",
            
            # Brand & Strategy
            "brand management", "brand positioning", "brand identity", "brand awareness",
            "brand equity", "rebranding strategy", "brand guidelines", "corporate branding",
            "product marketing", "go to market strategy", "competitive positioning", "market segmentation",
            "target audience analysis", "buyer persona development", "customer research",
            
            # Emerging Marketing Trends
            "growth hacking", "viral marketing", "guerrilla marketing", "experiential marketing",
            "event marketing", "trade show marketing", "partnership marketing", "affiliate marketing",
            "referral marketing", "loyalty programs", "gamification marketing", "ai marketing",
            "chatbot marketing", "voice marketing", "ar marketing", "mobile marketing"
        ]
        
    def setup_form_structures(self):
        """Setup form templates for each category"""
        
        self.it_form_templates = {
            "registration": {
                "fields": ["full_name", "email", "phone", "company", "position", "experience_level", "programming_skills", "certifications", "github_profile", "linkedin_profile"],
                "complexity": "Moderate",
                "sections": ["personal_info", "professional_info", "technical_skills", "portfolio"]
            },
            "survey": {
                "fields": ["satisfaction_rating", "usage_frequency", "preferred_technologies", "current_challenges", "feature_requests", "improvement_suggestions"],
                "complexity": "Simple",
                "sections": ["feedback", "preferences", "suggestions"]
            },
            "application": {
                "fields": ["personal_info", "technical_skills", "work_experience", "education", "portfolio_projects", "availability", "salary_expectation", "references"],
                "complexity": "Complex",
                "sections": ["personal", "technical", "experience", "portfolio", "logistics"]
            },
            "assessment": {
                "fields": ["current_knowledge", "practical_experience", "project_examples", "learning_objectives", "skill_certifications", "career_goals"],
                "complexity": "Moderate",
                "sections": ["current_skills", "experience", "goals", "objectives"]
            },
            "consultation": {
                "fields": ["project_description", "technical_requirements", "timeline", "budget_range", "team_size", "existing_infrastructure"],
                "complexity": "Complex",
                "sections": ["project_details", "requirements", "constraints", "resources"]
            },
            "feedback": {
                "fields": ["overall_rating", "feature_usefulness", "performance_rating", "ui_feedback", "suggestions", "recommendation_likelihood"],
                "complexity": "Simple",
                "sections": ["ratings", "detailed_feedback", "recommendations"]
            }
        }
        
        self.economics_form_templates = {
            "financial_assessment": {
                "fields": ["annual_income", "monthly_expenses", "current_assets", "liabilities", "investment_goals", "risk_tolerance", "investment_timeline"],
                "complexity": "Complex",
                "sections": ["current_situation", "financial_goals", "risk_profile", "timeline"]
            },
            "investment_application": {
                "fields": ["investment_amount", "investment_period", "expected_returns", "risk_profile", "investment_experience", "financial_knowledge"],
                "complexity": "Moderate",
                "sections": ["investment_details", "investor_profile", "experience_level"]
            },
            "market_survey": {
                "fields": ["industry_sector", "market_size", "key_competitors", "pricing_analysis", "customer_segments", "growth_trends", "market_challenges"],
                "complexity": "Complex",
                "sections": ["market_overview", "competitive_analysis", "trends_challenges"]
            },
            "business_plan": {
                "fields": ["business_model", "revenue_streams", "cost_structure", "market_opportunity", "funding_requirements", "financial_projections"],
                "complexity": "Complex",
                "sections": ["business_model", "market_analysis", "financial_planning"]
            },
            "consultation": {
                "fields": ["current_financial_situation", "financial_objectives", "time_horizon", "constraints", "preferred_investment_types"],
                "complexity": "Moderate",
                "sections": ["current_status", "objectives", "preferences", "constraints"]
            },
            "portfolio_review": {
                "fields": ["current_portfolio", "performance_evaluation", "risk_analysis", "rebalancing_needs", "new_opportunities"],
                "complexity": "Complex",
                "sections": ["current_portfolio", "performance", "optimization"]
            }
        }
        
        self.marketing_form_templates = {
            "campaign_brief": {
                "fields": ["campaign_objectives", "target_audience", "marketing_budget", "campaign_timeline", "preferred_channels", "success_metrics", "brand_guidelines"],
                "complexity": "Complex",
                "sections": ["objectives", "audience", "budget_timeline", "execution", "measurement"]
            },
            "customer_survey": {
                "fields": ["demographics", "psychographics", "buying_preferences", "brand_perception", "satisfaction_rating", "feedback_comments", "referral_likelihood"],
                "complexity": "Moderate",
                "sections": ["demographics", "preferences", "brand_perception", "satisfaction"]
            },
            "lead_capture": {
                "fields": ["contact_information", "company_details", "business_needs", "budget_range", "decision_timeline", "current_solutions"],
                "complexity": "Simple",
                "sections": ["contact_info", "company_info", "requirements"]
            },
            "consultation_request": {
                "fields": ["business_type", "current_marketing_challenges", "marketing_goals", "target_market", "budget_constraints", "preferred_communication"],
                "complexity": "Moderate",
                "sections": ["business_info", "challenges", "goals", "constraints"]
            },
            "event_registration": {
                "fields": ["attendee_information", "session_preferences", "networking_interests", "dietary_requirements", "special_accommodations", "follow_up_preferences"],
                "complexity": "Simple",
                "sections": ["personal_info", "event_preferences", "special_requirements"]
            },
            "brand_audit": {
                "fields": ["current_brand_perception", "competitive_landscape", "brand_strengths", "brand_weaknesses", "improvement_areas", "brand_goals"],
                "complexity": "Complex",
                "sections": ["current_state", "competitive_analysis", "improvement_opportunities"]
            }
        }

    def generate_form_fields(self, form_type, category):
        """Tạo các trường form dựa trên loại form và category"""
        
        if category == "it":
            templates = self.it_form_templates
        elif category == "economics":
            templates = self.economics_form_templates
        else:  # marketing
            templates = self.marketing_form_templates
            
        template = templates.get(form_type, templates[list(templates.keys())[0]])
        
        fields = []
        for field_name in template["fields"]:
            field_type = self._get_field_type(field_name)
            validation_rules = self._get_validation_rules(field_name, field_type)
            
            field = {
                "name": field_name,
                "type": field_type,
                "label": self._generate_field_label(field_name),
                "required": random.choice([True, True, False]),  # 66% chance required
                "validation": validation_rules,
                "placeholder": self._generate_placeholder(field_name),
                "description": self._generate_field_description(field_name, category)
            }
            
            if field_type in ["select", "checkbox", "radio"]:
                field["options"] = self._generate_field_options(field_name, category)
            
            fields.append(field)
        
        return fields, template["complexity"], template["sections"]
    
    def _get_field_type(self, field_name):
        """Xác định loại field dựa trên tên"""
        field_type_mapping = {
            # Contact fields
            "email": "email", "phone": "tel", "password": "password",
            # Numeric fields
            "age": "number", "salary": "number", "budget": "number", "income": "number",
            "amount": "number", "cost": "number", "price": "number", "revenue": "number",
            # Rating fields
            "rating": "range", "satisfaction": "range", "score": "range", "likelihood": "range",
            # Date/Time fields
            "date": "date", "time": "time", "birthday": "date", "timeline": "date",
            "deadline": "date", "schedule": "datetime-local",
            # URL fields
            "url": "url", "website": "url", "github": "url", "linkedin": "url", "portfolio": "url",
            # Text area fields
            "comment": "textarea", "message": "textarea", "description": "textarea",
            "feedback": "textarea", "suggestion": "textarea", "requirement": "textarea",
            "objective": "textarea", "challenge": "textarea", "note": "textarea",
            # Multiple choice fields
            "skill": "checkbox", "preference": "checkbox", "interest": "checkbox",
            "feature": "checkbox", "service": "checkbox", "benefit": "checkbox",
            # Single choice fields
            "level": "select", "type": "select", "category": "select", "status": "select",
            "priority": "select", "frequency": "select", "size": "select", "industry": "select",
            # File upload fields
            "resume": "file", "cv": "file", "document": "file", "image": "file", "attachment": "file"
        }
        
        field_lower = field_name.lower()
        for keyword, field_type in field_type_mapping.items():
            if keyword in field_lower:
                return field_type
        
        return "text"  # default
    
    def _get_validation_rules(self, field_name, field_type):
        """Tạo validation rules cho field"""
        rules = {}
        
        if field_type == "email":
            rules["pattern"] = r'^[^\s@]+@[^\s@]+\.[^\s@]+$'
            rules["message"] = "Please enter a valid email address"
        elif field_type == "tel":
            rules["pattern"] = r'^\+?[\d\s\-\(\)]{10,}$'
            rules["message"] = "Please enter a valid phone number"
        elif field_type == "url":
            rules["pattern"] = r'^https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$'
            rules["message"] = "Please enter a valid URL"
        elif field_type == "number":
            if any(keyword in field_name.lower() for keyword in ["budget", "salary", "income", "amount", "cost"]):
                rules["min"] = 0
                rules["max"] = 10000000
            elif "age" in field_name.lower():
                rules["min"] = 18
                rules["max"] = 100
            elif any(keyword in field_name.lower() for keyword in ["rating", "score"]):
                rules["min"] = 1
                rules["max"] = 10
        elif field_type == "range":
            rules["min"] = 1
            rules["max"] = 10
            rules["step"] = 1
        elif field_type in ["text", "textarea"]:
            if field_type == "textarea":
                rules["minLength"] = random.randint(10, 50)
                rules["maxLength"] = random.randint(500, 2000)
            else:
                rules["minLength"] = random.randint(2, 10)
                rules["maxLength"] = random.randint(50, 200)
            
        return rules
    
    def _generate_field_label(self, field_name):
        """Generate human-readable field label"""
        # Handle common abbreviations and technical terms
        replacements = {
            "github": "GitHub",
            "linkedin": "LinkedIn", 
            "api": "API",
            "ui": "UI",
            "ux": "UX",
            "roi": "ROI",
            "kpi": "KPI",
            "seo": "SEO",
            "ppc": "PPC",
            "crm": "CRM",
            "erp": "ERP",
            "gdp": "GDP"
        }
        
        words = field_name.split("_")
        processed_words = []
        
        for word in words:
            if word.lower() in replacements:
                processed_words.append(replacements[word.lower()])
            else:
                processed_words.append(word.capitalize())
        
        return " ".join(processed_words)
    
    def _generate_placeholder(self, field_name):
        """Generate placeholder text for field"""
        placeholders = {
            "name": "Enter your full name",
            "email": "your.email@company.com",
            "phone": "+1 (555) 123-4567",
            "company": "Your company name",
            "position": "Your job title",
            "budget": "Enter budget amount (USD)",
            "timeline": "Select target date",
            "description": "Provide detailed description...",
            "objective": "Describe your objectives...",
            "challenge": "Describe current challenges...",
            "requirement": "List your requirements...",
            "experience": "Describe your experience...",
            "skill": "List relevant skills...",
            "goal": "Describe your goals...",
            "website": "https://www.example.com",
            "github": "https://github.com/username",
            "linkedin": "https://linkedin.com/in/username"
        }
        
        field_lower = field_name.lower()
        for keyword, placeholder in placeholders.items():
            if keyword in field_lower:
                return placeholder
        
        return f"Enter {field_name.replace('_', ' ')}"
    
    def _generate_field_description(self, field_name, category):
        """Generate field description based on category"""
        descriptions = {
            "it": {
                "programming_skills": "Select all programming languages you're proficient in",
                "experience_level": "Your years of professional software development experience",
                "certifications": "List any relevant technical certifications or training",
                "github_profile": "Link to your GitHub profile to showcase your code",
                "technical_requirements": "Specify technical constraints and requirements",
                "project_description": "Provide detailed description of your project scope",
                "budget_range": "Estimated budget for this software development project"
            },
            "economics": {
                "risk_tolerance": "How comfortable are you with investment volatility?",
                "investment_goals": "What are your primary investment objectives?",
                "time_horizon": "When do you expect to need access to these funds?",
                "annual_income": "Your total annual household income before taxes",
                "investment_experience": "Your level of experience with various investment types",
                "financial_objectives": "Specific financial goals you want to achieve",
                "market_analysis": "Your assessment of current market conditions"
            },
            "marketing": {
                "target_audience": "Describe your ideal customer profile in detail",
                "marketing_budget": "Total budget allocated for this marketing initiative",
                "success_metrics": "Key performance indicators you'll use to measure success",
                "campaign_objectives": "Primary goals you want to achieve with this campaign",
                "brand_guidelines": "Existing brand standards and guidelines to follow",
                "customer_segments": "Different groups within your target market",
                "competitive_analysis": "Analysis of your main competitors and their strategies"
            }
        }
        
        category_desc = descriptions.get(category, {})
        for key, desc in category_desc.items():
            if key in field_name.lower():
                return desc
        
        return f"Please provide information about {field_name.replace('_', ' ')}"
    
    def _generate_field_options(self, field_name, category):
        """Generate options for select/checkbox/radio fields"""
        options_map = {
            "it": {
                "programming_skills": [
                    "Python", "JavaScript", "Java", "C++", "C#", "TypeScript", "Go", "Rust",
                    "PHP", "Ruby", "Kotlin", "Swift", "Dart", "Scala", "R", "MATLAB"
                ],
                "experience_level": [
                    "Entry Level (0-1 years)", "Junior (1-3 years)", "Mid-Level (3-5 years)",
                    "Senior (5-8 years)", "Lead (8-12 years)", "Principal/Architect (12+ years)"
                ],
                "preferred_technologies": [
                    "React", "Angular", "Vue.js", "Node.js", "Django", "Flask", "Spring Boot",
                    "Docker", "Kubernetes", "AWS", "Azure", "GCP", "MongoDB", "PostgreSQL"
                ],
                "project_type": [
                    "Web Application", "Mobile App", "Desktop Application", "API/Backend",
                    "Database System", "DevOps/Infrastructure", "AI/ML Project", "Blockchain"
                ]
            },
            "economics": {
                "risk_tolerance": [
                    "Very Conservative", "Conservative", "Moderate", "Aggressive", "Very Aggressive"
                ],
                "investment_period": [
                    "Less than 1 year", "1-3 years", "3-5 years", "5-10 years", "More than 10 years"
                ],
                "investment_goals": [
                    "Capital Growth", "Regular Income", "Capital Preservation", "Tax Benefits",
                    "Retirement Planning", "Education Funding", "Emergency Fund"
                ],
                "income_range": [
                    "Under $25,000", "$25,000-$50,000", "$50,000-$100,000", "$100,000-$200,000", 
                    "$200,000-$500,000", "Over $500,000"
                ],
                "investment_experience": [
                    "No Experience", "Beginner (< 2 years)", "Intermediate (2-5 years)",
                    "Advanced (5-10 years)", "Expert (10+ years)"
                ]
            },
            "marketing": {
                "marketing_channels": [
                    "Social Media Marketing", "Email Marketing", "Content Marketing", 
                    "Search Engine Marketing", "Display Advertising", "Influencer Marketing",
                    "Public Relations", "Event Marketing", "Direct Mail", "Affiliate Marketing"
                ],
                "budget_range": [
                    "Under $1,000", "$1,000-$5,000", "$5,000-$25,000", 
                    "$25,000-$100,000", "$100,000-$500,000", "Over $500,000"
                ],
                "target_audience": [
                    "Small Businesses", "Enterprise", "Consumers (B2C)", "Students", 
                    "Professionals", "Seniors", "Millennials", "Gen Z"
                ],
                "campaign_objectives": [
                    "Brand Awareness", "Lead Generation", "Sales Conversion", 
                    "Customer Retention", "Market Penetration", "Product Launch"
                ]
            }
        }
        
        category_options = options_map.get(category, {})
        
        # Find matching options based on field name keywords
        for key, options in category_options.items():
            if any(keyword in field_name.lower() for keyword in key.split("_")):
                return options
        
        # Default options if no match found
        return ["Option A", "Option B", "Option C", "Option D", "Other"]

    def generate_single_record(self, record_id):
        """Tạo một bản ghi dữ liệu"""
        # Chọn category ngẫu nhiên với phân bố đều
        categories = ["it", "economics", "marketing"]
        category = random.choice(categories)
        
        # Chọn keyword ngẫu nhiên từ category
        if category == "it":
            keyword = random.choice(self.it_keywords)
            form_types = list(self.it_form_templates.keys())
        elif category == "economics":
            keyword = random.choice(self.economics_keywords)
            form_types = list(self.economics_form_templates.keys())
        else:  # marketing
            keyword = random.choice(self.marketing_keywords)
            form_types = list(self.marketing_form_templates.keys())
            
        form_type = random.choice(form_types)
        
        # Tạo form fields
        fields, complexity, sections = self.generate_form_fields(form_type, category)
        
        # Tạo metadata
        priority = random.choice(["Low", "Medium", "High"])
        target_audience = self._get_target_audience(category)
        use_case = self._get_use_case(form_type, category)
        
        # Tạo record với nhiều thông tin hơn
        record = {
            "record_id": record_id,
            "keyword": keyword,
            "category": category,
            "form_type": form_type,
            "complexity": complexity,
            "priority": priority,
            "target_audience": target_audience,
            "use_case": use_case,
            "num_fields": len(fields),
            "num_sections": len(sections),
            "required_fields_count": sum(1 for field in fields if field.get("required", False)),
            "sections": json.dumps(sections),
            "fields_json": json.dumps(fields),
            "estimated_completion_time": self._estimate_completion_time(len(fields), complexity),
            "created_date": self._random_date().strftime("%Y-%m-%d %H:%M:%S"),
            "tags": json.dumps(self._generate_tags(keyword, category, form_type)),
            "form_title": self._generate_form_title(keyword, form_type),
            "form_description": self._generate_form_description(keyword, category, form_type),
            "confidence_score": round(random.uniform(0.7, 0.99), 3),
            "language": "en",
            "region": random.choice(["US", "EU", "APAC", "GLOBAL"]),
            "industry_vertical": self._get_industry_vertical(category, keyword),
            "form_length": self._calculate_form_length(fields),
            "accessibility_score": round(random.uniform(0.8, 1.0), 2),
            "mobile_optimized": random.choice([True, True, True, False]),  # 75% mobile optimized
            "analytics_enabled": random.choice([True, True, False]),  # 66% analytics enabled
        }
        
        return record
    
    def _get_target_audience(self, category):
        """Get target audience based on category"""
        audiences = {
            "it": [
                "Software Developers", "IT Professionals", "Tech Startups", "Enterprise Teams",
                "DevOps Engineers", "Data Scientists", "System Administrators", "Technical Managers",
                "CTO/Technical Leaders", "Students/Learners", "Consultants", "Freelancers"
            ],
            "economics": [
                "Individual Investors", "Financial Advisors", "Business Owners", "Financial Analysts",
                "Portfolio Managers", "Investment Firms", "Corporate Finance Teams", "Economists",
                "Financial Consultants", "Retirement Planners", "Wealth Managers", "Entrepreneurs"
            ],
            "marketing": [
                "Marketing Teams", "Business Owners", "Marketing Agencies", "Entrepreneurs",
                "Digital Marketers", "Content Creators", "Brand Managers", "Marketing Directors",
                "Growth Hackers", "Social Media Managers", "PR Professionals", "Sales Teams"
            ]
        }
        return random.choice(audiences[category])
    
    def _get_use_case(self, form_type, category):
        """Generate detailed use case description"""
        use_cases = {
            "registration": f"User registration and onboarding for {category} platform or service",
            "survey": f"Collect feedback, insights, and preferences from {category} professionals",
            "application": f"Job application or service application process for {category} positions",
            "assessment": f"Evaluate skills, knowledge, and capabilities in {category} domain",
            "consultation": f"Initial consultation and requirements gathering for {category} services",
            "feedback": f"Product or service feedback collection from {category} users"
        }
        return use_cases.get(form_type, f"General {form_type} form for {category} use case")
    
    def _random_date(self):
        """Generate random date in the past 2 years"""
        start = datetime.now() - timedelta(days=730)
        end = datetime.now()
        return start + (end - start) * random.random()
    
    def _generate_tags(self, keyword, category, form_type):
        """Generate relevant tags for better categorization"""
        base_tags = [category, form_type]
        
        # Extract keywords from the main keyword
        keyword_words = re.findall(r'\b\w+\b', keyword.lower())
        keyword_tags = [word for word in keyword_words if len(word) > 3][:4]  # Take first 4 meaningful words
        
        # Add category-specific tags
        additional_tags = {
            "it": ["technology", "software", "digital", "programming", "development"],
            "economics": ["finance", "business", "investment", "analysis", "planning"],
            "marketing": ["promotion", "branding", "digital", "campaign", "strategy"]
        }
        
        # Add complexity and priority tags
        complexity_tag = ["simple", "moderate", "complex"][random.randint(0, 2)]
        priority_tag = ["low-priority", "medium-priority", "high-priority"][random.randint(0, 2)]
        
        all_tags = (base_tags + keyword_tags + 
                   random.sample(additional_tags[category], 3) + 
                   [complexity_tag, priority_tag])
        
        return list(set(all_tags))  # Remove duplicates
    
    def _generate_form_title(self, keyword, form_type):
        """Generate professional form title"""
        title_templates = {
            "registration": f"{keyword.title()} - Registration Form",
            "survey": f"{keyword.title()} Survey & Feedback",
            "application": f"{keyword.title()} - Application Form",
            "assessment": f"{keyword.title()} Skills Assessment",
            "consultation": f"{keyword.title()} - Consultation Request",
            "feedback": f"{keyword.title()} - Feedback Form"
        }
        return title_templates.get(form_type, f"{keyword.title()} - {form_type.replace('_', ' ').title()}")
    
    def _generate_form_description(self, keyword, category, form_type):
        """Generate detailed form description"""
        descriptions = {
            "registration": f"Professional registration form for {keyword} services in {category} domain. Complete this form to get started.",
            "survey": f"Help us improve our {keyword} services by sharing your experience and feedback in this comprehensive survey.",
            "application": f"Apply for {keyword} opportunities. This application form will help us understand your qualifications and interests.",
            "assessment": f"Evaluate your skills and knowledge in {keyword}. This assessment will help identify your current level and areas for improvement.",
            "consultation": f"Request a consultation for {keyword} services. Provide details about your requirements and we'll get back to you.",
            "feedback": f"Share your experience with our {keyword} services. Your feedback helps us continuously improve our offerings."
        }
        return descriptions.get(form_type, f"Professional {form_type} form for {keyword} in {category} domain.")
    
    def _estimate_completion_time(self, num_fields, complexity):
        """Estimate form completion time in minutes"""
        base_time_per_field = {
            "Simple": 0.5,
            "Moderate": 1.0,
            "Complex": 1.5
        }
        
        time_per_field = base_time_per_field.get(complexity, 1.0)
        estimated_time = max(2, int(num_fields * time_per_field + random.uniform(1, 3)))
        return min(estimated_time, 30)  # Cap at 30 minutes
    
    def _get_industry_vertical(self, category, keyword):
        """Determine industry vertical based on category and keyword"""
        verticals = {
            "it": {
                "cloud": "Cloud Computing", "security": "Cybersecurity", "ai": "Artificial Intelligence",
                "ml": "Machine Learning", "web": "Web Development", "mobile": "Mobile Development",
                "data": "Data Analytics", "devops": "DevOps", "blockchain": "Blockchain"
            },
            "economics": {
                "investment": "Investment Management", "banking": "Banking & Finance", "insurance": "Insurance",
                "trading": "Trading & Securities", "real estate": "Real Estate", "fintech": "Financial Technology",
                "crypto": "Cryptocurrency", "market": "Market Analysis"
            },
            "marketing": {
                "digital": "Digital Marketing", "social": "Social Media", "content": "Content Marketing",
                "email": "Email Marketing", "seo": "Search Marketing", "brand": "Brand Management",
                "event": "Event Marketing", "ecommerce": "E-commerce Marketing"
            }
        }
        
        keyword_lower = keyword.lower()
        for key, vertical in verticals.get(category, {}).items():
            if key in keyword_lower:
                return vertical
        
        # Default verticals by category
        default_verticals = {
            "it": "Information Technology",
            "economics": "Financial Services", 
            "marketing": "Marketing & Advertising"
        }
        return default_verticals.get(category, "General")
    
    def _calculate_form_length(self, fields):
        """Calculate form length category based on number of fields"""
        num_fields = len(fields)
        if num_fields <= 5:
            return "Short"
        elif num_fields <= 12:
            return "Medium"
        else:
            return "Long"

    def generate_dataset(self, total_records=500000000, batch_size=100000):
        """
        Generate large dataset in batches to manage memory
        Tạo 500 triệu bản ghi chia thành các batch để quản lý bộ nhớ
        """
        logger.info(f" Starting generation of {total_records:,} records in batches of {batch_size:,}")
        
        # Create output directory
        output_dir = "datasets"
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize counters and tracking
        generated_count = 0
        batch_number = 0
        start_time = time.time()
        
        # Progress bar for overall progress
        with tqdm(total=total_records, desc="Generating Dataset", unit="records", ncols=100) as pbar:
            
            while generated_count < total_records:
                batch_start_time = time.time()
                current_batch_size = min(batch_size, total_records - generated_count)
                
                logger.info(f"📊 Generating batch {batch_number + 1}, records {generated_count + 1:,} to {generated_count + current_batch_size:,}")
                
                # Generate batch of records
                batch_records = []
                for i in range(current_batch_size):
                    record_id = generated_count + i + 1
                    record = self.generate_single_record(record_id)
                    batch_records.append(record)
                
                # Convert to DataFrame
                batch_df = pd.DataFrame(batch_records)
                
                # Save batch to CSV
                batch_filename = f"{output_dir}/batch_{batch_number + 1:04d}.csv"
                batch_df.to_csv(batch_filename, index=False)
                
                # Update counters
                generated_count += current_batch_size
                batch_number += 1
                
                # Update progress bar
                pbar.update(current_batch_size)
                
                # Log progress with performance metrics
                batch_time = time.time() - batch_start_time
                total_time = time.time() - start_time
                rate = generated_count / total_time if total_time > 0 else 0
                eta_seconds = (total_records - generated_count) / rate if rate > 0 else 0
                eta_hours = eta_seconds / 3600
                
                logger.info(f" Batch {batch_number} completed in {batch_time:.1f}s. "
                          f"Total: {generated_count:,}/{total_records:,} records. "
                          f"Rate: {rate:.0f} records/sec. ETA: {eta_hours:.1f} hours")
                
                # Save checkpoint every 50 batches
                if batch_number % 50 == 0:
                    self._save_checkpoint(generated_count, batch_number, total_records)
                
                # Memory cleanup
                del batch_records, batch_df
                gc.collect()
        
        total_time = time.time() - start_time
        logger.info(f" Dataset generation completed! Generated {generated_count:,} records "
                   f"in {batch_number} batches. Total time: {total_time/3600:.1f} hours")
        
        # Combine batches and create samples
        return self._finalize_dataset(output_dir, batch_number, total_records, total_time)
    
    def _save_checkpoint(self, generated_count, batch_number, total_records):
        """Save generation checkpoint"""
        checkpoint = {
            "generated_count": generated_count,
            "batch_number": batch_number,
            "total_records": total_records,
            "timestamp": datetime.now().isoformat(),
            "progress_percent": (generated_count / total_records) * 100,
            "estimated_completion": datetime.now() + timedelta(
                seconds=(total_records - generated_count) * 0.001  # rough estimate
            )
        }
        
        checkpoint_path = "datasets/checkpoint.json"
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2, default=str)
        
        logger.info(f"💾 Checkpoint saved: {generated_count:,}/{total_records:,} records "
                   f"({checkpoint['progress_percent']:.1f}%)")
    
    def _finalize_dataset(self, output_dir, batch_count, total_records, total_time):
        """Finalize dataset and create summary files"""
        logger.info(f"📋 Finalizing dataset with {batch_count} batch files...")
        
        # Create sample datasets for testing and development
        sample_sizes = [1000, 10000, 100000, 1000000]
        self._create_sample_datasets(output_dir, sample_sizes)
        
        # Create dataset manifest
        manifest = {
            "total_records": total_records,
            "batch_count": batch_count,
            "batch_size": 100000,
            "generation_time_hours": total_time / 3600,
            "created_at": datetime.now().isoformat(),
            "categories": ["it", "economics", "marketing"],
            "form_types": {
                "it": list(self.it_form_templates.keys()),
                "economics": list(self.economics_form_templates.keys()),
                "marketing": list(self.marketing_form_templates.keys())
            },
            "file_structure": {
                "batch_files": f"batch_0001.csv to batch_{batch_count:04d}.csv",
                "sample_files": [f"sample_{size}.csv" for size in sample_sizes],
                "category_samples": ["sample_it_10k.csv", "sample_economics_10k.csv", "sample_marketing_10k.csv"]
            }
        }
        
        manifest_path = f"{output_dir}/dataset_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        
        # Create README file
        readme_content = self._generate_dataset_readme(manifest)
        with open(f"{output_dir}/README.md", "w") as f:
            f.write(readme_content)
        
        logger.info(f"📁 Dataset files created in {output_dir}/:")
        logger.info(f"   - {batch_count} batch files (batch_xxxx.csv)")
        logger.info(f"   - {len(sample_sizes)} sample datasets for testing")
        logger.info(f"   - 3 category-specific sample files")
        logger.info(f"   - Dataset manifest and documentation")
        
        return f"Successfully created dataset with {total_records:,} records in {total_time/3600:.1f} hours"
    
    def _create_sample_datasets(self, output_dir, sample_sizes):
        """Create sample datasets from the first few batches"""
        logger.info("📝 Creating sample datasets for testing and development...")
        
        # Load first few batches to create samples
        combined_df = pd.DataFrame()
        batch_files = sorted([f for f in os.listdir(output_dir) 
                            if f.startswith("batch_") and f.endswith(".csv")])
        
        # Load enough batches to create largest sample
        max_sample = max(sample_sizes)
        loaded_records = 0
        
        for batch_file in batch_files[:20]:  # Load first 20 batches (2M records)
            if loaded_records >= max_sample * 2:  # Load extra for variety
                break
                
            batch_path = os.path.join(output_dir, batch_file)
            try:
                batch_df = pd.read_csv(batch_path)
                combined_df = pd.concat([combined_df, batch_df], ignore_index=True)
                loaded_records += len(batch_df)
            except Exception as e:
                logger.warning(f"Error loading {batch_file}: {e}")
        
        # Create stratified sample files
        for size in sample_sizes:
            if len(combined_df) >= size:
                # Create stratified sample to ensure category balance
                sample_df = self._create_stratified_sample(combined_df, size)
                sample_path = f"{output_dir}/sample_{size}.csv"
                sample_df.to_csv(sample_path, index=False)
                logger.info(f"   ✓ Created sample dataset: {sample_path} ({len(sample_df):,} records)")
        
        # Create category-specific samples
        for category in ["it", "economics", "marketing"]:
            category_df = combined_df[combined_df['category'] == category]
            if len(category_df) >= 10000:
                category_sample = category_df.sample(n=10000, random_state=42)
                category_path = f"{output_dir}/sample_{category}_10k.csv"
                category_sample.to_csv(category_path, index=False)
                logger.info(f"   ✓ Created {category} sample: {category_path}")
    
    def _create_stratified_sample(self, df, sample_size):
        """Create stratified sample maintaining category proportions"""
        # Calculate samples per category
        category_counts = df['category'].value_counts()
        total_records = len(df)
        
        stratified_samples = []
        
        for category, count in category_counts.items():
            proportion = count / total_records
            category_sample_size = int(sample_size * proportion)
            
            if category_sample_size > 0:
                category_df = df[df['category'] == category]
                if len(category_df) >= category_sample_size:
                    sample = category_df.sample(n=category_sample_size, random_state=42)
                    stratified_samples.append(sample)
        
        if stratified_samples:
            return pd.concat(stratified_samples, ignore_index=True).sample(frac=1, random_state=42)
        else:
            return df.sample(n=min(sample_size, len(df)), random_state=42)
    
    def _generate_dataset_readme(self, manifest):
        """Generate comprehensive README for the dataset"""
        return f"""# Form Agent AI Dataset

## Overview
This dataset contains {manifest['total_records']:,} synthetic form records for training AI models in form generation and classification.

## Generation Details
- **Created**: {manifest['created_at']}
- **Generation Time**: {manifest['generation_time_hours']:.1f} hours
- **Categories**: IT, Economics, Marketing
- **Total Batches**: {manifest['batch_count']}

## File Structure
```
datasets/
├── batch_0001.csv to batch_{manifest['batch_count']:04d}.csv  # Main dataset batches
├── sample_1000.csv                                            # Small test sample
├── sample_10000.csv                                           # Medium test sample  
├── sample_100000.csv                                          # Large test sample
├── sample_1000000.csv                                         # XL test sample
├── sample_it_10k.csv                                          # IT category sample
├── sample_economics_10k.csv                                   # Economics category sample
├── sample_marketing_10k.csv                                   # Marketing category sample
├── dataset_manifest.json                                      # Dataset metadata
├── checkpoint.json                                             # Generation progress
└── README.md                                                   # This file
```

## Data Schema
Each record contains the following fields:
- `record_id`: Unique identifier
- `keyword`: Input keyword for form generation
- `category`: Domain category (it/economics/marketing)
- `form_type`: Type of form (registration/survey/application/etc.)
- `complexity`: Form complexity (Simple/Moderate/Complex)
- `fields_json`: Complete form field definitions
- `estimated_completion_time`: Time to complete form (minutes)
- `target_audience`: Intended user group
- And many more metadata fields...

## Usage
1. **Training**: Use batch files for training large models
2. **Testing**: Use sample files for development and testing
3. **Category-specific**: Use category samples for specialized models

## Categories Distribution
- **IT**: Software development, cloud computing, AI/ML, cybersecurity
- **Economics**: Finance, investment, market analysis, business planning  
- **Marketing**: Digital marketing, campaigns, brand management, analytics

## Form Types per Category
{json.dumps(manifest['form_types'], indent=2)}

## Loading Data
```python
import pandas as pd

# Load a sample for testing
df = pd.read_csv('datasets/sample_10000.csv')

# Load a specific batch
batch_df = pd.read_csv('datasets/batch_0001.csv')

# Load category-specific data
it_df = pd.read_csv('datasets/sample_it_10k.csv')
```

## Notes
- All data is synthetically generated
- No personally identifiable information
- Balanced across categories and form types
- Optimized for form generation AI training
"""


def main():
    """Main function để chạy dataset generation"""
    print("🤖 Form Agent AI - Dataset Generator")
    print("📊 Generating 500,000,000 records for IT, Economics, and Marketing")
    print("⏰ This will take several hours to complete...")
    
    # Initialize generator
    logger.info("Initializing dataset generator...")
    generator = DatasetGenerator()
    
    # Generate dataset
    try:
        result = generator.generate_dataset(
            total_records=500000000,  # 500 million records
            batch_size=100000         # 100k records per batch (5000 batches total)
        )
        
        logger.info("🎉 Dataset generation completed successfully!")
        logger.info(f"📋 Result: {result}")
        
    except KeyboardInterrupt:
        logger.info("⚠️  Generation interrupted by user")
        logger.info("💾 Partial dataset and checkpoints are saved in datasets/ folder")
        
    except Exception as e:
        logger.error(f"❌ Error during generation: {e}")
        logger.info("💾 Any partial progress is saved in datasets/ folder")
    
    finally:
        # Show final statistics regardless of completion status
        output_dir = "datasets"
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            batch_files = [f for f in files if f.startswith("batch_")]
            sample_files = [f for f in files if f.startswith("sample_")]
            
            print(f"\n📊 Final Statistics:")
            print(f"   📁 Output directory: {os.path.abspath(output_dir)}")
            print(f"   📄 Batch files: {len(batch_files)}")
            print(f"   🧪 Sample files: {len(sample_files)}")
            
            # Estimate total records generated
            if batch_files:
                try:
                    # Count records in a sample batch to estimate total
                    sample_batch = os.path.join(output_dir, batch_files[0])
                    sample_df = pd.read_csv(sample_batch)
                    estimated_total = len(sample_df) * len(batch_files)
                    print(f"   📊 Estimated records generated: {estimated_total:,}")
                except:
                    print(f"   📊 Records: Could not estimate total")
            
            # Show sample file info
            print(f"\n🧪 Sample Files:")
            for sample_file in sorted(sample_files):
                if sample_file.endswith('.csv'):
                    sample_path = os.path.join(output_dir, sample_file)
                    try:
                        sample_df = pd.read_csv(sample_path)
                        print(f"   📄 {sample_file}: {len(sample_df):,} records")
                    except:
                        print(f"   📄 {sample_file}: Error reading file")

if __name__ == "__main__":
    main()
