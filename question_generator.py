#!/usr/bin/env python3
"""
Question Dataset Generator
Tạo 1 triệu câu hỏi dựa trên keywords của 3 lĩnh vực: IT, Economics, Marketing
Mục tiêu: Train model để từ keyword → generate câu hỏi phù hợp
"""

import pandas as pd
import numpy as np
import random
import os
import json
import re
from datetime import datetime
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuestionDatasetGenerator:
    """Tạo dataset câu hỏi từ keywords"""
    
    def __init__(self):
        print("🤖 Initializing Question Dataset Generator...")
        self.load_existing_keywords()
        self.setup_question_templates()
        print("✅ Question Generator ready!")
        
    def load_existing_keywords(self):
        """Load keywords từ dataset có sẵn"""
        print("📊 Loading existing keywords from dataset...")
        
        # Load từ batch đầu tiên để lấy keywords
        try:
            df = pd.read_csv("datasets/batch_0001.csv")
            
            # Lấy unique keywords theo category
            self.it_keywords = df[df['category'] == 'it']['keyword'].unique().tolist()
            self.economics_keywords = df[df['category'] == 'economics']['keyword'].unique().tolist()
            self.marketing_keywords = df[df['category'] == 'marketing']['keyword'].unique().tolist()
            
            print(f"   IT Keywords: {len(self.it_keywords)}")
            print(f"   Economics Keywords: {len(self.economics_keywords)}")
            print(f"   Marketing Keywords: {len(self.marketing_keywords)}")
            
        except Exception as e:
            print(f"⚠️ Could not load from existing dataset: {e}")
            print("📝 Using predefined keywords...")
            self.setup_fallback_keywords()
    
    def setup_fallback_keywords(self):
        """Backup keywords nếu không load được từ dataset"""
        self.it_keywords = [
            "python programming", "web development", "cloud computing", "machine learning",
            "cybersecurity", "database design", "api development", "mobile app development",
            "artificial intelligence", "data science", "software engineering", "devops"
        ]
        
        self.economics_keywords = [
            "investment planning", "financial analysis", "market research", "portfolio management",
            "risk assessment", "business strategy", "economic forecasting", "budget planning",
            "cryptocurrency", "stock market", "insurance", "banking"
        ]
        
        self.marketing_keywords = [
            "digital marketing", "social media", "content marketing", "brand management",
            "seo optimization", "email marketing", "advertising campaigns", "market analysis",
            "customer retention", "lead generation", "viral marketing", "analytics"
        ]
    
    def setup_question_templates(self):
        """Setup question templates cho từng lĩnh vực"""
        
        # IT Question Templates
        self.it_question_templates = [
            # Technical Implementation
            "How do you implement {keyword} in a production environment?",
            "What are the best practices for {keyword} development?",
            "How can {keyword} improve system performance and scalability?",
            "What security considerations should be taken when using {keyword}?",
            "How do you troubleshoot common issues with {keyword}?",
            
            # Architecture & Design
            "What is the recommended architecture for {keyword} systems?",
            "How do you design a scalable {keyword} solution?",
            "What are the key components of a {keyword} implementation?",
            "How do you integrate {keyword} with existing systems?",
            "What design patterns work best with {keyword}?",
            
            # Tools & Technologies
            "What tools are essential for {keyword} development?",
            "Which frameworks support {keyword} implementation?",
            "How do you choose the right technology stack for {keyword}?",
            "What are the latest trends in {keyword} technology?",
            "How do you optimize {keyword} for different platforms?",
            
            # Learning & Career
            "What skills are needed to master {keyword}?",
            "How do you get started with {keyword} as a beginner?",
            "What certifications are valuable for {keyword} professionals?",
            "What career opportunities exist in {keyword}?",
            "How do you stay updated with {keyword} developments?",
            
            # Business Impact
            "How does {keyword} impact business operations?",
            "What ROI can be expected from {keyword} implementation?",
            "How do you measure the success of {keyword} projects?",
            "What are the cost considerations for {keyword}?",
            "How do you convince stakeholders to invest in {keyword}?"
        ]
        
        # Economics Question Templates
        self.economics_question_templates = [
            # Investment & Finance
            "What factors should be considered when evaluating {keyword}?",
            "How do economic conditions affect {keyword} performance?",
            "What are the risks associated with {keyword}?",
            "How do you diversify your portfolio using {keyword}?",
            "What is the optimal allocation for {keyword} in a portfolio?",
            
            # Market Analysis
            "How do you analyze market trends in {keyword}?",
            "What indicators predict {keyword} market movements?",
            "How does global economics impact {keyword}?",
            "What are the seasonal patterns in {keyword}?",
            "How do you identify opportunities in {keyword} markets?",
            
            # Strategy & Planning
            "What strategies work best for {keyword} investment?",
            "How do you create a long-term plan for {keyword}?",
            "What timeline is realistic for {keyword} returns?",
            "How do you adjust your {keyword} strategy during volatility?",
            "What exit strategies should be considered for {keyword}?",
            
            # Risk Management
            "How do you assess and manage {keyword} risks?",
            "What insurance options are available for {keyword}?",
            "How do you hedge against {keyword} market downturns?",
            "What are the regulatory considerations for {keyword}?",
            "How do you protect your {keyword} investments?",
            
            # Performance & Analytics
            "How do you measure {keyword} performance?",
            "What metrics are most important for {keyword}?",
            "How do you benchmark {keyword} against alternatives?",
            "What tools help analyze {keyword} data?",
            "How do you report on {keyword} performance to stakeholders?"
        ]
        
        # Marketing Question Templates
        self.marketing_question_templates = [
            # Strategy & Planning
            "How do you develop an effective {keyword} strategy?",
            "What budget should be allocated to {keyword}?",
            "How do you measure the ROI of {keyword} campaigns?",
            "What KPIs are most important for {keyword}?",
            "How do you create a {keyword} roadmap?",
            
            # Target Audience
            "How do you identify the right audience for {keyword}?",
            "What demographics respond best to {keyword}?",
            "How do you personalize {keyword} for different segments?",
            "What messaging works best in {keyword} campaigns?",
            "How do you reach new customers through {keyword}?",
            
            # Channels & Tactics
            "What channels are most effective for {keyword}?",
            "How do you optimize {keyword} for mobile users?",
            "What content types work best for {keyword}?",
            "How do you automate {keyword} processes?",
            "What timing strategies work for {keyword}?",
            
            # Analytics & Optimization
            "How do you track and analyze {keyword} performance?",
            "What A/B tests should you run for {keyword}?",
            "How do you improve {keyword} conversion rates?",
            "What tools are essential for {keyword} analytics?",
            "How do you attribute success to {keyword} efforts?",
            
            # Trends & Innovation
            "What are the latest trends in {keyword}?",
            "How is AI changing {keyword} approaches?",
            "What emerging platforms support {keyword}?",
            "How do you innovate within {keyword}?",
            "What does the future hold for {keyword}?"
        ]
        
        # Question Types (dạng câu hỏi)
        self.question_types = [
            "how_to",        # Hướng dẫn
            "best_practice", # Thực tiễn tốt nhất
            "comparison",    # So sánh
            "troubleshooting", # Khắc phục sự cố
            "strategy",      # Chiến lược
            "analysis",      # Phân tích
            "implementation", # Triển khai
            "optimization",  # Tối ưu hóa
            "planning",      # Lập kế hoạch
            "evaluation"     # Đánh giá
        ]
        
        # Difficulty Levels
        self.difficulty_levels = ["Beginner", "Intermediate", "Advanced", "Expert"]
        
    def generate_single_question(self, record_id):
        """Tạo một câu hỏi từ keyword"""
        
        # Chọn category ngẫu nhiên
        category = random.choice(["it", "economics", "marketing"])
        
        # Chọn keyword từ category
        if category == "it":
            keyword = random.choice(self.it_keywords)
            templates = self.it_question_templates
        elif category == "economics":
            keyword = random.choice(self.economics_keywords)
            templates = self.economics_question_templates
        else:  # marketing
            keyword = random.choice(self.marketing_keywords)
            templates = self.marketing_question_templates
        
        # Chọn template và tạo câu hỏi
        template = random.choice(templates)
        question = template.format(keyword=keyword)
        
        # Tạo metadata
        question_type = random.choice(self.question_types)
        difficulty = random.choice(self.difficulty_levels)
        
        # Tạo answer outline (khung trả lời)
        answer_outline = self.generate_answer_outline(question, category, keyword)
        
        # Tạo tags
        tags = self.generate_question_tags(keyword, category, question_type)
        
        record = {
            "record_id": record_id,
            "keyword": keyword,
            "category": category,
            "question": question,
            "question_type": question_type,
            "difficulty_level": difficulty,
            "answer_outline": json.dumps(answer_outline),
            "tags": json.dumps(tags),
            "estimated_read_time": self.estimate_read_time(question, answer_outline),
            "target_audience": self.get_target_audience(category, difficulty),
            "related_topics": json.dumps(self.get_related_topics(keyword, category)),
            "question_length": len(question),
            "word_count": len(question.split()),
            "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "language": "en",
            "source_template": template,
            "engagement_score": round(random.uniform(0.6, 0.95), 2),
            "complexity_score": self.calculate_complexity_score(question, difficulty),
            "popularity_score": round(random.uniform(0.3, 0.9), 2)
        }
        
        return record
    
    def generate_answer_outline(self, question, category, keyword):
        """Tạo khung trả lời cho câu hỏi"""
        
        # Outline templates theo loại câu hỏi
        if "how do you" in question.lower() or "how to" in question.lower():
            outline = [
                f"Overview of {keyword}",
                "Step-by-step implementation",
                "Best practices and tips", 
                "Common challenges and solutions",
                "Tools and resources needed",
                "Conclusion and next steps"
            ]
        elif "what are" in question.lower():
            outline = [
                f"Introduction to {keyword}",
                "Key concepts and definitions",
                "Main categories or types",
                "Examples and case studies",
                "Benefits and advantages",
                "Summary and recommendations"
            ]
        elif "compare" in question.lower() or "vs" in question.lower():
            outline = [
                f"Overview of {keyword}",
                "Comparison criteria",
                "Advantages and disadvantages",
                "Use case scenarios", 
                "Performance analysis",
                "Recommendation and conclusion"
            ]
        else:
            outline = [
                f"Introduction to {keyword}",
                "Key points and analysis",
                "Practical examples",
                "Expert insights",
                "Action items",
                "Conclusion"
            ]
            
        return outline
    
    def generate_question_tags(self, keyword, category, question_type):
        """Tạo tags cho câu hỏi"""
        
        # Base tags
        tags = [category, question_type]
        
        # Keyword-based tags
        keyword_words = keyword.lower().split()
        tags.extend(keyword_words[:3])  # Lấy 3 từ đầu
        
        # Category-specific tags
        category_tags = {
            "it": ["technology", "programming", "software", "development"],
            "economics": ["finance", "investment", "business", "market"],
            "marketing": ["advertising", "promotion", "branding", "digital"]
        }
        
        tags.extend(random.sample(category_tags[category], 2))
        
        return list(set(tags))  # Remove duplicates
    
    def estimate_read_time(self, question, answer_outline):
        """Ước tính thời gian đọc (phút)"""
        total_words = len(question.split()) + sum(len(point.split()) for point in answer_outline) * 50
        # Tính trung bình 200 từ/phút
        return max(1, round(total_words / 200))
    
    def get_target_audience(self, category, difficulty):
        """Xác định target audience"""
        audiences = {
            "it": {
                "Beginner": "Students & Entry-level Developers",
                "Intermediate": "Junior to Mid-level Developers", 
                "Advanced": "Senior Developers & Tech Leads",
                "Expert": "Architects & Technical Directors"
            },
            "economics": {
                "Beginner": "Students & New Investors",
                "Intermediate": "Individual Investors & Analysts",
                "Advanced": "Professional Investors & Advisors", 
                "Expert": "Portfolio Managers & Financial Experts"
            },
            "marketing": {
                "Beginner": "Marketing Students & Interns",
                "Intermediate": "Marketing Coordinators & Specialists",
                "Advanced": "Marketing Managers & Strategists",
                "Expert": "Marketing Directors & CMOs"
            }
        }
        
        return audiences[category][difficulty]
    
    def get_related_topics(self, keyword, category):
        """Tạo danh sách topics liên quan"""
        
        # Lấy keywords liên quan từ cùng category
        if category == "it":
            all_keywords = self.it_keywords
        elif category == "economics":
            all_keywords = self.economics_keywords
        else:
            all_keywords = self.marketing_keywords
            
        # Chọn 3-5 keywords liên quan (loại trừ keyword hiện tại)
        related = [kw for kw in all_keywords if kw != keyword]
        return random.sample(related, min(4, len(related)))
    
    def calculate_complexity_score(self, question, difficulty):
        """Tính điểm phức tạp từ 0-1"""
        base_scores = {
            "Beginner": 0.2,
            "Intermediate": 0.5,
            "Advanced": 0.7,
            "Expert": 0.9
        }
        
        base = base_scores[difficulty]
        
        # Điều chỉnh dựa trên độ dài câu hỏi
        length_factor = min(len(question.split()) / 20, 0.2)
        
        return min(base + length_factor + random.uniform(-0.1, 0.1), 1.0)

    def generate_dataset(self, total_records=1000000, batch_size=50000):
        """Tạo dataset câu hỏi lớn"""
        
        print(f"🚀 Starting generation of {total_records:,} questions")
        print(f"📊 Batch size: {batch_size:,} questions per batch")
        
        # Tạo thư mục output
        output_dir = "question_datasets"
        os.makedirs(output_dir, exist_ok=True)
        
        generated_count = 0
        batch_number = 0
        start_time = datetime.now()
        
        # Progress bar
        pbar = tqdm(total=total_records, desc="Generating Questions", unit="questions")
        
        try:
            while generated_count < total_records:
                current_batch_size = min(batch_size, total_records - generated_count)
                
                print(f"\n📋 Generating question batch {batch_number + 1}")
                print(f"   Questions: {generated_count + 1:,} to {generated_count + current_batch_size:,}")
                
                # Tạo batch questions
                batch_records = []
                for i in range(current_batch_size):
                    record_id = generated_count + i + 1
                    question_record = self.generate_single_question(record_id)
                    batch_records.append(question_record)
                
                # Lưu batch
                batch_df = pd.DataFrame(batch_records)
                batch_file = f"{output_dir}/question_batch_{batch_number + 1:04d}.csv"
                batch_df.to_csv(batch_file, index=False)
                
                # Cập nhật progress
                generated_count += current_batch_size
                batch_number += 1
                pbar.update(current_batch_size)
                
                # Stats
                elapsed = datetime.now() - start_time
                rate = generated_count / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
                
                print(f"   ✅ Batch completed")
                print(f"   📊 Total: {generated_count:,}/{total_records:,}")
                print(f"   ⚡ Rate: {rate:.0f} questions/sec")
                
                # Checkpoint mỗi 5 batches
                if batch_number % 5 == 0:
                    self._save_checkpoint(generated_count, batch_number, total_records, output_dir)
                
                # Cleanup memory
                del batch_records, batch_df
                
        finally:
            pbar.close()
            
        # Tạo sample files và summary
        self._create_samples_and_summary(output_dir, batch_number, generated_count)
        
        elapsed = datetime.now() - start_time
        print(f"\n🎉 Question generation completed!")
        print(f"📊 Generated {generated_count:,} questions in {batch_number} batches")
        print(f"⏱️  Total time: {elapsed}")
        print(f"📁 Files saved in: {output_dir}/")
        
        return f"Generated {generated_count:,} questions in {batch_number} batches"
    
    def _save_checkpoint(self, generated_count, batch_number, total_records, output_dir):
        """Lưu checkpoint"""
        checkpoint = {
            "generated_count": generated_count,
            "batch_number": batch_number,
            "total_records": total_records,
            "timestamp": datetime.now().isoformat(),
            "progress_percent": (generated_count / total_records) * 100
        }
        
        checkpoint_file = f"{output_dir}/checkpoint.json"
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint, f, indent=2)
            
        print(f"   💾 Checkpoint saved: {generated_count:,} questions ({checkpoint['progress_percent']:.1f}%)")
    
    def _create_samples_and_summary(self, output_dir, batch_count, total_generated):
        """Tạo sample files và dataset summary"""
        print(f"\n📝 Creating samples and summary...")
        
        # Load batch đầu tiên để tạo samples
        first_batch = f"{output_dir}/question_batch_0001.csv"
        if os.path.exists(first_batch):
            df = pd.read_csv(first_batch)
            
            # Tạo samples với kích thước khác nhau
            sample_sizes = [100, 1000, 5000, 10000]
            
            for size in sample_sizes:
                if len(df) >= size:
                    sample = df.head(size)
                    sample_file = f"{output_dir}/question_sample_{size}.csv"
                    sample.to_csv(sample_file, index=False)
                    print(f"   ✅ Created: {sample_file}")
            
            # Tạo samples theo category
            for category in ["it", "economics", "marketing"]:
                category_data = df[df['category'] == category]
                if len(category_data) > 0:
                    sample_size = min(2000, len(category_data))
                    sample = category_data.head(sample_size)
                    category_file = f"{output_dir}/question_sample_{category}_{sample_size}.csv"
                    sample.to_csv(category_file, index=False)
                    print(f"   ✅ Created: {category_file}")
                    
            # Tạo summary statistics
            summary = self._generate_dataset_summary(df, batch_count, total_generated)
            summary_file = f"{output_dir}/dataset_summary.json"
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"   ✅ Created: {summary_file}")
    
    def _generate_dataset_summary(self, sample_df, batch_count, total_generated):
        """Tạo summary statistics"""
        
        summary = {
            "dataset_info": {
                "total_questions": total_generated,
                "total_batches": batch_count,
                "creation_date": datetime.now().isoformat(),
                "purpose": "Question generation training dataset"
            },
            "category_distribution": sample_df['category'].value_counts().to_dict(),
            "difficulty_distribution": sample_df['difficulty_level'].value_counts().to_dict(),
            "question_type_distribution": sample_df['question_type'].value_counts().to_dict(),
            "statistics": {
                "avg_question_length": float(sample_df['question_length'].mean()),
                "avg_word_count": float(sample_df['word_count'].mean()),
                "avg_read_time": float(sample_df['estimated_read_time'].mean()),
                "avg_complexity_score": float(sample_df['complexity_score'].mean()),
                "avg_engagement_score": float(sample_df['engagement_score'].mean())
            },
            "sample_questions": {
                category: sample_df[sample_df['category'] == category]['question'].head(3).tolist()
                for category in ["it", "economics", "marketing"]
            }
        }
        
        return summary


def main():
    """Main function"""
    print("🤖 Question Dataset Generator")
    print("📊 Creating 1 million training questions from keywords...")
    
    try:
        # Initialize generator
        generator = QuestionDatasetGenerator()
        
        # Generate dataset
        result = generator.generate_dataset(
            total_records=1000000,  # 1M questions
            batch_size=50000        # 50k per batch
        )
        
        print(f"🎉 Success: {result}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Generation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
