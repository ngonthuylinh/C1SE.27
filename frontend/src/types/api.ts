// API Types for Form Agent AI Frontend

// Question Generation Types
export interface QuestionGenerationRequest {
  keyword: string;
  num_questions: number;
  category?: string;
}

export interface GeneratedQuestion {
  question: string;
  category: string;
  confidence: number;
  method: string;
  source_keyword: string;
  similarity_score?: number;
}

export interface QuestionGenerationResponse {
  questions: GeneratedQuestion[];
  keyword: string;
  total_generated: number;
  execution_time: number;
}

// Category Prediction Types
export interface CategoryPredictionRequest {
  keyword: string;
}

export interface CategoryPredictionResponse {
  keyword: string;
  predicted_category: string;
  confidence: number;
  all_probabilities: Record<string, number>;
}

// Batch Category Prediction Types
export interface BatchCategoryRequest {
  keywords: string[];
}

export interface BatchCategoryResponse {
  results: Record<string, CategoryPredictionResponse>;
  total_keywords: number;
}

// Model Info Types
export interface ModelInfo {
  model_loaded: boolean;
  training_date: string;
  total_keywords: number;
  total_questions: number;
  categories: string[];
  model_accuracy?: number;
}

// Health Check Types
export interface HealthResponse {
  status: string;
  message: string;
  model_loaded: boolean;
  timestamp: string;
}

// API Response Wrapper
export interface ApiResponse<T> {
  data: T;
  success: boolean;
  message?: string;
}

// Error Types
export interface ApiError {
  message: string;
  code?: number;
  details?: any;
}

// Categories enum
export enum Category {
  ECONOMICS = 'economics',
  IT = 'it', 
  MARKETING = 'marketing',
  NAN = 'nan'
}

// Question Methods enum
export enum QuestionMethod {
  DIRECT_MATCH = 'direct_match',
  SIMILAR_KEYWORD = 'similar_keyword', 
  CATEGORY_ADAPTATION = 'category_adaptation',
  CONTEXTUAL_GENERATION = 'contextual_generation'
}
