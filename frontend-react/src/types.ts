// API Types
export interface Question {
  question: string;
  category: string;
  confidence: number;
  method: string;
  source_keyword: string;
  similarity_score?: number;
}

export interface QuestionGenerationRequest {
  keyword: string;
  num_questions: number;
  category?: string;
}

export interface QuestionGenerationResponse {
  questions: Question[];
  keyword: string;
  total_generated: number;
  execution_time: number;
}

export interface CategoryPredictionRequest {
  keyword: string;
}

export interface CategoryPredictionResponse {
  keyword: string;
  predicted_category: string;
  confidence: number;
  all_probabilities: Record<string, number>;
}

export interface ModelInfo {
  success: boolean;
  model_loaded: boolean;
  training_date: string;
  total_keywords: number;
  total_questions: number;
  categories: string[];
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  timestamp: string;
  model_path: string;
}

export interface ApiError {
  message: string;
  error?: string;
}