import axios from 'axios';
import {
  QuestionGenerationRequest,
  QuestionGenerationResponse,
  CategoryPredictionRequest,
  CategoryPredictionResponse,
  ModelInfo,
  HealthResponse,
  ApiError
} from './types';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8001';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    const apiError: ApiError = {
      message: error.response?.data?.message || error.message || 'An error occurred',
      error: error.response?.data?.error || 'Unknown error'
    };
    return Promise.reject(apiError);
  }
);

export const questionApi = {
  // Health check
  async checkHealth(): Promise<HealthResponse> {
    const response = await api.get<HealthResponse>('/api/health');
    return response.data;
  },

  // Get model information
  async getModelInfo(): Promise<ModelInfo> {
    const response = await api.get<ModelInfo>('/api/model/info');
    return response.data;
  },

  // Generate questions
  async generateQuestions(request: QuestionGenerationRequest): Promise<QuestionGenerationResponse> {
    const response = await api.post<QuestionGenerationResponse>('/api/questions/generate', request);
    return response.data;
  },

  // Predict category
  async predictCategory(request: CategoryPredictionRequest): Promise<CategoryPredictionResponse> {
    const response = await api.post<CategoryPredictionResponse>('/api/predict/category', request);
    return response.data;
  },

  // Get available categories
  async getCategories(): Promise<string[]> {
    const modelInfo = await this.getModelInfo();
    return modelInfo.categories || [];
  },

  // Test connection
  async testConnection(): Promise<boolean> {
    try {
      await this.checkHealth();
      return true;
    } catch {
      return false;
    }
  }
};

export default questionApi;