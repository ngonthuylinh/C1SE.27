import {
  QuestionGenerationRequest,
  QuestionGenerationResponse,
  CategoryPredictionRequest,
  CategoryPredictionResponse,
  BatchCategoryRequest,
  BatchCategoryResponse,
  ModelInfo,
  HealthResponse,
  ApiError
} from '../types/api';

/**
 * API Service for Form Agent AI Backend
 * Handles all communication with the Flask backend using fetch
 */
class QuestionService {
  private baseURL: string;

  constructor() {
    this.baseURL = 'http://127.0.0.1:8000';
  }

  /**
   * Generic fetch wrapper with error handling
   */
  private async apiRequest<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    
    const defaultOptions: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      console.log(`🚀 API Request: ${options.method || 'GET'} ${endpoint}`);
      
      const response = await fetch(url, defaultOptions);
      
      if (!response.ok) {
        let errorMessage = `HTTP Error ${response.status}`;
        
        try {
          const errorData = await response.json();
          errorMessage = errorData.message || errorMessage;
        } catch {
          errorMessage = await response.text() || errorMessage;
        }
        
        throw new Error(errorMessage);
      }

      const data = await response.json();
      console.log(`✅ API Response: ${response.status} ${endpoint}`);
      
      return data;
    } catch (error: any) {
      console.error('❌ API Error:', error);
      
      const apiError: ApiError = {
        message: error.message || 'Network error occurred',
        code: error.status || 500,
        details: error
      };
      
      throw apiError;
    }
  }

  /**
   * Check backend health and model status
   */
  async checkHealth(): Promise<HealthResponse> {
    return this.apiRequest<HealthResponse>('/api/health');
  }

  /**
   * Get model information and statistics
   */
  async getModelInfo(): Promise<ModelInfo> {
    return this.apiRequest<ModelInfo>('/api/model/info');
  }

  /**
   * Generate questions from keyword
   */
  async generateQuestions(request: QuestionGenerationRequest): Promise<QuestionGenerationResponse> {
    // Validate input
    if (!request.keyword?.trim()) {
      throw new Error('Keyword is required');
    }
    
    if (request.num_questions < 1 || request.num_questions > 20) {
      throw new Error('Number of questions must be between 1 and 20');
    }

    return this.apiRequest<QuestionGenerationResponse>('/api/questions/generate', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /**
   * Predict category for a keyword
   */
  async predictCategory(request: CategoryPredictionRequest): Promise<CategoryPredictionResponse> {
    if (!request.keyword?.trim()) {
      throw new Error('Keyword is required');
    }

    return this.apiRequest<CategoryPredictionResponse>('/api/predict/category', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /**
   * Predict categories for multiple keywords
   */
  async predictCategoriesBatch(request: BatchCategoryRequest): Promise<BatchCategoryResponse> {
    if (!request.keywords || request.keywords.length === 0) {
      throw new Error('Keywords array is required');
    }

    if (request.keywords.length > 50) {
      throw new Error('Maximum 50 keywords per batch');
    }

    return this.apiRequest<BatchCategoryResponse>('/api/predict/batch-category', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /**
   * Test connection to backend
   */
  async testConnection(): Promise<boolean> {
    try {
      await this.checkHealth();
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Get available categories from model
   */
  async getCategories(): Promise<string[]> {
    try {
      const modelInfo = await this.getModelInfo();
      return modelInfo.categories || [];
    } catch (error: any) {
      throw new Error(`Failed to get categories: ${error.message}`);
    }
  }

  /**
   * Get API base URL
   */
  getBaseURL(): string {
    return this.baseURL;
  }

  /**
   * Update API base URL
   */
  setBaseURL(url: string): void {
    this.baseURL = url;
  }
}

// Export singleton instance
export const questionService = new QuestionService();
export default questionService;