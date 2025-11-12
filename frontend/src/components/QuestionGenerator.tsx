import React, { useState, useEffect } from 'react';
import questionService from '../services/questionService';
import {
  QuestionGenerationRequest,
  GeneratedQuestion,
  ModelInfo,
  HealthResponse,
  Category
} from '../types/api';
import './QuestionGenerator.css';

interface QuestionGeneratorState {
  keyword: string;
  numQuestions: number;
  category: string;
  questions: GeneratedQuestion[];
  loading: boolean;
  error: string;
  modelInfo: ModelInfo | null;
  isConnected: boolean;
  executionTime: number;
}

const QuestionGenerator: React.FC = () => {
  const [state, setState] = useState<QuestionGeneratorState>({
    keyword: '',
    numQuestions: 5,
    category: '',
    questions: [],
    loading: false,
    error: '',
    modelInfo: null,
    isConnected: false,
    executionTime: 0
  });

  // Check connection and load model info on component mount
  useEffect(() => {
    checkConnection();
    loadModelInfo();
  }, []);

  const checkConnection = async () => {
    try {
      const health: HealthResponse = await questionService.checkHealth();
      setState(prev => ({
        ...prev,
        isConnected: health.model_loaded,
        error: health.model_loaded ? '' : 'Model chưa được load'
      }));
    } catch (error: any) {
      setState(prev => ({
        ...prev,
        isConnected: false,
        error: `Kết nối backend thất bại: ${error.message}`
      }));
    }
  };

  const loadModelInfo = async () => {
    try {
      const modelInfo = await questionService.getModelInfo();
      setState(prev => ({
        ...prev,
        modelInfo
      }));
    } catch (error: any) {
      console.error('Failed to load model info:', error);
    }
  };

  const handleInputChange = (field: keyof QuestionGeneratorState, value: string | number) => {
    setState(prev => ({
      ...prev,
      [field]: value,
      error: ''
    }));
  };

  const generateQuestions = async () => {
    if (!state.keyword.trim()) {
      setState(prev => ({
        ...prev,
        error: 'Vui lòng nhập keyword'
      }));
      return;
    }

    setState(prev => ({
      ...prev,
      loading: true,
      error: '',
      questions: []
    }));

    try {
      const request: QuestionGenerationRequest = {
        keyword: state.keyword.trim(),
        num_questions: state.numQuestions,
        ...(state.category && { category: state.category })
      };

      const startTime = performance.now();
      const response = await questionService.generateQuestions(request);
      const endTime = performance.now();

      setState(prev => ({
        ...prev,
        questions: response.questions,
        executionTime: Math.round(endTime - startTime),
        loading: false
      }));
    } catch (error: any) {
      setState(prev => ({
        ...prev,
        error: `Lỗi tạo câu hỏi: ${error.message}`,
        loading: false
      }));
    }
  };

  const renderConnectionStatus = () => (
    <div className={`connection-status ${state.isConnected ? 'connected' : 'disconnected'}`}>
      <div className="status-indicator">
        <span className={`status-dot ${state.isConnected ? 'green' : 'red'}`}></span>
        <span className="status-text">
          {state.isConnected ? 'Kết nối thành công' : 'Mất kết nối'}
        </span>
      </div>
      {state.modelInfo && (
        <div className="model-info">
          <span>📊 {state.modelInfo.total_questions?.toLocaleString()} câu hỏi</span>
          <span>🏷️ {state.modelInfo.categories?.length} categories</span>
          <span>📅 {new Date(state.modelInfo.training_date).toLocaleDateString('vi-VN')}</span>
        </div>
      )}
    </div>
  );

  const renderQuestionCard = (question: GeneratedQuestion, index: number) => (
    <div key={index} className="question-card">
      <div className="question-header">
        <span className="question-number">#{index + 1}</span>
        <span className={`category-badge ${question.category}`}>
          {question.category}
        </span>
        <span className="confidence">
          {Math.round(question.confidence * 100)}%
        </span>
      </div>
      <div className="question-text">
        {question.question}
      </div>
      <div className="question-meta">
        <span className="method">{question.method}</span>
        {question.source_keyword && (
          <span className="source-keyword">từ: "{question.source_keyword}"</span>
        )}
      </div>
    </div>
  );

  return (
    <div className="question-generator">
      <header className="header">
        <h1>🤖 Form Agent AI - Question Generator</h1>
        <p>Tạo câu hỏi thông minh từ keyword bằng AI</p>
        {renderConnectionStatus()}
      </header>

      <div className="generator-form">
        <div className="input-group">
          <label htmlFor="keyword">Keyword</label>
          <input
            id="keyword"
            type="text"
            value={state.keyword}
            onChange={(e) => handleInputChange('keyword', e.target.value)}
            placeholder="Nhập keyword để tạo câu hỏi..."
            disabled={state.loading || !state.isConnected}
          />
        </div>

        <div className="input-row">
          <div className="input-group">
            <label htmlFor="numQuestions">Số lượng câu hỏi</label>
            <select
              id="numQuestions"
              value={state.numQuestions}
              onChange={(e) => handleInputChange('numQuestions', parseInt(e.target.value))}
              disabled={state.loading || !state.isConnected}
            >
              {[1, 2, 3, 5, 7, 10, 15, 20].map(num => (
                <option key={num} value={num}>{num} câu hỏi</option>
              ))}
            </select>
          </div>

          <div className="input-group">
            <label htmlFor="category">Category (tùy chọn)</label>
            <select
              id="category"
              value={state.category}
              onChange={(e) => handleInputChange('category', e.target.value)}
              disabled={state.loading || !state.isConnected}
            >
              <option value="">Tự động phát hiện</option>
              {state.modelInfo?.categories?.map(cat => (
                <option key={cat} value={cat}>{cat}</option>
              ))}
            </select>
          </div>
        </div>

        {state.error && (
          <div className="error-message">
            ❌ {state.error}
          </div>
        )}

        <button
          className="generate-button"
          onClick={generateQuestions}
          disabled={state.loading || !state.isConnected || !state.keyword.trim()}
        >
          {state.loading ? (
            <>
              <span className="loading-spinner"></span>
              Đang tạo câu hỏi...
            </>
          ) : (
            <>
              ✨ Tạo câu hỏi
            </>
          )}
        </button>
      </div>

      {state.questions.length > 0 && (
        <div className="results-section">
          <div className="results-header">
            <h2>📝 Kết quả ({state.questions.length} câu hỏi)</h2>
            <div className="execution-info">
              <span>⚡ {state.executionTime}ms</span>
              <span>🎯 Keyword: "{state.keyword}"</span>
            </div>
          </div>
          
          <div className="questions-grid">
            {state.questions.map((question, index) => renderQuestionCard(question, index))}
          </div>
        </div>
      )}

      {!state.isConnected && (
        <div className="reconnect-section">
          <p>Mất kết nối với backend. Vui lòng kiểm tra:</p>
          <ul>
            <li>Backend server đang chạy ở http://127.0.0.1:8000</li>
            <li>Model đã được load thành công</li>
            <li>Không có lỗi CORS</li>
          </ul>
          <button className="retry-button" onClick={checkConnection}>
            🔄 Thử kết nối lại
          </button>
        </div>
      )}
    </div>
  );
};

export default QuestionGenerator;