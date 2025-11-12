const express = require('express');
const router = express.Router();
const { v4: uuidv4 } = require('uuid');

/**
 * @route   POST /api/predict/category
 * @desc    Predict category for a keyword
 * @access  Public
 */
router.post('/category', async (req, res) => {
    const requestId = uuidv4();
    const startTime = Date.now();
    
    try {
        // Validate request body
        if (!req.body) {
            return res.status(400).json({
                success: false,
                error: 'Request body is required',
                timestamp: req.timestamp
            });
        }
        
        const { keyword } = req.body;
        
        // Validate keyword
        if (!keyword || typeof keyword !== 'string' || !keyword.trim()) {
            return res.status(400).json({
                success: false,
                error: 'Keyword is required and must be a non-empty string',
                timestamp: req.timestamp
            });
        }
        
        // Check if model is ready
        if (!req.pythonBridge.isReady()) {
            return res.status(503).json({
                success: false,
                error: 'Model is not loaded. Please check server status.',
                timestamp: req.timestamp
            });
        }
        
        console.log(`🎯 [${requestId}] Predicting category for keyword: "${keyword.trim()}"`);
        
        // Predict category
        const result = await req.pythonBridge.predictCategory(keyword.trim());
        
        const executionTime = Date.now() - startTime;
        
        if (result.success) {
            console.log(`✅ [${requestId}] Predicted category "${result.predicted_category}" with confidence ${result.confidence} in ${executionTime}ms`);
            
            const response = {
                success: true,
                keyword: keyword.trim(),
                predicted_category: result.predicted_category,
                confidence: result.confidence,
                all_probabilities: result.all_probabilities || {},
                execution_time: executionTime,
                request_id: requestId,
                timestamp: req.timestamp
            };
            
            res.json(response);
        } else {
            console.error(`❌ [${requestId}] Category prediction failed:`, result.error);
            
            res.status(500).json({
                success: false,
                error: result.error || 'Failed to predict category',
                keyword: keyword.trim(),
                execution_time: executionTime,
                request_id: requestId,
                timestamp: req.timestamp
            });
        }
        
    } catch (error) {
        const executionTime = Date.now() - startTime;
        console.error(`❌ [${requestId}] Category prediction error:`, error);
        
        res.status(500).json({
            success: false,
            error: error.message || 'Internal server error',
            execution_time: executionTime,
            request_id: requestId,
            timestamp: req.timestamp
        });
    }
});

/**
 * @route   POST /api/predict/batch
 * @desc    Predict categories for multiple keywords in batch
 * @access  Public
 */
router.post('/batch', async (req, res) => {
    const requestId = uuidv4();
    const startTime = Date.now();
    
    try {
        const { keywords } = req.body;
        
        // Validate keywords array
        if (!Array.isArray(keywords) || keywords.length === 0) {
            return res.status(400).json({
                success: false,
                error: 'Keywords array is required',
                timestamp: req.timestamp
            });
        }
        
        if (keywords.length > 50) {
            return res.status(400).json({
                success: false,
                error: 'Maximum 50 keywords allowed per batch request',
                timestamp: req.timestamp
            });
        }
        
        // Check if model is ready
        if (!req.pythonBridge.isReady()) {
            return res.status(503).json({
                success: false,
                error: 'Model is not loaded. Please check server status.',
                timestamp: req.timestamp
            });
        }
        
        console.log(`🎯 [${requestId}] Batch predicting categories for ${keywords.length} keywords`);
        
        // Process batch prediction
        const result = await req.pythonBridge.batchPredictCategories(keywords);
        
        const executionTime = Date.now() - startTime;
        
        if (result.success) {
            console.log(`✅ [${requestId}] Batch prediction completed for ${result.total_keywords} keywords in ${executionTime}ms`);
            
            const response = {
                success: true,
                results: result.results,
                total_keywords: result.total_keywords,
                execution_time: executionTime,
                request_id: requestId,
                timestamp: req.timestamp
            };
            
            res.json(response);
        } else {
            console.error(`❌ [${requestId}] Batch prediction failed:`, result.error);
            
            res.status(500).json({
                success: false,
                error: result.error || 'Failed to predict categories',
                results: {},
                total_keywords: 0,
                execution_time: executionTime,
                request_id: requestId,
                timestamp: req.timestamp
            });
        }
        
    } catch (error) {
        const executionTime = Date.now() - startTime;
        console.error(`❌ [${requestId}] Batch prediction error:`, error);
        
        res.status(500).json({
            success: false,
            error: error.message || 'Internal server error',
            execution_time: executionTime,
            request_id: requestId,
            timestamp: req.timestamp
        });
    }
});

/**
 * @route   GET /api/predict/examples
 * @desc    Get prediction examples and usage information
 * @access  Public
 */
router.get('/examples', async (req, res) => {
    try {
        const examples = {
            single_prediction: {
                url: 'POST /api/predict/category',
                description: 'Predict category for a single keyword',
                example_request: {
                    keyword: 'machine learning algorithms'
                },
                example_response: {
                    success: true,
                    keyword: 'machine learning algorithms',
                    predicted_category: 'it',
                    confidence: 0.95,
                    all_probabilities: {
                        'it': 0.95,
                        'economics': 0.03,
                        'marketing': 0.02
                    }
                }
            },
            batch_prediction: {
                url: 'POST /api/predict/batch',
                description: 'Predict categories for multiple keywords',
                example_request: {
                    keywords: [
                        'artificial intelligence',
                        'financial planning',
                        'digital marketing strategy'
                    ]
                },
                example_response: {
                    success: true,
                    results: {
                        'artificial intelligence': {
                            success: true,
                            predicted_category: 'it',
                            confidence: 0.92
                        },
                        'financial planning': {
                            success: true,
                            predicted_category: 'economics',
                            confidence: 0.88
                        }
                    },
                    total_keywords: 3
                }
            },
            parameters: {
                keyword: {
                    type: 'string',
                    required: true,
                    description: 'The keyword to predict category for'
                },
                keywords: {
                    type: 'array',
                    required: true,
                    max_length: 50,
                    description: 'Array of keywords for batch prediction'
                }
            },
            categories: [
                'economics',
                'marketing',
                'it',
                'nan'
            ]
        };
        
        res.json(examples);
    } catch (error) {
        res.status(500).json({
            error: error.message,
            timestamp: req.timestamp
        });
    }
});

module.exports = router;