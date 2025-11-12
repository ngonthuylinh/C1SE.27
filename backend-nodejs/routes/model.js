const express = require('express');
const router = express.Router();

/**
 * @route   GET /api/model/info
 * @desc    Get detailed information about the loaded model
 * @access  Public
 */
router.get('/info', async (req, res) => {
    try {
        const modelInfo = await req.pythonBridge.getModelInfo();
        
        if (modelInfo.success) {
            const response = {
                success: true,
                model_loaded: modelInfo.model_loaded || false,
                training_date: modelInfo.training_date,
                total_keywords: modelInfo.total_keywords || 0,
                total_questions: modelInfo.total_questions || 0,
                categories: modelInfo.categories || [],
                model_path: req.pythonBridge.getModelPath(),
                timestamp: req.timestamp
            };
            
            res.json(response);
        } else {
            res.status(500).json({
                success: false,
                error: modelInfo.error || 'Failed to get model information',
                model_loaded: false,
                timestamp: req.timestamp
            });
        }
        
    } catch (error) {
        console.error('Model info error:', error);
        res.status(500).json({
            success: false,
            error: error.message || 'Internal server error',
            model_loaded: false,
            timestamp: req.timestamp
        });
    }
});

/**
 * @route   GET /api/model/stats
 * @desc    Get model statistics and performance metrics
 * @access  Public
 */
router.get('/stats', async (req, res) => {
    try {
        const [modelInfo, healthStatus] = await Promise.all([
            req.pythonBridge.getModelInfo(),
            req.pythonBridge.getHealthStatus()
        ]);
        
        const response = {
            success: true,
            model_loaded: modelInfo.model_loaded || false,
            statistics: {
                total_keywords: modelInfo.total_keywords || 0,
                total_questions: modelInfo.total_questions || 0,
                categories: modelInfo.categories || [],
                training_date: modelInfo.training_date
            },
            health: {
                status: healthStatus.status || 'unknown',
                model_loaded: healthStatus.model_loaded || false
            },
            system: {
                model_path: req.pythonBridge.getModelPath(),
                python_bridge_ready: req.pythonBridge.isReady(),
                server_uptime: process.uptime(),
                memory_usage: process.memoryUsage()
            },
            timestamp: req.timestamp
        };
        
        res.json(response);
        
    } catch (error) {
        console.error('Model stats error:', error);
        res.status(500).json({
            success: false,
            error: error.message || 'Failed to get model statistics',
            timestamp: req.timestamp
        });
    }
});

/**
 * @route   GET /api/model/categories
 * @desc    Get available categories from the model
 * @access  Public
 */
router.get('/categories', async (req, res) => {
    try {
        const modelInfo = await req.pythonBridge.getModelInfo();
        
        if (modelInfo.success) {
            const categories = modelInfo.categories || [];
            
            res.json({
                success: true,
                categories,
                total_categories: categories.length,
                timestamp: req.timestamp
            });
        } else {
            res.status(500).json({
                success: false,
                error: 'Failed to get categories from model',
                categories: [],
                total_categories: 0,
                timestamp: req.timestamp
            });
        }
        
    } catch (error) {
        console.error('Categories error:', error);
        res.status(500).json({
            success: false,
            error: error.message || 'Internal server error',
            categories: [],
            total_categories: 0,
            timestamp: req.timestamp
        });
    }
});

/**
 * @route   POST /api/model/reload
 * @desc    Reload the model (same as health reload but specific to model)
 * @access  Public
 */
router.post('/reload', async (req, res) => {
    try {
        console.log('🔄 Reloading model via model endpoint...');
        
        // Reinitialize Python bridge
        await req.pythonBridge.initialize();
        
        // Get updated model info
        const modelInfo = await req.pythonBridge.getModelInfo();
        
        if (modelInfo.success) {
            console.log('✅ Model reloaded successfully');
            
            res.json({
                success: true,
                message: 'Model reloaded successfully',
                model_info: modelInfo,
                timestamp: req.timestamp
            });
        } else {
            throw new Error(modelInfo.error || 'Model reload failed');
        }
        
    } catch (error) {
        console.error('❌ Model reload error:', error);
        res.status(500).json({
            success: false,
            message: 'Failed to reload model',
            error: error.message,
            timestamp: req.timestamp
        });
    }
});

/**
 * @route   GET /api/model/test
 * @desc    Test model functionality with sample data
 * @access  Public
 */
router.get('/test', async (req, res) => {
    try {
        if (!req.pythonBridge.isReady()) {
            return res.status(503).json({
                success: false,
                error: 'Model is not ready for testing',
                timestamp: req.timestamp
            });
        }
        
        console.log('🧪 Testing model functionality...');
        
        const testKeywords = [
            'artificial intelligence',
            'financial planning',
            'digital marketing'
        ];
        
        const testResults = {};
        
        for (const keyword of testKeywords) {
            try {
                // Test category prediction
                const categoryResult = await req.pythonBridge.predictCategory(keyword);
                
                // Test question generation
                const questionResult = await req.pythonBridge.generateQuestions(keyword, 3);
                
                testResults[keyword] = {
                    category_prediction: {
                        success: categoryResult.success,
                        predicted_category: categoryResult.predicted_category,
                        confidence: categoryResult.confidence
                    },
                    question_generation: {
                        success: questionResult.success,
                        questions_count: questionResult.questions?.length || 0,
                        questions: questionResult.questions || []
                    }
                };
                
            } catch (error) {
                testResults[keyword] = {
                    error: error.message
                };
            }
        }
        
        console.log('✅ Model testing completed');
        
        res.json({
            success: true,
            message: 'Model testing completed',
            test_results: testResults,
            timestamp: req.timestamp
        });
        
    } catch (error) {
        console.error('❌ Model test error:', error);
        res.status(500).json({
            success: false,
            error: error.message || 'Model testing failed',
            timestamp: req.timestamp
        });
    }
});

module.exports = router;