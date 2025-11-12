const express = require('express');
const router = express.Router();
const { v4: uuidv4 } = require('uuid');

/**
 * @route   POST /api/questions/generate
 * @desc    Generate questions from keyword using AI model
 * @access  Public
 */
router.post('/generate', async (req, res) => {
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
        
        const { keyword, num_questions = 5, category } = req.body;
        
        // Validate keyword
        if (!keyword || typeof keyword !== 'string' || !keyword.trim()) {
            return res.status(400).json({
                success: false,
                error: 'Keyword is required and must be a non-empty string',
                timestamp: req.timestamp
            });
        }
        
        // Validate num_questions
        const numQuestions = parseInt(num_questions);
        if (isNaN(numQuestions) || numQuestions < 1 || numQuestions > 20) {
            return res.status(400).json({
                success: false,
                error: 'num_questions must be a number between 1 and 20',
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
        
        console.log(`📝 [${requestId}] Generating ${numQuestions} questions for keyword: "${keyword.trim()}"${category ? ` (category: ${category})` : ''}`);
        
        // Generate questions
        const result = await req.pythonBridge.generateQuestions(
            keyword.trim(), 
            numQuestions, 
            category?.trim() || null
        );
        
        const executionTime = Date.now() - startTime;
        
        if (result.success) {
            console.log(`✅ [${requestId}] Generated ${result.questions?.length || 0} questions in ${executionTime}ms`);
            
            const response = {
                success: true,
                questions: result.questions || [],
                keyword: keyword.trim(),
                total_generated: result.questions?.length || 0,
                execution_time: executionTime,
                predicted_category: result.predicted_category,
                confidence: result.confidence,
                request_id: requestId,
                timestamp: req.timestamp
            };
            
            res.json(response);
        } else {
            console.error(`❌ [${requestId}] Question generation failed:`, result.error);
            
            res.status(500).json({
                success: false,
                error: result.error || 'Failed to generate questions',
                keyword: keyword.trim(),
                questions: [],
                execution_time: executionTime,
                request_id: requestId,
                timestamp: req.timestamp
            });
        }
        
    } catch (error) {
        const executionTime = Date.now() - startTime;
        console.error(`❌ [${requestId}] Question generation error:`, error);
        
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
 * @route   GET /api/questions/examples
 * @desc    Get example questions and usage
 * @access  Public
 */
router.get('/examples', async (req, res) => {
    try {
        const examples = {
            usage: 'POST /api/questions/generate',
            description: 'Generate AI-powered questions from keywords',
            examples: [
                {
                    request: {
                        keyword: 'artificial intelligence',
                        num_questions: 5,
                        category: 'it'
                    },
                    description: 'Generate 5 IT-related questions about AI'
                },
                {
                    request: {
                        keyword: 'investment planning',
                        num_questions: 3
                    },
                    description: 'Generate 3 questions about investment planning (auto-detect category)'
                },
                {
                    request: {
                        keyword: 'digital marketing',
                        num_questions: 7,
                        category: 'marketing'
                    },
                    description: 'Generate 7 marketing questions about digital marketing'
                }
            ],
            parameters: {
                keyword: {
                    type: 'string',
                    required: true,
                    description: 'The keyword to generate questions from'
                },
                num_questions: {
                    type: 'number',
                    required: false,
                    default: 5,
                    range: '1-20',
                    description: 'Number of questions to generate'
                },
                category: {
                    type: 'string',
                    required: false,
                    options: ['economics', 'marketing', 'it'],
                    description: 'Optional category hint for better question generation'
                }
            }
        };
        
        res.json(examples);
    } catch (error) {
        res.status(500).json({
            error: error.message,
            timestamp: req.timestamp
        });
    }
});

/**
 * @route   POST /api/questions/batch
 * @desc    Generate questions for multiple keywords in batch
 * @access  Public
 */
router.post('/batch', async (req, res) => {
    const requestId = uuidv4();
    const startTime = Date.now();
    
    try {
        const { keywords, num_questions = 5 } = req.body;
        
        if (!Array.isArray(keywords) || keywords.length === 0) {
            return res.status(400).json({
                success: false,
                error: 'Keywords array is required',
                timestamp: req.timestamp
            });
        }
        
        if (keywords.length > 10) {
            return res.status(400).json({
                success: false,
                error: 'Maximum 10 keywords allowed per batch request',
                timestamp: req.timestamp
            });
        }
        
        console.log(`📝 [${requestId}] Batch generating questions for ${keywords.length} keywords`);
        
        const results = {};
        const promises = keywords.map(async (keyword) => {
            if (typeof keyword === 'string' && keyword.trim()) {
                try {
                    const result = await req.pythonBridge.generateQuestions(keyword.trim(), num_questions);
                    return { keyword: keyword.trim(), result };
                } catch (error) {
                    return { 
                        keyword: keyword.trim(), 
                        result: { 
                            success: false, 
                            error: error.message,
                            questions: []
                        } 
                    };
                }
            }
            return null;
        });
        
        const batchResults = await Promise.all(promises);
        
        batchResults.forEach(item => {
            if (item && item.keyword) {
                results[item.keyword] = item.result;
            }
        });
        
        const executionTime = Date.now() - startTime;
        
        console.log(`✅ [${requestId}] Batch generation completed in ${executionTime}ms`);
        
        res.json({
            success: true,
            results,
            total_keywords: Object.keys(results).length,
            execution_time: executionTime,
            request_id: requestId,
            timestamp: req.timestamp
        });
        
    } catch (error) {
        const executionTime = Date.now() - startTime;
        console.error(`❌ [${requestId}] Batch generation error:`, error);
        
        res.status(500).json({
            success: false,
            error: error.message,
            execution_time: executionTime,
            request_id: requestId,
            timestamp: req.timestamp
        });
    }
});

module.exports = router;