const express = require('express');
const router = express.Router();

/**
 * @route   GET /api/health
 * @desc    Check server and model health status
 * @access  Public
 */
router.get('/', async (req, res) => {
    try {
        const healthStatus = await req.pythonBridge.getHealthStatus();
        
        const response = {
            status: 'ok',
            message: 'Server đang hoạt động',
            model_loaded: healthStatus.model_loaded || false,
            timestamp: req.timestamp,
            server: {
                uptime: process.uptime(),
                memory: process.memoryUsage(),
                version: process.version,
                platform: process.platform
            }
        };
        
        // Set appropriate status code
        const statusCode = healthStatus.model_loaded ? 200 : 503;
        
        res.status(statusCode).json(response);
    } catch (error) {
        console.error('Health check error:', error);
        res.status(503).json({
            status: 'error',
            message: 'Service unavailable',
            model_loaded: false,
            error: error.message,
            timestamp: req.timestamp
        });
    }
});

/**
 * @route   GET /api/health/detailed
 * @desc    Get detailed health information including model info
 * @access  Public
 */
router.get('/detailed', async (req, res) => {
    try {
        const [healthStatus, modelInfo] = await Promise.all([
            req.pythonBridge.getHealthStatus(),
            req.pythonBridge.getModelInfo()
        ]);
        
        const response = {
            status: 'ok',
            timestamp: req.timestamp,
            server: {
                uptime: process.uptime(),
                memory: process.memoryUsage(),
                cpu: process.cpuUsage(),
                version: process.version,
                platform: process.platform,
                environment: process.env.NODE_ENV || 'development'
            },
            model: {
                loaded: healthStatus.model_loaded || false,
                path: req.pythonBridge.getModelPath(),
                ...modelInfo
            },
            python_bridge: healthStatus
        };
        
        res.json(response);
    } catch (error) {
        console.error('Detailed health check error:', error);
        res.status(500).json({
            status: 'error',
            message: 'Failed to get detailed health information',
            error: error.message,
            timestamp: req.timestamp
        });
    }
});

/**
 * @route   POST /api/health/reload
 * @desc    Reload the model
 * @access  Public
 */
router.post('/reload', async (req, res) => {
    try {
        console.log('🔄 Reloading model...');
        await req.pythonBridge.initialize();
        
        const modelInfo = await req.pythonBridge.getModelInfo();
        
        res.json({
            success: true,
            message: 'Model reloaded successfully',
            model_info: modelInfo,
            timestamp: req.timestamp
        });
    } catch (error) {
        console.error('Model reload error:', error);
        res.status(500).json({
            success: false,
            message: 'Failed to reload model',
            error: error.message,
            timestamp: req.timestamp
        });
    }
});

module.exports = router;