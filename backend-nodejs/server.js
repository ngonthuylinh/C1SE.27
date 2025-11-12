const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
const compression = require('compression');
const rateLimit = require('express-rate-limit');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs-extra');
require('dotenv').config({ path: path.join(__dirname, '.env') });

// Import routes
const healthRoutes = require('./routes/health');
const questionRoutes = require('./routes/questions');
const predictionRoutes = require('./routes/predictions');
const modelRoutes = require('./routes/model');
const surveyRoutes = require('./routes/surveyRoutes');
const formRoutes = require('./routes/formRoutes');

// Import services
const PythonBridgeService = require('./services/PythonBridgeService');

class FormAgentServer {
    constructor() {
        this.app = express();
        this.port = process.env.PORT || 8001;
        this.pythonBridge = new PythonBridgeService();
        
        this.setupMiddleware();
        this.setupRoutes();
        this.setupErrorHandling();
    }
    
    setupMiddleware() {
        // Security middleware
        this.app.use(helmet({
            contentSecurityPolicy: false, // Allow inline styles for development
        }));
        
        // Compression
        this.app.use(compression());
        
        // CORS configuration
        const corsOptions = {
            origin: (process.env.ALLOWED_ORIGINS || 'http://localhost:3000').split(','),
            credentials: true,
            methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
            allowedHeaders: ['Content-Type', 'Authorization', 'x-requested-with']
        };
        this.app.use(cors(corsOptions));
        
        // Rate limiting
        const limiter = rateLimit({
            windowMs: parseInt(process.env.RATE_LIMIT_WINDOW_MS) || 15 * 60 * 1000, // 15 minutes
            max: parseInt(process.env.RATE_LIMIT_MAX_REQUESTS) || 100, // limit each IP to 100 requests per windowMs
            message: {
                error: 'Too many requests from this IP, please try again later.',
                code: 'RATE_LIMIT_EXCEEDED'
            },
            standardHeaders: true,
            legacyHeaders: false
        });
        this.app.use('/api/', limiter);
        
        // Logging
        if (process.env.NODE_ENV !== 'test') {
            this.app.use(morgan('combined'));
        }
        
        // Body parsing
        this.app.use(express.json({ limit: '10mb' }));
        this.app.use(express.urlencoded({ extended: true, limit: '10mb' }));
        
        // Request timestamp
        this.app.use((req, res, next) => {
            req.timestamp = new Date().toISOString();
            next();
        });
        
        // Inject Python bridge service
        this.app.use((req, res, next) => {
            req.pythonBridge = this.pythonBridge;
            next();
        });
    }
    
    setupRoutes() {
        // API routes
        this.app.use('/api/health', healthRoutes);
        this.app.use('/api/questions', questionRoutes);
        this.app.use('/api/predict', predictionRoutes);
        this.app.use('/api/model', modelRoutes);
        this.app.use('/api/survey', surveyRoutes);
        
        // Public form routes
        this.app.use('/form', formRoutes);
        
        // Root route
        this.app.get('/', (req, res) => {
            res.json({
                name: 'Form Agent AI Backend',
                version: '1.0.0',
                status: 'running',
                timestamp: req.timestamp,
                endpoints: {
                    health: '/api/health',
                    questions: '/api/questions',
                    predictions: '/api/predict',
                    model: '/api/model'
                }
            });
        });
        
        // API documentation
        this.app.get('/api', (req, res) => {
            res.json({
                title: 'Form Agent AI API',
                version: '1.0.0',
                description: 'AI-powered question generation API',
                endpoints: {
                    'GET /api/health': 'Check server and model health',
                    'GET /api/model/info': 'Get model information',
                    'POST /api/questions/generate': 'Generate questions from keyword',
                    'POST /api/predict/category': 'Predict category for keyword',
                    'POST /api/predict/batch': 'Batch predict categories'
                },
                examples: {
                    generateQuestions: {
                        url: 'POST /api/questions/generate',
                        body: {
                            keyword: 'artificial intelligence',
                            num_questions: 5,
                            category: 'it'
                        }
                    },
                    predictCategory: {
                        url: 'POST /api/predict/category',
                        body: {
                            keyword: 'machine learning'
                        }
                    }
                }
            });
        });
    }
    
    setupErrorHandling() {
        // 404 handler
        this.app.use('*', (req, res) => {
            res.status(404).json({
                error: 'Route not found',
                message: `Cannot ${req.method} ${req.originalUrl}`,
                timestamp: req.timestamp
            });
        });
        
        // Global error handler
        this.app.use((error, req, res, next) => {
            console.error('Global error handler:', error);
            
            const status = error.status || 500;
            const message = error.message || 'Internal server error';
            
            res.status(status).json({
                error: message,
                ...(process.env.NODE_ENV === 'development' && { 
                    stack: error.stack,
                    details: error 
                }),
                timestamp: req.timestamp
            });
        });
    }
    
    async start() {
        try {
            // Initialize Python bridge
            console.log('🔧 Initializing Python Bridge...');
            await this.pythonBridge.initialize();
            console.log('✅ Python Bridge initialized successfully');
            
            // Start server
            this.server = this.app.listen(this.port, '0.0.0.0', () => {
                console.log('\n🚀 Form Agent AI Backend Started!');
                console.log('=======================================');
                console.log(`📋 Environment: ${process.env.NODE_ENV || 'development'}`);
                console.log(`🌐 Server: http://0.0.0.0:${this.port}`);
                console.log(`📚 API Docs: http://0.0.0.0:${this.port}/api`);
                console.log(`🔧 Health Check: http://0.0.0.0:${this.port}/api/health`);
                console.log('=======================================\n');
            });
            
            return this.server;
        } catch (error) {
            console.error('❌ Failed to start server:', error);
            process.exit(1);
        }
    }
    
    async stop() {
        if (this.server) {
            return new Promise((resolve) => {
                this.server.close(resolve);
            });
        }
    }
}

// Handle graceful shutdown
process.on('SIGTERM', async () => {
    console.log('🛑 SIGTERM received, shutting down gracefully...');
    if (global.server) {
        await global.server.stop();
    }
    process.exit(0);
});

process.on('SIGINT', async () => {
    console.log('🛑 SIGINT received, shutting down gracefully...');
    if (global.server) {
        await global.server.stop();
    }
    process.exit(0);
});

// Start server if this file is run directly
if (require.main === module) {
    const server = new FormAgentServer();
    global.server = server;
    server.start().catch(console.error);
}

module.exports = FormAgentServer;