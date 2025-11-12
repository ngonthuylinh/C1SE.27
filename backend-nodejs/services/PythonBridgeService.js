const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs-extra');

class PythonBridgeService {
    constructor() {
        this.modelPath = path.resolve(process.env.MODEL_PATH || '../models/real_data_question_model.pkl');
        this.pythonPath = process.env.PYTHON_PATH || 'python';
        this.bridgeScript = path.join(__dirname, '../python-bridge-clean.py');
        this.isModelLoaded = false;
        this.modelInfo = null;
    }
    
    async initialize() {
        try {
            // Check if Python bridge script exists
            if (!await fs.pathExists(this.bridgeScript)) {
                throw new Error(`Python bridge script not found: ${this.bridgeScript}`);
            }
            
            // Check if model file exists
            if (!await fs.pathExists(this.modelPath)) {
                throw new Error(`Model file not found: ${this.modelPath}`);
            }
            
            // Test Python bridge
            const healthResult = await this.executeCommand('health_check');
            console.log('🔍 Python Bridge Health Check:', healthResult);
            
            // Load model info
            const modelInfo = await this.executeCommand('model_info');
            if (modelInfo.success) {
                this.isModelLoaded = modelInfo.model_loaded;
                this.modelInfo = modelInfo;
                console.log('📊 Model Info:', modelInfo);
            }
            
            return true;
        } catch (error) {
            console.error('❌ Failed to initialize Python Bridge:', error);
            throw error;
        }
    }
    
    async executeCommand(command, ...args) {
        return new Promise((resolve, reject) => {
            const pythonArgs = [this.bridgeScript, command, ...args];
            const pythonProcess = spawn(this.pythonPath, pythonArgs, {
                stdio: ['pipe', 'pipe', 'pipe'],
                cwd: path.dirname(this.bridgeScript)
            });
            
            let stdout = '';
            let stderr = '';
            
            pythonProcess.stdout.on('data', (data) => {
                stdout += data.toString();
            });
            
            pythonProcess.stderr.on('data', (data) => {
                stderr += data.toString();
            });
            
            pythonProcess.on('close', (code) => {
                try {
                    if (code === 0 && stdout.trim()) {
                        const result = JSON.parse(stdout.trim());
                        resolve(result);
                    } else {
                        reject(new Error(`Python process failed with code ${code}. stderr: ${stderr}`));
                    }
                } catch (parseError) {
                    reject(new Error(`Failed to parse Python output: ${parseError.message}. Raw output: ${stdout}`));
                }
            });
            
            pythonProcess.on('error', (error) => {
                reject(new Error(`Failed to spawn Python process: ${error.message}`));
            });
            
            // Set timeout
            setTimeout(() => {
                pythonProcess.kill('SIGTERM');
                reject(new Error('Python process timeout'));
            }, 30000); // 30 seconds timeout
        });
    }
    
    async getHealthStatus() {
        try {
            const result = await this.executeCommand('health_check');
            return {
                success: true,
                ...result,
                model_loaded: this.isModelLoaded
            };
        } catch (error) {
            return {
                success: false,
                status: 'error',
                model_loaded: false,
                error: error.message,
                timestamp: new Date().toISOString()
            };
        }
    }
    
    async getModelInfo() {
        try {
            if (!this.isModelLoaded) {
                // Try to load model info again
                const result = await this.executeCommand('model_info');
                if (result.success) {
                    this.isModelLoaded = result.model_loaded;
                    this.modelInfo = result;
                }
                return result;
            }
            
            return {
                success: true,
                ...this.modelInfo
            };
        } catch (error) {
            return {
                success: false,
                error: error.message
            };
        }
    }
    
    async predictCategory(keyword) {
        try {
            if (!keyword || typeof keyword !== 'string') {
                throw new Error('Valid keyword is required');
            }
            
            const result = await this.executeCommand('predict_category', keyword);
            return result;
        } catch (error) {
            return {
                success: false,
                error: error.message
            };
        }
    }
    
    async generateQuestions(keyword, numQuestions = 5, category = null) {
        try {
            if (!keyword || typeof keyword !== 'string') {
                throw new Error('Valid keyword is required');
            }
            
            if (numQuestions < 1 || numQuestions > 20) {
                throw new Error('Number of questions must be between 1 and 20');
            }
            
            const args = [keyword, numQuestions.toString()];
            if (category) {
                args.push(category);
            }
            
            const result = await this.executeCommand('generate_questions', ...args);
            
            if (result.success) {
                // Add execution time
                result.execution_time = Date.now();
                result.timestamp = new Date().toISOString();
            }
            
            return result;
        } catch (error) {
            return {
                success: false,
                error: error.message,
                keyword,
                questions: []
            };
        }
    }
    
    async batchPredictCategories(keywords) {
        try {
            if (!Array.isArray(keywords) || keywords.length === 0) {
                throw new Error('Valid keywords array is required');
            }
            
            if (keywords.length > 50) {
                throw new Error('Maximum 50 keywords allowed per batch');
            }
            
            const results = {};
            
            // Process keywords in parallel with limit
            const promises = keywords.map(async (keyword) => {
                if (typeof keyword === 'string' && keyword.trim()) {
                    try {
                        const result = await this.predictCategory(keyword.trim());
                        return { keyword, result };
                    } catch (error) {
                        return { 
                            keyword, 
                            result: { 
                                success: false, 
                                error: error.message 
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
            
            return {
                success: true,
                results,
                total_keywords: Object.keys(results).length
            };
            
        } catch (error) {
            return {
                success: false,
                error: error.message,
                results: {},
                total_keywords: 0
            };
        }
    }
    
    // Utility methods
    isReady() {
        return this.isModelLoaded;
    }
    
    getModelPath() {
        return this.modelPath;
    }
    
    setModelPath(newPath) {
        this.modelPath = newPath;
        this.isModelLoaded = false;
        this.modelInfo = null;
    }
}

module.exports = PythonBridgeService;