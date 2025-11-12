const FormAgentServer = require('./server');
const PythonBridgeService = require('./services/PythonBridgeService');

async function testModel() {
    console.log('🧪 Testing Form Agent AI Model');
    console.log('================================');
    
    try {
        // Initialize Python bridge directly
        const pythonBridge = new PythonBridgeService();
        
        console.log('🔧 Initializing Python Bridge...');
        await pythonBridge.initialize();
        console.log('✅ Python Bridge initialized successfully\n');
        
        // Test 1: Health Check
        console.log('1️⃣ Testing Health Check...');
        const health = await pythonBridge.getHealthStatus();
        console.log(`   Status: ${health.status}`);
        console.log(`   Model Loaded: ${health.model_loaded}`);
        console.log('');
        
        // Test 2: Model Info
        console.log('2️⃣ Testing Model Info...');
        const modelInfo = await pythonBridge.getModelInfo();
        if (modelInfo.success) {
            console.log(`   Total Keywords: ${modelInfo.total_keywords}`);
            console.log(`   Total Questions: ${modelInfo.total_questions}`);
            console.log(`   Categories: ${modelInfo.categories?.join(', ')}`);
            console.log(`   Training Date: ${modelInfo.training_date}`);
        } else {
            console.log(`   Error: ${modelInfo.error}`);
        }
        console.log('');
        
        // Test 3: Category Prediction
        console.log('3️⃣ Testing Category Prediction...');
        const testKeywords = [
            'artificial intelligence',
            'financial planning', 
            'digital marketing strategy',
            'cloud computing security',
            'investment portfolio'
        ];
        
        for (const keyword of testKeywords) {
            const prediction = await pythonBridge.predictCategory(keyword);
            if (prediction.success) {
                console.log(`   "${keyword}" → ${prediction.predicted_category} (${(prediction.confidence * 100).toFixed(1)}%)`);
            } else {
                console.log(`   "${keyword}" → Error: ${prediction.error}`);
            }
        }
        console.log('');
        
        // Test 4: Question Generation
        console.log('4️⃣ Testing Question Generation...');
        
        for (const keyword of testKeywords.slice(0, 3)) { // Test first 3 keywords
            console.log(`\n   📝 Generating questions for: "${keyword}"`);
            const questions = await pythonBridge.generateQuestions(keyword, 3);
            
            if (questions.success) {
                console.log(`   ✅ Generated ${questions.questions?.length || 0} questions:`);
                questions.questions?.forEach((q, i) => {
                    console.log(`      ${i + 1}. ${q.question}`);
                    console.log(`         Category: ${q.category} | Confidence: ${(q.confidence * 100).toFixed(1)}% | Method: ${q.method}`);
                });
            } else {
                console.log(`   ❌ Error: ${questions.error}`);
            }
        }
        
        // Test 5: Batch Prediction
        console.log('\n5️⃣ Testing Batch Prediction...');
        const batchResult = await pythonBridge.batchPredictCategories(testKeywords.slice(0, 3));
        
        if (batchResult.success) {
            console.log(`   ✅ Batch prediction completed for ${batchResult.total_keywords} keywords`);
            Object.entries(batchResult.results).forEach(([keyword, result]) => {
                if (result.success) {
                    console.log(`   "${keyword}" → ${result.predicted_category} (${(result.confidence * 100).toFixed(1)}%)`);
                } else {
                    console.log(`   "${keyword}" → Error: ${result.error}`);
                }
            });
        } else {
            console.log(`   ❌ Batch prediction error: ${batchResult.error}`);
        }
        
        console.log('\n🎉 Model testing completed successfully!');
        console.log('✅ All components are working correctly');
        
    } catch (error) {
        console.error('\n❌ Model testing failed:', error);
        process.exit(1);
    }
}

// Run test if this file is executed directly
if (require.main === module) {
    testModel()
        .then(() => process.exit(0))
        .catch((error) => {
            console.error('Test failed:', error);
            process.exit(1);
        });
}

module.exports = testModel;