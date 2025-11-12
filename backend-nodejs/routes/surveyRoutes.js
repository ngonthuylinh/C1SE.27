const express = require('express');
const router = express.Router();
const path = require('path');
const fs = require('fs-extra');
const { spawn } = require('child_process');

// Generate simple ID without uuid
const generateId = () => {
    return Date.now().toString(36) + Math.random().toString(36).substr(2, 5);
};

// Survey forms storage
const FORMS_DIR = path.join(__dirname, '../forms');
const FORMS_INDEX_FILE = path.join(FORMS_DIR, 'forms_index.json');

// Ensure forms directory exists
fs.ensureDirSync(FORMS_DIR);

// Load forms index
let formsIndex = {};
if (fs.existsSync(FORMS_INDEX_FILE)) {
    try {
        formsIndex = fs.readJsonSync(FORMS_INDEX_FILE);
    } catch (error) {
        console.log('Creating new forms index...');
        formsIndex = {};
    }
}

// Save forms index
const saveFormsIndex = () => {
    fs.writeJsonSync(FORMS_INDEX_FILE, formsIndex, { spaces: 2 });
};

/**
 * POST /api/survey/create
 * Create a new survey form
 */
router.post('/create', async (req, res) => {
    try {
        const { title, category, description, questions, settings } = req.body;
        
        // Validation
        if (!title || !questions || !Array.isArray(questions) || questions.length === 0) {
            return res.status(400).json({
                success: false,
                error: 'Title and questions are required'
            });
        }

        // Generate unique form ID
        const formId = generateId();
        const timestamp = new Date().toISOString();
        
        // Create form data
        const formData = {
            id: formId,
            title: title.trim(),
            category: category || 'general',
            description: description || '',
            questions: questions.map((q, index) => ({
                id: index + 1,
                question: typeof q === 'string' ? q : q.question,
                type: (typeof q === 'object' ? q.type : null) || 'text',
                required: (typeof q === 'object' ? q.required : null) || false,
                options: typeof q === 'object' ? q.options : null
            })),
            settings: {
                allowAnonymous: settings?.allowAnonymous || true,
                collectEmail: settings?.collectEmail || false,
                multipleSubmissions: settings?.multipleSubmissions || false,
                ...settings
            },
            createdAt: timestamp,
            updatedAt: timestamp,
            responses: []
        };
        
        // Save form data
        const formFile = path.join(FORMS_DIR, `${formId}.json`);
        await fs.writeJson(formFile, formData, { spaces: 2 });
        
        // Update forms index
        formsIndex[formId] = {
            id: formId,
            title: formData.title,
            category: formData.category,
            questionCount: formData.questions.length,
            responseCount: 0,
            createdAt: timestamp,
            isActive: true
        };
        saveFormsIndex();
        
        // Generate PDF in background
        const pdfPath = await generateFormPDF(formData);
        
        // Generate share URLs
        const shareUrl = `${req.protocol}://${req.get('host')}/form/${formId}`;
        const pdfUrl = `${req.protocol}://${req.get('host')}/api/survey/pdf/${formId}`;
        
        res.json({
            success: true,
            form: {
                id: formId,
                title: formData.title,
                questionCount: formData.questions.length,
                shareUrl,
                pdfUrl,
                createdAt: timestamp
            }
        });
        
    } catch (error) {
        console.error('Error creating survey:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to create survey'
        });
    }
});

/**
 * GET /api/survey/list
 * List all surveys
 */
router.get('/list', (req, res) => {
    try {
        const surveys = Object.values(formsIndex)
            .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
        
        res.json({
            success: true,
            surveys
        });
    } catch (error) {
        res.status(500).json({
            success: false,
            error: 'Failed to list surveys'
        });
    }
});

/**
 * GET /api/survey/:id
 * Get survey details
 */
router.get('/:id', async (req, res) => {
    try {
        const formId = req.params.id;
        const formFile = path.join(FORMS_DIR, `${formId}.json`);
        
        if (!await fs.pathExists(formFile)) {
            return res.status(404).json({
                success: false,
                error: 'Survey not found'
            });
        }
        
        const formData = await fs.readJson(formFile);
        
        res.json({
            success: true,
            survey: formData
        });
        
    } catch (error) {
        res.status(500).json({
            success: false,
            error: 'Failed to get survey'
        });
    }
});

/**
 * GET /api/survey/pdf/:id
 * Download survey PDF
 */
router.get('/pdf/:id', async (req, res) => {
    try {
        const formId = req.params.id;
        const pdfPath = path.join(FORMS_DIR, `${formId}.pdf`);
        
        if (!await fs.pathExists(pdfPath)) {
            return res.status(404).json({
                success: false,
                error: 'PDF not found'
            });
        }
        
        res.download(pdfPath, `survey_${formId}.pdf`);
        
    } catch (error) {
        res.status(500).json({
            success: false,
            error: 'Failed to download PDF'
        });
    }
});

/**
 * POST /api/survey/:id/submit
 * Submit survey response
 */
router.post('/:id/submit', async (req, res) => {
    try {
        const formId = req.params.id;
        const formFile = path.join(FORMS_DIR, `${formId}.json`);
        
        if (!await fs.pathExists(formFile)) {
            return res.status(404).json({
                success: false,
                error: 'Survey not found'
            });
        }
        
        const formData = await fs.readJson(formFile);
        const { responses, metadata } = req.body;
        
        // Validate responses
        if (!responses || typeof responses !== 'object') {
            return res.status(400).json({
                success: false,
                error: 'Invalid responses'
            });
        }
        
        // Create response record
        const responseRecord = {
            id: generateId(),
            responses,
            metadata: {
                userAgent: req.get('User-Agent'),
                ip: req.ip,
                timestamp: new Date().toISOString(),
                ...metadata
            }
        };
        
        // Add response to form data
        formData.responses.push(responseRecord);
        formData.updatedAt = new Date().toISOString();
        
        // Save updated form
        await fs.writeJson(formFile, formData, { spaces: 2 });
        
        // Update forms index
        if (formsIndex[formId]) {
            formsIndex[formId].responseCount = formData.responses.length;
            saveFormsIndex();
        }
        
        res.json({
            success: true,
            message: 'Response submitted successfully',
            responseId: responseRecord.id
        });
        
    } catch (error) {
        console.error('Error submitting response:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to submit response'
        });
    }
});

/**
 * Generate PDF for survey form
 */
async function generateFormPDF(formData) {
    return new Promise((resolve, reject) => {
        const pythonScript = path.join(__dirname, '../pdf_generator.py');
        const pdfPath = path.join(FORMS_DIR, `${formData.id}.pdf`);
        
        const pythonProcess = spawn('python', [
            pythonScript,
            JSON.stringify(formData),
            pdfPath
        ]);
        
        let stderr = '';
        
        pythonProcess.stderr.on('data', (data) => {
            stderr += data.toString();
        });
        
        pythonProcess.on('close', (code) => {
            if (code === 0) {
                resolve(pdfPath);
            } else {
                console.error('PDF generation error:', stderr);
                reject(new Error('PDF generation failed'));
            }
        });
        
        pythonProcess.on('error', (error) => {
            reject(error);
        });
    });
}

module.exports = router;