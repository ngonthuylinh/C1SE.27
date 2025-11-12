const express = require('express');
const fs = require('fs-extra');
const path = require('path');
const { spawn } = require('child_process');
const crypto = require('crypto');

const router = express.Router();

// Generate random ID using crypto
function generateId() {
  return crypto.randomBytes(4).toString('hex');
}

// Storage for forms (in production, use database)
const FORMS_FILE = path.join(__dirname, '../forms/forms.json');
const FORMS_DIR = path.join(__dirname, '../forms');

// Ensure forms directory exists
fs.ensureDirSync(FORMS_DIR);

// Load forms from file
async function loadForms() {
  try {
    if (await fs.pathExists(FORMS_FILE)) {
      return await fs.readJson(FORMS_FILE);
    }
    return {};
  } catch (error) {
    console.error('Error loading forms:', error);
    return {};
  }
}

// Save forms to file
async function saveForms(forms) {
  try {
    await fs.writeJson(FORMS_FILE, forms, { spaces: 2 });
  } catch (error) {
    console.error('Error saving forms:', error);
    throw error;
  }
}

// Create survey form
router.post('/create', async (req, res) => {
  try {
    const { title, category, questions, description } = req.body;
    
    if (!title || !questions || !Array.isArray(questions) || questions.length === 0) {
      return res.status(400).json({
        success: false,
        error: 'Title and questions array are required'
      });
    }
    
    // Generate unique form ID
    const formId = generateId();
    
    // Form data structure
    const formData = {
      id: formId,
      title: title.trim(),
      description: description || '',
      category: category || 'general',
      questions: questions.map((q, index) => ({
        id: index + 1,
        question: q.trim(),
        type: 'text', // Can be extended: text, multiple_choice, rating, etc.
        required: true
      })),
      created_at: new Date().toISOString(),
      responses: []
    };
    
    // Load existing forms
    const forms = await loadForms();
    forms[formId] = formData;
    
    // Save forms
    await saveForms(forms);
    
    // Generate public URL
    const baseUrl = process.env.BASE_URL || `http://localhost:${process.env.PORT || 8001}`;
    const shareUrl = `${baseUrl}/survey/form/${formId}`;
    
    res.json({
      success: true,
      form_id: formId,
      share_url: shareUrl,
      form_data: formData
    });
    
  } catch (error) {
    console.error('Error creating survey:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to create survey'
    });
  }
});

// Export survey to PDF
router.post('/export-pdf/:formId', async (req, res) => {
  try {
    const { formId } = req.params;
    
    // Load forms
    const forms = await loadForms();
    const formData = forms[formId];
    
    if (!formData) {
      return res.status(404).json({
        success: false,
        error: 'Form not found'
      });
    }
    
    // Call Python script to generate PDF
    const pythonScript = path.join(__dirname, '../pdf_generator.py');
    const outputFile = path.join(FORMS_DIR, `${formId}.pdf`);
    
    return new Promise((resolve, reject) => {
      const pythonProcess = spawn('python', [
        pythonScript,
        JSON.stringify(formData),
        outputFile
      ]);
      
      let stderr = '';
      
      pythonProcess.stderr.on('data', (data) => {
        stderr += data.toString();
      });
      
      pythonProcess.on('close', async (code) => {
        if (code === 0) {
          try {
            // Check if file was created
            if (await fs.pathExists(outputFile)) {
              res.download(outputFile, `${formData.title}.pdf`, (err) => {
                if (err) {
                  console.error('Download error:', err);
                  res.status(500).json({
                    success: false,
                    error: 'Failed to download PDF'
                  });
                }
              });
              resolve();
            } else {
              res.status(500).json({
                success: false,
                error: 'PDF file was not generated'
              });
              resolve();
            }
          } catch (error) {
            res.status(500).json({
              success: false,
              error: 'Failed to access PDF file'
            });
            resolve();
          }
        } else {
          console.error('Python script failed:', stderr);
          res.status(500).json({
            success: false,
            error: 'Failed to generate PDF',
            details: stderr
          });
          resolve();
        }
      });
      
      pythonProcess.on('error', (error) => {
        console.error('Python process error:', error);
        res.status(500).json({
          success: false,
          error: 'Failed to start PDF generation process'
        });
        resolve();
      });
    });
    
  } catch (error) {
    console.error('Error exporting PDF:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to export PDF'
    });
  }
});

// Get survey form data
router.get('/form/:formId', async (req, res) => {
  try {
    const { formId } = req.params;
    
    // Load forms
    const forms = await loadForms();
    const formData = forms[formId];
    
    if (!formData) {
      return res.status(404).json({
        success: false,
        error: 'Form not found'
      });
    }
    
    res.json({
      success: true,
      form: formData
    });
    
  } catch (error) {
    console.error('Error getting form:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to get form'
    });
  }
});

// Submit survey response
router.post('/form/:formId/submit', async (req, res) => {
  try {
    const { formId } = req.params;
    const { responses } = req.body;
    
    if (!responses || typeof responses !== 'object') {
      return res.status(400).json({
        success: false,
        error: 'Responses object is required'
      });
    }
    
    // Load forms
    const forms = await loadForms();
    const formData = forms[formId];
    
    if (!formData) {
      return res.status(404).json({
        success: false,
        error: 'Form not found'
      });
    }
    
    // Add response
    const responseData = {
      id: generateId(),
      responses: responses,
      submitted_at: new Date().toISOString(),
      ip: req.ip
    };
    
    formData.responses.push(responseData);
    forms[formId] = formData;
    
    // Save updated forms
    await saveForms(forms);
    
    res.json({
      success: true,
      message: 'Response submitted successfully',
      response_id: responseData.id
    });
    
  } catch (error) {
    console.error('Error submitting response:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to submit response'
    });
  }
});

// List all forms (admin endpoint)
router.get('/list', async (req, res) => {
  try {
    const forms = await loadForms();
    
    // Return summary without responses
    const formsList = Object.values(forms).map(form => ({
      id: form.id,
      title: form.title,
      category: form.category,
      created_at: form.created_at,
      questions_count: form.questions.length,
      responses_count: form.responses.length
    }));
    
    res.json({
      success: true,
      forms: formsList
    });
    
  } catch (error) {
    console.error('Error listing forms:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to list forms'
    });
  }
});

module.exports = router;