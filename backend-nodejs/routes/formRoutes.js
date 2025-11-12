const express = require('express');
const router = express.Router();
const path = require('path');
const fs = require('fs-extra');

// Serve survey form page
router.get('/:id', async (req, res) => {
    try {
        const formId = req.params.id;
        const formsDir = path.join(__dirname, '../forms');
        const formFile = path.join(formsDir, `${formId}.json`);
        
        if (!await fs.pathExists(formFile)) {
            return res.status(404).send(`
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Survey Not Found</title>
                    <meta charset="utf-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1">
                    <style>
                        body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
                        .error { color: #f44336; }
                    </style>
                </head>
                <body>
                    <h1 class="error">Survey Not Found</h1>
                    <p>The survey you're looking for doesn't exist or has been removed.</p>
                </body>
                </html>
            `);
        }
        
        const formData = await fs.readJson(formFile);
        
        // Generate HTML form
        const html = generateFormHTML(formData, req);
        res.send(html);
        
    } catch (error) {
        console.error('Error serving form:', error);
        res.status(500).send('Internal server error');
    }
});

function generateFormHTML(formData, req) {
    const baseUrl = `${req.protocol}://${req.get('host')}`;
    
    return `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${formData.title}</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            font-weight: 300;
        }
        
        .header .description {
            opacity: 0.9;
            font-size: 1.1rem;
        }
        
        .form-content {
            padding: 40px;
        }
        
        .question-block {
            margin-bottom: 30px;
            padding: 20px;
            border-radius: 8px;
            background: #f8f9fa;
            border-left: 4px solid #667eea;
        }
        
        .question-title {
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 15px;
            color: #333;
        }
        
        .required {
            color: #e74c3c;
        }
        
        .form-control {
            width: 100%;
            padding: 12px;
            border: 2px solid #e9ecef;
            border-radius: 6px;
            font-size: 1rem;
            transition: border-color 0.3s ease;
        }
        
        .form-control:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        .radio-group, .checkbox-group {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        
        .radio-option, .checkbox-option {
            display: flex;
            align-items: center;
            padding: 8px 0;
        }
        
        .radio-option input, .checkbox-option input {
            margin-right: 10px;
            transform: scale(1.2);
        }
        
        .rating-group {
            display: flex;
            gap: 15px;
            align-items: center;
        }
        
        .rating-option {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 5px;
        }
        
        .submit-section {
            text-align: center;
            padding: 30px 40px;
            background: #f8f9fa;
            border-top: 1px solid #dee2e6;
        }
        
        .submit-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 40px;
            border: none;
            border-radius: 8px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease;
        }
        
        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .submit-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .success-message {
            display: none;
            padding: 20px;
            background: #d4edda;
            color: #155724;
            border-radius: 8px;
            margin: 20px;
            text-align: center;
            font-weight: 600;
        }
        
        .error-message {
            display: none;
            padding: 20px;
            background: #f8d7da;
            color: #721c24;
            border-radius: 8px;
            margin: 20px;
            text-align: center;
            font-weight: 600;
        }
        
        @media (max-width: 768px) {
            .container {
                margin: 10px;
            }
            
            .header {
                padding: 30px 20px;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .form-content {
                padding: 20px;
            }
            
            .submit-section {
                padding: 20px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>${formData.title}</h1>
            ${formData.description ? `<div class="description">${formData.description}</div>` : ''}
        </div>
        
        <div class="success-message" id="successMessage">
            Thank you! Your response has been submitted successfully.
        </div>
        
        <div class="error-message" id="errorMessage">
            There was an error submitting your response. Please try again.
        </div>
        
        <form id="surveyForm" class="form-content">
            ${generateQuestionsHTML(formData.questions)}
            
            <div class="submit-section">
                <button type="submit" class="submit-btn" id="submitBtn">
                    Submit Survey
                </button>
            </div>
        </form>
    </div>

    <script>
        document.getElementById('surveyForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const submitBtn = document.getElementById('submitBtn');
            const successMessage = document.getElementById('successMessage');
            const errorMessage = document.getElementById('errorMessage');
            
            // Hide previous messages
            successMessage.style.display = 'none';
            errorMessage.style.display = 'none';
            
            // Disable submit button
            submitBtn.disabled = true;
            submitBtn.textContent = 'Submitting...';
            
            // Collect form data
            const formData = new FormData(this);
            const responses = {};
            
            for (let [key, value] of formData.entries()) {
                if (responses[key]) {
                    // Handle multiple values (checkboxes)
                    if (!Array.isArray(responses[key])) {
                        responses[key] = [responses[key]];
                    }
                    responses[key].push(value);
                } else {
                    responses[key] = value;
                }
            }
            
            try {
                const response = await fetch('${baseUrl}/api/survey/${formData.id}/submit', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        responses: responses,
                        metadata: {
                            userAgent: navigator.userAgent,
                            timestamp: new Date().toISOString()
                        }
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    successMessage.style.display = 'block';
                    this.style.display = 'none';
                } else {
                    throw new Error(result.error || 'Submission failed');
                }
                
            } catch (error) {
                console.error('Error:', error);
                errorMessage.style.display = 'block';
                
                // Re-enable submit button
                submitBtn.disabled = false;
                submitBtn.textContent = 'Submit Survey';
            }
        });
    </script>
</body>
</html>`;
}

function generateQuestionsHTML(questions) {
    return questions.map(q => {
        const required = q.required ? '<span class="required">*</span>' : '';
        const questionTitle = `<div class="question-title">${q.question} ${required}</div>`;
        
        let inputHTML = '';
        
        switch (q.type) {
            case 'multiple_choice':
                inputHTML = `
                    <div class="radio-group">
                        ${q.options.map((option, idx) => `
                            <div class="radio-option">
                                <input type="radio" id="q${q.id}_${idx}" name="question_${q.id}" value="${option}" ${q.required ? 'required' : ''}>
                                <label for="q${q.id}_${idx}">${option}</label>
                            </div>
                        `).join('')}
                    </div>
                `;
                break;
                
            case 'checkbox':
                inputHTML = `
                    <div class="checkbox-group">
                        ${q.options.map((option, idx) => `
                            <div class="checkbox-option">
                                <input type="checkbox" id="q${q.id}_${idx}" name="question_${q.id}" value="${option}">
                                <label for="q${q.id}_${idx}">${option}</label>
                            </div>
                        `).join('')}
                    </div>
                `;
                break;
                
            case 'rating':
                inputHTML = `
                    <div class="rating-group">
                        ${[1, 2, 3, 4, 5].map(rating => `
                            <div class="rating-option">
                                <input type="radio" id="q${q.id}_${rating}" name="question_${q.id}" value="${rating}" ${q.required ? 'required' : ''}>
                                <label for="q${q.id}_${rating}">${rating}</label>
                            </div>
                        `).join('')}
                    </div>
                `;
                break;
                
            case 'yes_no':
                inputHTML = `
                    <div class="radio-group">
                        <div class="radio-option">
                            <input type="radio" id="q${q.id}_yes" name="question_${q.id}" value="Yes" ${q.required ? 'required' : ''}>
                            <label for="q${q.id}_yes">Yes</label>
                        </div>
                        <div class="radio-option">
                            <input type="radio" id="q${q.id}_no" name="question_${q.id}" value="No" ${q.required ? 'required' : ''}>
                            <label for="q${q.id}_no">No</label>
                        </div>
                    </div>
                `;
                break;
                
            case 'textarea':
                inputHTML = `<textarea class="form-control" name="question_${q.id}" rows="4" placeholder="Enter your answer..." ${q.required ? 'required' : ''}></textarea>`;
                break;
                
            default: // text
                inputHTML = `<input type="text" class="form-control" name="question_${q.id}" placeholder="Enter your answer..." ${q.required ? 'required' : ''}>`;
                break;
        }
        
        return `
            <div class="question-block">
                ${questionTitle}
                ${inputHTML}
            </div>
        `;
    }).join('');
}

module.exports = router;