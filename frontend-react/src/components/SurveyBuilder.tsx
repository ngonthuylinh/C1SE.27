import React, { useState } from 'react';
import {
  Paper,
  Typography,
  Box,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Checkbox,
  FormControlLabel,
  List,
  ListItem,
  ListItemText,
  IconButton,
  Chip,
  Alert,
  CircularProgress,
  Card,
  CardContent,
  CardActions,
  Divider,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  FormGroup,
  Switch,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material';
import {
  PlaylistAddCheck as SurveyIcon,
  Delete as DeleteIcon,
  PictureAsPdf as PdfIcon,
  Share as ShareIcon,
  ContentCopy as CopyIcon,
  CheckCircle as CheckIcon,
  Send as SendIcon,
  GetApp as DownloadIcon,
  OpenInNew as OpenInNewIcon,
  Add as AddIcon,
  Edit as EditIcon,
  ExpandMore as ExpandMoreIcon
} from '@mui/icons-material';
import { Question } from '../types';
import questionApi from '../api';

interface SurveyBuilderProps {
  questions: Question[];
  keyword: string;
  category: string;
}

interface SurveySettings {
  allowAnonymous: boolean;
  collectEmail: boolean;
  multipleSubmissions: boolean;
  requireAll: boolean;
}

interface CreatedSurvey {
  id: string;
  title: string;
  questionCount: number;
  shareUrl: string;
  pdfUrl: string;
  createdAt: string;
}

const SurveyBuilder: React.FC<SurveyBuilderProps> = ({ questions, keyword, category }) => {
  const [open, setOpen] = useState(false);
  const [activeStep, setActiveStep] = useState(0);
  const [selectedQuestions, setSelectedQuestions] = useState<Question[]>([]);
  const [customQuestions, setCustomQuestions] = useState<Question[]>([]);
  const [newQuestion, setNewQuestion] = useState('');
  const [newQuestionCategory, setNewQuestionCategory] = useState(category || 'general');
  const [surveyTitle, setSurveyTitle] = useState(`${keyword} Survey`);
  const [surveyDescription, setSurveyDescription] = useState('');
  const [settings, setSettings] = useState<SurveySettings>({
    allowAnonymous: true,
    collectEmail: false,
    multipleSubmissions: false,
    requireAll: false
  });
  const [loading, setLoading] = useState(false);
  const [createdSurvey, setCreatedSurvey] = useState<CreatedSurvey | null>(null);
  const [error, setError] = useState('');

  const steps = [
    'Select Questions',
    'Configure Survey', 
    'Settings & Preview',
    'Create Survey'
  ];

  const handleOpen = () => {
    setOpen(true);
    setSelectedQuestions([...questions]); // Select all by default
    setActiveStep(0);
  };

  const handleClose = () => {
    setOpen(false);
    setCreatedSurvey(null);
    setError('');
    setActiveStep(0);
  };

  const handleQuestionSelect = (question: Question, checked: boolean) => {
    if (checked) {
      setSelectedQuestions([...selectedQuestions, question]);
    } else {
      setSelectedQuestions(selectedQuestions.filter(q => q.question !== question.question));
    }
  };

  const handleSelectAll = (checked: boolean) => {
    const allQuestions = getAllQuestions();
    if (checked) {
      setSelectedQuestions([...allQuestions]);
    } else {
      setSelectedQuestions([]);
    }
  };

  const removeQuestion = (questionToRemove: Question) => {
    setSelectedQuestions(selectedQuestions.filter(q => q.question !== questionToRemove.question));
  };

  // Custom questions functions
  const addCustomQuestion = () => {
    if (!newQuestion.trim()) return;
    
    const customQuestion: Question = {
      question: newQuestion.trim(),
      category: newQuestionCategory,
      confidence: 1.0, // User-created questions have 100% confidence
      method: 'user_created',
      source_keyword: 'custom'
    };
    
    setCustomQuestions([...customQuestions, customQuestion]);
    setSelectedQuestions([...selectedQuestions, customQuestion]);
    setNewQuestion('');
  };

  const removeCustomQuestion = (questionToRemove: Question) => {
    setCustomQuestions(customQuestions.filter(q => q.question !== questionToRemove.question));
    setSelectedQuestions(selectedQuestions.filter(q => q.question !== questionToRemove.question));
  };

  const editCustomQuestion = (oldQuestion: Question, newText: string) => {
    const updatedCustomQuestions = customQuestions.map(q => 
      q.question === oldQuestion.question 
        ? { ...q, question: newText.trim() }
        : q
    );
    setCustomQuestions(updatedCustomQuestions);
    
    const updatedSelectedQuestions = selectedQuestions.map(q => 
      q.question === oldQuestion.question 
        ? { ...q, question: newText.trim() }
        : q
    );
    setSelectedQuestions(updatedSelectedQuestions);
  };

  // Get all questions (AI generated + custom)
  const getAllQuestions = () => [...questions, ...customQuestions];

  const handleNext = () => {
    setActiveStep(prev => prev + 1);
  };

  const handleBack = () => {
    setActiveStep(prev => prev - 1);
  };

  const createSurvey = async () => {
    if (selectedQuestions.length === 0) {
      setError('Please select at least one question');
      return;
    }

    setLoading(true);
    setError('');

    try {
      // Prepare survey data
      const surveyData = {
        title: surveyTitle.trim(),
        category: category || 'general',
        description: surveyDescription.trim(),
        questions: selectedQuestions.map(q => ({
          question: q.question,
          type: 'text', // Default type, could be enhanced
          required: settings.requireAll,
          options: null
        })),
        settings
      };

      // Create survey via API
      const response = await fetch('http://localhost:8001/api/survey/create', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(surveyData)
      });

      const result = await response.json();

      if (result.success) {
        setCreatedSurvey(result.form);
        setActiveStep(prev => prev + 1);
      } else {
        throw new Error(result.error || 'Failed to create survey');
      }

    } catch (err: any) {
      console.error('Error creating survey:', err);
      setError(`Failed to create survey: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      // Could add a snackbar notification here
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const downloadPDF = () => {
    if (createdSurvey) {
      window.open(createdSurvey.pdfUrl, '_blank');
    }
  };

  const openSurvey = () => {
    if (createdSurvey) {
      window.open(createdSurvey.shareUrl, '_blank');
    }
  };

  const renderStepContent = (step: number) => {
    switch (step) {
      case 0:
        const allQuestions = getAllQuestions();
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Select Questions for Your Survey
            </Typography>
            <Typography variant="body2" color="textSecondary" sx={{ mb: 3 }}>
              Choose which questions you want to include in your survey form, or add your own custom questions.
            </Typography>

            {/* Add Custom Question Section */}
            <Accordion sx={{ mb: 3 }}>
              <AccordionSummary
                expandIcon={<ExpandMoreIcon />}
                aria-controls="add-custom-question-content"
                id="add-custom-question-header"
              >
                <Typography variant="subtitle1" sx={{ color: 'primary.main', fontWeight: 'bold' }}>
                  <AddIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                  Add Your Own Questions
                </Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    <TextField
                      fullWidth
                      label="Your Question"
                      placeholder="Enter your custom question..."
                      value={newQuestion}
                      onChange={(e) => setNewQuestion(e.target.value)}
                      multiline
                      rows={2}
                    />
                  </Grid>
                  <Grid item xs={12} md={3}>
                    <FormControl fullWidth>
                      <InputLabel>Category</InputLabel>
                      <Select
                        value={newQuestionCategory}
                        onChange={(e) => setNewQuestionCategory(e.target.value)}
                        label="Category"
                      >
                        <MenuItem value="general">General</MenuItem>
                        <MenuItem value="it">IT</MenuItem>
                        <MenuItem value="marketing">Marketing</MenuItem>
                        <MenuItem value="economics">Economics</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>
                  <Grid item xs={12} md={3}>
                    <Button
                      fullWidth
                      variant="contained"
                      onClick={addCustomQuestion}
                      disabled={!newQuestion.trim()}
                      startIcon={<AddIcon />}
                      sx={{ height: '56px' }}
                    >
                      Add Question
                    </Button>
                  </Grid>
                </Grid>
              </AccordionDetails>
            </Accordion>

            {/* Select All Checkbox */}
            <Box sx={{ mb: 2 }}>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={selectedQuestions.length === allQuestions.length && allQuestions.length > 0}
                    indeterminate={selectedQuestions.length > 0 && selectedQuestions.length < allQuestions.length}
                    onChange={(e) => handleSelectAll(e.target.checked)}
                  />
                }
                label={`Select All (${allQuestions.length} questions)`}
              />
            </Box>

            {/* AI Generated Questions */}
            {questions.length > 0 && (
              <>
                <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mb: 1, color: 'text.secondary' }}>
                  AI Generated Questions ({questions.length})
                </Typography>
                <List sx={{ mb: 2 }}>
                  {questions.map((question, index) => (
                    <ListItem key={`ai-${index}`} divider>
                      <Checkbox
                        checked={selectedQuestions.some(q => q.question === question.question)}
                        onChange={(e) => handleQuestionSelect(question, e.target.checked)}
                        sx={{ mr: 1 }}
                      />
                      <ListItemText
                        primary={question.question}
                        secondary={
                          <Box>
                            <Chip size="small" label={question.category} color="primary" sx={{ mr: 1 }} />
                            <Chip size="small" label={`${Math.round(question.confidence * 100)}% confidence`} variant="outlined" />
                            <Chip size="small" label="AI Generated" sx={{ ml: 1, bgcolor: 'rgba(25, 118, 210, 0.1)' }} />
                          </Box>
                        }
                      />
                    </ListItem>
                  ))}
                </List>
              </>
            )}

            {/* Custom Questions */}
            {customQuestions.length > 0 && (
              <>
                <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mb: 1, color: 'text.secondary' }}>
                  Your Custom Questions ({customQuestions.length})
                </Typography>
                <List sx={{ mb: 2 }}>
                  {customQuestions.map((question, index) => (
                    <ListItem key={`custom-${index}`} divider>
                      <Checkbox
                        checked={selectedQuestions.some(q => q.question === question.question)}
                        onChange={(e) => handleQuestionSelect(question, e.target.checked)}
                        sx={{ mr: 1 }}
                      />
                      <ListItemText
                        primary={question.question}
                        secondary={
                          <Box>
                            <Chip size="small" label={question.category} color="secondary" sx={{ mr: 1 }} />
                            <Chip size="small" label="100% confidence" variant="outlined" sx={{ mr: 1 }} />
                            <Chip size="small" label="Custom" sx={{ bgcolor: 'rgba(156, 39, 176, 0.1)' }} />
                          </Box>
                        }
                      />
                      <IconButton
                        onClick={() => removeCustomQuestion(question)}
                        size="small"
                        sx={{ ml: 1 }}
                        color="error"
                      >
                        <DeleteIcon fontSize="small" />
                      </IconButton>
                    </ListItem>
                  ))}
                </List>
              </>
            )}

            {/* Selection Summary */}
            <Alert 
              severity={selectedQuestions.length > 0 ? "success" : "info"} 
              sx={{ mt: 2 }}
            >
              <Typography variant="body2">
                <strong>Selected: {selectedQuestions.length} questions</strong>
                {selectedQuestions.length > 0 && (
                  <>
                    <br />
                    • AI Generated: {selectedQuestions.filter(q => q.method !== 'user_created').length}
                    • Custom: {selectedQuestions.filter(q => q.method === 'user_created').length}
                  </>
                )}
              </Typography>
            </Alert>
          </Box>
        );

      case 1:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Configure Your Survey
            </Typography>

            <TextField
              fullWidth
              label="Survey Title"
              value={surveyTitle}
              onChange={(e) => setSurveyTitle(e.target.value)}
              sx={{ mb: 3 }}
              placeholder="Enter a descriptive title for your survey"
            />

            <TextField
              fullWidth
              multiline
              rows={3}
              label="Survey Description (Optional)"
              value={surveyDescription}
              onChange={(e) => setSurveyDescription(e.target.value)}
              sx={{ mb: 3 }}
              placeholder="Provide context or instructions for survey participants"
            />

            <Typography variant="subtitle1" gutterBottom>
              Selected Questions Preview:
            </Typography>
            <Box sx={{ maxHeight: 300, overflow: 'auto', border: 1, borderColor: 'grey.300', borderRadius: 1, p: 2 }}>
              {selectedQuestions.map((question, index) => (
                <Box key={index} sx={{ mb: 2, p: 1, bgcolor: 'grey.50', borderRadius: 1 }}>
                  <Typography variant="body2" sx={{ fontWeight: 'medium' }}>
                    {index + 1}. {question.question}
                  </Typography>
                  <IconButton
                    size="small"
                    onClick={() => removeQuestion(question)}
                    sx={{ float: 'right', mt: -4 }}
                  >
                    <DeleteIcon fontSize="small" />
                  </IconButton>
                </Box>
              ))}
            </Box>
          </Box>
        );

      case 2:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Survey Settings
            </Typography>

            <FormGroup>
              <FormControlLabel
                control={
                  <Switch
                    checked={settings.allowAnonymous}
                    onChange={(e) => setSettings({...settings, allowAnonymous: e.target.checked})}
                  />
                }
                label="Allow Anonymous Responses"
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={settings.collectEmail}
                    onChange={(e) => setSettings({...settings, collectEmail: e.target.checked})}
                  />
                }
                label="Collect Email Addresses"
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={settings.multipleSubmissions}
                    onChange={(e) => setSettings({...settings, multipleSubmissions: e.target.checked})}
                  />
                }
                label="Allow Multiple Submissions per User"
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={settings.requireAll}
                    onChange={(e) => setSettings({...settings, requireAll: e.target.checked})}
                  />
                }
                label="Make All Questions Required"
              />
            </FormGroup>

            <Divider sx={{ my: 3 }} />

            <Typography variant="subtitle1" gutterBottom>
              Survey Preview
            </Typography>
            <Paper variant="outlined" sx={{ p: 3, bgcolor: 'grey.50' }}>
              <Typography variant="h6" gutterBottom color="primary">
                {surveyTitle}
              </Typography>
              {surveyDescription && (
                <Typography variant="body2" sx={{ mb: 2, color: 'text.secondary' }}>
                  {surveyDescription}
                </Typography>
              )}
              <Typography variant="body2" sx={{ mb: 2 }}>
                <strong>Questions:</strong> {selectedQuestions.length}
              </Typography>
              <Typography variant="body2">
                <strong>Category:</strong> {category}
              </Typography>
            </Paper>
          </Box>
        );

      case 3:
        return (
          <Box textAlign="center">
            {loading ? (
              <Box>
                <CircularProgress size={60} sx={{ mb: 3 }} />
                <Typography variant="h6" gutterBottom>
                  Creating Your Survey...
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Please wait while we generate your survey form and PDF.
                </Typography>
              </Box>
            ) : createdSurvey ? (
              <Box>
                <CheckIcon sx={{ fontSize: 80, color: 'success.main', mb: 2 }} />
                <Typography variant="h5" gutterBottom color="success.main">
                  Survey Created Successfully!
                </Typography>
                
                <Card sx={{ mt: 3, textAlign: 'left' }}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      {createdSurvey.title}
                    </Typography>
                    <Typography variant="body2" color="textSecondary" gutterBottom>
                      Survey ID: {createdSurvey.id}
                    </Typography>
                    <Typography variant="body2" color="textSecondary" gutterBottom>
                      Questions: {createdSurvey.questionCount}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Created: {new Date(createdSurvey.createdAt).toLocaleString()}
                    </Typography>
                  </CardContent>
                  <CardActions>
                    <Button
                      startIcon={<ShareIcon />}
                      variant="contained"
                      onClick={openSurvey}
                      size="small"
                    >
                      Open Survey
                    </Button>
                    <Button
                      startIcon={<CopyIcon />}
                      onClick={() => copyToClipboard(createdSurvey.shareUrl)}
                      size="small"
                    >
                      Copy Link
                    </Button>
                    <Button
                      startIcon={<PdfIcon />}
                      onClick={downloadPDF}
                      size="small"
                    >
                      Download PDF
                    </Button>
                  </CardActions>
                </Card>

                <Box sx={{ mt: 3, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
                  <Typography variant="body2" gutterBottom>
                    <strong>Share URL:</strong>
                  </Typography>
                  <Typography variant="body2" sx={{ wordBreak: 'break-all', color: 'primary.main' }}>
                    {createdSurvey.shareUrl}
                  </Typography>
                </Box>
              </Box>
            ) : (
              <Box>
                <Typography variant="h6" gutterBottom>
                  Ready to Create Survey
                </Typography>
                <Typography variant="body2" color="textSecondary" sx={{ mb: 3 }}>
                  Review your settings and click "Create Survey" to generate your form.
                </Typography>
                
                <Alert severity="info" sx={{ mb: 3 }}>
                  This will create a shareable survey form and generate a PDF version for download.
                </Alert>

                {error && (
                  <Alert severity="error" sx={{ mb: 3 }}>
                    {error}
                  </Alert>
                )}

                <Typography variant="body2" color="textSecondary">
                  Click "Create Survey" button below to proceed.
                </Typography>
              </Box>
            )}
          </Box>
        );

      default:
        return null;
    }
  };

  return (
    <>
      <Button
        variant="outlined"
        startIcon={<SurveyIcon />}
        onClick={handleOpen}
        disabled={questions.length === 0}
        sx={{
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          backgroundClip: 'text',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          borderColor: '#667eea',
          '&:hover': {
            borderColor: '#764ba2',
          },
        }}
      >
        Create Survey Form
      </Button>

      <Dialog 
        open={open} 
        onClose={handleClose}
        maxWidth="md"
        fullWidth
        PaperProps={{
          sx: { minHeight: '600px' }
        }}
      >
        <DialogTitle>
          <Box display="flex" alignItems="center">
            <SurveyIcon sx={{ mr: 1, color: 'primary.main' }} />
            AI Survey Builder
          </Box>
        </DialogTitle>
        
        <DialogContent>
          <Stepper activeStep={activeStep} orientation="vertical">
            {steps.map((label, index) => (
              <Step key={label}>
                <StepLabel>{label}</StepLabel>
                <StepContent>
                  {renderStepContent(index)}
                </StepContent>
              </Step>
            ))}
          </Stepper>
        </DialogContent>

        <DialogActions>
          {activeStep > 0 && (
            <Button onClick={handleBack}>
              Back
            </Button>
          )}
          
          {activeStep < 2 && (
            <Button
              variant="contained"
              onClick={handleNext}
              disabled={activeStep === 0 && selectedQuestions.length === 0}
            >
              {activeStep === 1 ? 'Review & Settings' : 'Next'}
            </Button>
          )}
          
          {activeStep === 2 && !createdSurvey && (
            <Button
              variant="contained"
              onClick={createSurvey}
              disabled={selectedQuestions.length === 0}
              startIcon={<SendIcon />}
              color="primary"
            >
              Create Survey
            </Button>
          )}
          
          {createdSurvey && (
            <>
              <Button
                variant="outlined"
                onClick={downloadPDF}
                startIcon={<DownloadIcon />}
                sx={{ mr: 1 }}
              >
                Download PDF
              </Button>
              <Button
                variant="outlined"
                onClick={openSurvey}
                startIcon={<OpenInNewIcon />}
              >
                Open Survey
              </Button>
            </>
          )}
          
          <Button onClick={handleClose}>
            {createdSurvey ? 'Close' : 'Cancel'}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default SurveyBuilder;