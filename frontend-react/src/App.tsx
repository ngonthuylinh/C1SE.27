import React, { useState, useEffect } from 'react';
import {
  Container,
  Paper,
  Typography,
  TextField,
  Button,
  Grid,
  Card,
  CardContent,
  Chip,
  Box,
  Alert,
  CircularProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Avatar,
  LinearProgress
} from '@mui/material';
import {
  Psychology as BrainIcon,
  AutoAwesome as SparkleIcon,
  Category as CategoryIcon,
  Speed as SpeedIcon,
  CheckCircle as CheckIcon,
  Error as ErrorIcon
} from '@mui/icons-material';
import questionApi from './api';
import { Question, ModelInfo } from './types';
import SurveyBuilder from './components/SurveyBuilder';

const App: React.FC = () => {
  // State management
  const [keyword, setKeyword] = useState('');
  const [numQuestions, setNumQuestions] = useState(5);
  const [selectedCategory, setSelectedCategory] = useState('');
  const [questions, setQuestions] = useState<Question[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [executionTime, setExecutionTime] = useState<number>(0);

  // Check connection and load model info on mount
  useEffect(() => {
    checkConnectionAndLoadInfo();
  }, []);

  const checkConnectionAndLoadInfo = async () => {
    try {
      const [health, info] = await Promise.all([
        questionApi.checkHealth(),
        questionApi.getModelInfo()
      ]);
      
      setIsConnected(health.model_loaded);
      setModelInfo(info);
      setError('');
    } catch (err: any) {
      setIsConnected(false);
      setError(`Connection failed: ${err.message}`);
    }
  };

  const handleGenerateQuestions = async () => {
    if (!keyword.trim()) {
      setError('Please enter a keyword');
      return;
    }

    setLoading(true);
    setError('');
    setQuestions([]);

    try {
      const response = await questionApi.generateQuestions({
        keyword: keyword.trim(),
        num_questions: numQuestions,
        ...(selectedCategory && { category: selectedCategory })
      });

      setQuestions(response.questions);
      setExecutionTime(response.execution_time);
    } catch (err: any) {
      setError(`Failed to generate questions: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const getCategoryColor = (category: string) => {
    const colors: Record<string, string> = {
      'marketing': '#1976d2',
      'economics': '#388e3c',
      'it': '#f57c00',
      'nan': '#9e9e9e'
    };
    return colors[category] || '#9e9e9e';
  };

  const getMethodIcon = (method: string) => {
    switch (method) {
      case 'direct_match':
        return '🎯';
      case 'similarity_adaptation':
        return '🔗';
      case 'category_adaptation':
        return '📂';
      default:
        return '⚡';
    }
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Header */}
      <Paper 
        elevation={3} 
        sx={{ 
          p: 4, 
          mb: 4, 
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          color: 'white',
          textAlign: 'center'
        }}
      >
        <Box display="flex" alignItems="center" justifyContent="center" mb={2}>
          <Avatar sx={{ bgcolor: 'white', color: '#667eea', mr: 2, width: 56, height: 56 }}>
            <BrainIcon fontSize="large" />
          </Avatar>
          <Typography variant="h3" component="h1" fontWeight="bold">
            Form Agent AI
          </Typography>
        </Box>
        <Typography variant="h6" sx={{ opacity: 0.9, mb: 3 }}>
          AI-Powered Question Generator using Advanced Machine Learning
        </Typography>

        {/* Connection Status */}
        <Box 
          display="flex" 
          alignItems="center" 
          justifyContent="center" 
          p={2} 
          borderRadius={2}
          sx={{ 
            backgroundColor: isConnected ? 'rgba(76, 175, 80, 0.2)' : 'rgba(244, 67, 54, 0.2)',
            border: `1px solid ${isConnected ? '#4caf50' : '#f44336'}`
          }}
        >
          {isConnected ? <CheckIcon sx={{ mr: 1 }} /> : <ErrorIcon sx={{ mr: 1 }} />}
          <Typography variant="body1">
            {isConnected ? 'Connected - Model Ready' : 'Connection Issues'}
          </Typography>
        </Box>

        {/* Model Statistics - Hidden */}
        {/* {modelInfo && (
          <Grid container spacing={2} sx={{ mt: 2 }}>
            <Grid item xs={12} sm={4}>
              <Box textAlign="center">
                <Typography variant="h4" fontWeight="bold">
                  {modelInfo.total_questions?.toLocaleString()}
                </Typography>
                <Typography variant="body2">Questions</Typography>
              </Box>
            </Grid>
            <Grid item xs={12} sm={4}>
              <Box textAlign="center">
                <Typography variant="h4" fontWeight="bold">
                  {modelInfo.categories?.length}
                </Typography>
                <Typography variant="body2">Categories</Typography>
              </Box>
            </Grid>
            <Grid item xs={12} sm={4}>
              <Box textAlign="center">
                <Typography variant="h4" fontWeight="bold">
                  {modelInfo.total_keywords?.toLocaleString()}
                </Typography>
                <Typography variant="body2">Keywords</Typography>
              </Box>
            </Grid>
          </Grid>
        )} */}
      </Paper>

      {/* Main Form */}
      <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
        <Typography variant="h5" gutterBottom display="flex" alignItems="center">
          <SparkleIcon sx={{ mr: 1, color: '#667eea' }} />
          Generate Questions
        </Typography>

        <Grid container spacing={3}>
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Keyword"
              value={keyword}
              onChange={(e) => setKeyword(e.target.value)}
              placeholder="Enter a keyword (e.g., artificial intelligence, marketing, economics)"
              disabled={loading || !isConnected}
              onKeyPress={(e) => {
                if (e.key === 'Enter') {
                  handleGenerateQuestions();
                }
              }}
            />
          </Grid>

          <Grid item xs={12} sm={6}>
            <FormControl fullWidth disabled={loading || !isConnected}>
              <InputLabel>Number of Questions</InputLabel>
              <Select
                value={numQuestions}
                onChange={(e) => setNumQuestions(e.target.value as number)}
                label="Number of Questions"
              >
                {[1, 3, 5, 7, 10, 15, 20].map((num) => (
                  <MenuItem key={num} value={num}>
                    {num} questions
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} sm={6}>
            <FormControl fullWidth disabled={loading || !isConnected}>
              <InputLabel>Category (Optional)</InputLabel>
              <Select
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
                label="Category (Optional)"
              >
                <MenuItem value="">Auto Detect</MenuItem>
                {modelInfo?.categories?.map((category) => (
                  <MenuItem key={category} value={category}>
                    {category.charAt(0).toUpperCase() + category.slice(1)}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12}>
            {error && (
              <Alert severity="error" sx={{ mb: 2 }}>
                {error}
              </Alert>
            )}

            <Button
              variant="contained"
              size="large"
              fullWidth
              onClick={handleGenerateQuestions}
              disabled={loading || !isConnected || !keyword.trim()}
              startIcon={loading ? <CircularProgress size={20} /> : <SparkleIcon />}
              sx={{
                py: 2,
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                '&:hover': {
                  background: 'linear-gradient(135deg, #5a67d8 0%, #667eea 100%)',
                },
              }}
            >
              {loading ? 'Generating Questions...' : 'Generate Questions'}
            </Button>
          </Grid>
        </Grid>
      </Paper>

      {/* Loading Progress */}
      {loading && (
        <Box sx={{ mb: 4 }}>
          <LinearProgress />
        </Box>
      )}

      {/* Results */}
      {questions.length > 0 && (
        <Paper elevation={2} sx={{ p: 4 }}>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
            <Typography variant="h5" display="flex" alignItems="center">
              <CategoryIcon sx={{ mr: 1, color: '#667eea' }} />
              Generated Questions ({questions.length})
            </Typography>
            <Box display="flex" alignItems="center" gap={2}>
              <SurveyBuilder 
                questions={questions} 
                keyword={keyword} 
                category={selectedCategory || 'general'} 
              />
              <Box display="flex" alignItems="center">
                <SpeedIcon sx={{ mr: 1, color: '#667eea' }} />
                <Typography variant="body2" color="text.secondary">
                  {executionTime}ms
                </Typography>
              </Box>
            </Box>
          </Box>

          <Grid container spacing={3}>
            {questions.map((question, index) => (
              <Grid item xs={12} md={6} key={index}>
                <Card 
                  elevation={1} 
                  sx={{ 
                    height: '100%',
                    transition: 'transform 0.2s ease-in-out',
                    '&:hover': {
                      transform: 'translateY(-2px)',
                      boxShadow: 3,
                    },
                  }}
                >
                  <CardContent>
                    {/* Header */}
                    <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                      <Typography 
                        variant="h6" 
                        color="primary" 
                        sx={{ fontWeight: 'bold' }}
                      >
                        #{index + 1}
                      </Typography>
                      <Box display="flex" gap={1} alignItems="center">
                        <Chip
                          label={question.category}
                          size="small"
                          sx={{
                            backgroundColor: getCategoryColor(question.category),
                            color: 'white',
                            fontWeight: 'bold',
                          }}
                        />
                        <Chip
                          label={`${Math.round(question.confidence * 100)}%`}
                          size="small"
                          variant="outlined"
                          color="success"
                        />
                      </Box>
                    </Box>

                    {/* Question */}
                    <Typography 
                      variant="body1" 
                      sx={{ 
                        mb: 2, 
                        fontWeight: 500,
                        lineHeight: 1.6,
                        minHeight: '3em'
                      }}
                    >
                      {question.question}
                    </Typography>

                    {/* Metadata */}
                    <Box 
                      display="flex" 
                      justifyContent="space-between" 
                      alignItems="center"
                      pt={2}
                      borderTop={1}
                      borderColor="grey.200"
                    >
                      <Box display="flex" alignItems="center">
                        <Typography variant="body2" color="text.secondary">
                          {getMethodIcon(question.method)} {question.method.replace('_', ' ')}
                        </Typography>
                      </Box>
                      {question.source_keyword && (
                        <Typography variant="body2" color="text.secondary">
                          From: "{question.source_keyword}"
                        </Typography>
                      )}
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Paper>
      )}

      {/* Reconnection Section */}
      {!isConnected && (
        <Paper elevation={2} sx={{ p: 4, mt: 4 }}>
          <Alert severity="warning" sx={{ mb: 2 }}>
            Connection to backend server lost. Please check:
          </Alert>
          <ul style={{ marginLeft: 20 }}>
            <li>Backend server is running at http://localhost:8001</li>
            <li>Model is properly loaded</li>
            <li>No firewall or CORS issues</li>
          </ul>
          <Button 
            variant="outlined" 
            onClick={checkConnectionAndLoadInfo}
            sx={{ mt: 2 }}
          >
            Retry Connection
          </Button>
        </Paper>
      )}
    </Container>
  );
};

export default App;