# Form Agent AI Frontend (React)

🚀 Modern React frontend for AI-powered question generation system.

## Features

- **Modern UI**: Built with Material-UI (MUI) components
- **TypeScript**: Full type safety and IntelliSense
- **Real-time Connection**: Monitor backend connection status
- **AI Integration**: Seamless integration with Node.js backend
- **Responsive Design**: Works on desktop and mobile
- **Advanced Analytics**: View model statistics and performance metrics

## Tech Stack

- React 18
- TypeScript
- Material-UI (MUI)
- Axios for API calls
- Emotion for styling

## Quick Start

1. **Install dependencies**:
   ```bash
   cd frontend-react
   npm install
   ```

2. **Start development server**:
   ```bash
   npm start
   ```

3. **Open browser**: http://localhost:3000

## API Integration

The frontend connects to Node.js backend at `http://localhost:8000`:

- `GET /api/health` - Check server status
- `GET /api/model/info` - Get AI model information
- `POST /api/questions/generate` - Generate questions from keyword
- `POST /api/predict/category` - Predict keyword category

## Environment Variables

Create `.env` file:

```
REACT_APP_API_URL=http://localhost:8000
```

## Project Structure

```
frontend-react/
├── public/
│   └── index.html
├── src/
│   ├── App.tsx          # Main application component
│   ├── index.tsx        # Entry point
│   ├── api.ts           # API service layer
│   ├── types.ts         # TypeScript interfaces
│   └── react-app-env.d.ts
├── package.json
└── tsconfig.json
```

## Usage

1. **Enter Keyword**: Type any keyword (e.g., "artificial intelligence")
2. **Select Options**: Choose number of questions and category
3. **Generate**: Click generate button to create AI questions
4. **View Results**: See generated questions with confidence scores and categories

## Development

```bash
# Install dependencies
npm install

# Start dev server
npm start

# Build for production
npm run build

# Run tests
npm test
```

## Backend Integration

Make sure Node.js backend is running:

```bash
cd backend-nodejs
npm start
```

The backend should be accessible at http://localhost:8000 with the trained AI model loaded.

## Troubleshooting

1. **Connection Issues**: Check if backend server is running
2. **CORS Errors**: Ensure backend allows CORS from localhost:3000
3. **Model Not Loaded**: Verify .pkl model file exists in backend
4. **API Errors**: Check backend logs for detailed error messages

---

🤖 **Form Agent AI** - Powered by Advanced Machine Learning