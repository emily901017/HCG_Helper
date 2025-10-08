# HGC Helper - Flask + React Setup Guide

This project has been restructured to use Flask as the backend API and React as the frontend.

## Project Structure

```
HCG_Helper/
├── backend/
│   ├── api.py              # Flask REST API
│   └── __init__.py
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── components/
│   │   │   ├── StudentChat.js
│   │   │   ├── StudentChat.css
│   │   │   ├── TeacherDashboard.js
│   │   │   └── TeacherDashboard.css
│   │   ├── App.js
│   │   ├── App.css
│   │   ├── index.js
│   │   └── index.css
│   ├── package.json
│   └── .env.example
├── data/                   # Textbook data
├── chroma_db/             # Vector database
├── engine.py              # RAG engine
├── database.py            # Query logger
├── ingest.py              # Data ingestion
├── config.py              # Configuration
└── requirements.txt       # Python dependencies
```

## Setup Instructions

### 1. Backend Setup (Flask)

#### Install Python Dependencies

```bash
# Make sure you're in the project root directory
pip install -r requirements.txt
```

#### Set up Environment Variables

Create a `.env` file in the project root:

```bash
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Cohere (for reranking)
COHERE_API_KEY=your_cohere_api_key

# Optional: Google Gemini
GOOGLE_API_KEY=your_google_api_key

# LLM Configuration
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
EMBEDDING_MODEL=text-embedding-3-small
```

#### Ingest Data (First Time Only)

```bash
python ingest.py
```

#### Run the Flask Backend

```bash
cd backend
python api.py
```

The backend will run on `http://localhost:5000`

### 2. Frontend Setup (React)

#### Install Node Dependencies

```bash
cd frontend
npm install
```

#### Configure API URL

Create a `.env` file in the `frontend` directory:

```bash
REACT_APP_API_URL=http://localhost:5000
```

#### Run the React Development Server

```bash
npm start
```

The frontend will run on `http://localhost:3000`

## API Endpoints

### Student Chat

- `POST /api/chat/query` - Submit a question and get an answer
  - Request: `{ "question": str, "session_id": str (optional) }`
  - Response: `{ "answer": str, "session_id": str }`

- `GET /api/chat/health` - Health check endpoint

### Teacher Dashboard

- `GET /api/teacher/queries` - Get all student queries
- `GET /api/teacher/statistics` - Get query statistics
- `POST /api/teacher/search` - Search queries by keyword
  - Request: `{ "keyword": str }`
- `GET /api/teacher/keywords?limit=50` - Get common keywords

## Development Workflow

### Running Both Servers

You need to run both the Flask backend and React frontend simultaneously:

**Terminal 1 (Backend):**
```bash
cd backend
python api.py
```

**Terminal 2 (Frontend):**
```bash
cd frontend
npm start
```

### Making Changes

- **Backend changes**: Modify files in `backend/`, `engine.py`, `database.py`, etc.
- **Frontend changes**: Modify files in `frontend/src/`

The React development server will automatically reload on file changes.

## Production Deployment

### Build Frontend

```bash
cd frontend
npm run build
```

This creates optimized production files in `frontend/build/`

### Serve with Flask

You can serve the React build from Flask by updating `backend/api.py`:

```python
from flask import send_from_directory

# Serve React build
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

app.static_folder = '../frontend/build'
```

Then run with a production WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 backend.api:app
```

## Features

### Student Interface
- Real-time chat with AI tutor
- Session-based conversation tracking
- Clean, responsive UI
- Loading indicators
- Clear chat history option

### Teacher Dashboard
- View all student questions
- Statistics overview (total queries, recent queries, unique sessions)
- Subject distribution charts
- Search functionality
- Common keywords analysis
- Timeline visualization
- Export to CSV

## Technology Stack

### Backend
- **Flask**: Web framework
- **Flask-CORS**: Cross-origin resource sharing
- **LlamaIndex**: RAG pipeline orchestration
- **ChromaDB**: Vector database
- **OpenAI/Gemini**: LLM providers
- **Cohere**: Reranking

### Frontend
- **React**: UI library
- **React Router**: Navigation
- **Axios**: HTTP client
- **Recharts**: Data visualization
- **CSS3**: Styling

## Troubleshooting

### CORS Errors
Make sure Flask-CORS is installed and enabled in `backend/api.py`

### API Connection Issues
- Verify both servers are running
- Check the API URL in `frontend/.env`
- Ensure no firewall is blocking ports 3000 or 5000

### Data Not Loading
- Make sure you've run `python ingest.py` to populate the vector database
- Check that ChromaDB directory exists and is accessible
- Verify API keys are set in `.env`

## Original Streamlit App

The original Streamlit application is still available in `app.py` if you need to reference it or run it:

```bash
streamlit run app.py
```

## Next Steps

1. Add authentication for the teacher dashboard
2. Implement rate limiting
3. Add more advanced analytics
4. Deploy to production (e.g., Heroku, AWS, DigitalOcean)
5. Add user accounts and profiles
6. Implement real-time updates with WebSockets
