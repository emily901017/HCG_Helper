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

## Evaluation System

The project includes a comprehensive RAG evaluation system located in the `eval/` directory to assess and improve the quality of the AI tutor's responses.

### Evaluation Pipeline

The evaluation system consists of three main components:

1. **Document Chunking** (`chunker.py`) - Splits textbook content into manageable chunks
2. **Q&A Generation** (`qa_generator.py`) - Uses GPT-4o to generate question-answer pairs from chunks
3. **RAG Evaluation** (`evaluator.py`) - Evaluates the RAG pipeline's performance using multiple metrics

### Evaluation Metrics

The system evaluates answers using three binary metrics:

- **Faithfulness**: Does the answer only contain information from the retrieved context?
- **Relevance**: Does the answer address the question?
- **Correctness**: Does the answer accurately match the ground truth?

Each metric returns `true` or `false`, and aggregate metrics show the percentage of passing samples.

### Evaluation Dataset

The evaluation dataset (`eval_dataset.json`) contains:
- **258 samples** across 6 textbook files (Civic L1/L2, Geography L1/L2, History L1/L2)
- Each sample includes:
  - Question (student-like queries)
  - Ground truth answer (extracted from textbook)
  - Ground truth context (source chunk)
  - Source file and chunk ID metadata

### Running Evaluations

#### Generate Evaluation Dataset

```bash
# Generate Q&A pairs from textbook data
python eval/run_evaluation.py generate --data-dir ./data --output eval_data/eval_dataset.json --qa-pairs 1
```

Parameters:
- `--data-dir`: Directory containing textbook files (.txt)
- `--output`: Output path for evaluation dataset
- `--chunk-size`: Chunk size in characters (default: 2048)
- `--qa-pairs`: Number of Q&A pairs per chunk (default: 1)

#### Run Full Evaluation

```bash
# Evaluate RAG pipeline on the dataset
python eval/run_evaluation.py evaluate --dataset eval_data/eval_dataset.json --output eval_data/evaluation_result.json
```

**Note**: Full evaluation with 258 samples takes ~28 minutes due to API rate limits (6.5s delay between samples for Cohere reranking).

#### Quick Testing (10 Samples)

```bash
# Test with a subset for quick validation
python eval/test_evaluator.py
```

This evaluates 10 samples without retrieval evaluation, completing in ~1 minute.


### Evaluation Files Structure

```
eval/
├── chunker.py                  # Document chunking
├── qa_generator.py             # Q&A pair generation
├── evaluator.py                # RAG evaluation metrics
├── run_evaluation.py           # Main evaluation pipeline
├── test_evaluator.py           # Quick testing (10 samples)
```

### Configuration

Evaluation uses the same configuration from `config.py`:
- `LLM_MODEL`: Model for Q&A generation and evaluation (default: gpt-4o)
- `RERANKER_TYPE`: Reranker for RAG pipeline (cohere or qwen)
- API keys from `.env` file

### Best Practices

1. **Start with small datasets** - Use `--qa-pairs 1` to generate one question per chunk
2. **Test before full evaluation** - Run `test_evaluator.py` to verify setup
3. **Monitor API costs** - 258 samples = ~516 API calls (2 per sample: RAG + evaluation)
4. **Use filtering strategically** - Remove clear failures to focus on edge cases
5. **Cross-validate critical failures** - Use Gemini double-check for disputed samples

## Next Steps

### Immediate Improvements
1. Add authentication for the teacher dashboard
2. Implement rate limiting
3. Add more advanced analytics
4. Deploy to production (e.g., Heroku, AWS, DigitalOcean)
5. Add user accounts and profiles
6. Implement real-time updates with WebSockets
7. Migrate to UV for faster dependency management and environment setup

### Future Research Features

This project will be enhanced through three progressive research stages:

#### Stage 1: Learning Diagnosis Module (學習診斷模組)
- **Data Accumulation & Preprocessing**: Continuously collect and anonymize student query data through HGC Helper
- **Semantic Topic Discovery**: Apply clustering algorithms (DBSCAN) to group semantically similar questions regardless of wording variations
- **Misconception Labeling**: Use LLM to analyze question clusters and generate interpretable "core misconception" labels for teachers (e.g., "Confusing 'Separation of Powers' with 'Government System'")

#### Stage 2: Personalized Content Generation Module (個人化內容生成模組)
- **Trigger Mechanism**: Automatically detect when a student's new question maps to a known misconception pattern
- **Dynamic Prompting**: Design prompt templates that combine the student's original question with identified misconceptions to guide targeted LLM responses
- **Scaffolded Content Generation**: Generate multi-level support materials including:
  - Targeted concept clarification
  - Relevant case studies and examples
  - 1-2 practice questions to reinforce understanding

#### Stage 3: Cross-Disciplinary Knowledge Graph Enhanced Retrieval (跨學科知識圖譜增強檢索)
- **Knowledge Graph Construction**: Use LLM to analyze the history, geography, and civics knowledge base to automatically extract:
  - Key entities (e.g., Zheng Chenggong, Age of Exploration, International Law)
  - Inter-entity relationships (e.g., "occurred during", "influenced", "geographic context")
  - Cross-disciplinary connections spanning all three subjects
- **Graph RAG Implementation**: Enhance retrieval by:
  - Performing traditional vector search
  - Simultaneously locating relevant entities in the knowledge graph
  - Expanding search to connected nodes across different subjects
- **Multi-Dimensional Answer Synthesis**: Combine:
  - Traditional text chunk retrieval
  - Cross-disciplinary information traversed from the knowledge graph
  - Provide integrated context to LLM for generating answers with historical depth, geographic breadth, and civic perspectives

These enhancements will transform HGC Helper from a question-answering system into an intelligent tutoring system capable of diagnosing learning gaps, personalizing content, and facilitating cross-disciplinary understanding.
