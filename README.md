# HGC Helper - Educational Insights Platform

An intelligent tutoring system for high school History, Geography, and Civics using RAG (Retrieval-Augmented Generation) technology.

## Features

### For Students
- 💬 Interactive chat interface for asking academic questions
- 📚 Answers based on actual textbook content
- 🎯 Context-aware responses using advanced AI

### For Teachers
- 📊 Dashboard to analyze student questions
- 🔍 Search and filter functionality
- 📈 Insights with word clouds and frequency charts
- 💾 Export data for further analysis

## Architecture

- **RAG Pipeline**: Hybrid retrieval (dense + sparse search) with reranking
- **Vector Database**: ChromaDB for semantic search
- **LLM Support**: OpenAI GPT-4 and Google Gemini
- **Frontend**: Streamlit for intuitive interface

## Prerequisites

- Python 3.8+
- API keys for:
  - OpenAI (for GPT-4 and embeddings)
  - Google Gemini (optional)
  - Cohere (for reranking)

## Installation

1. **Clone the repository**
   ```bash
   cd HCG_Helper
   ```

2. **Install dependencies with uv**
   ```bash
   uv sync
   ```

   This will create a virtual environment and install all dependencies.

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   GOOGLE_API_KEY=your_google_api_key
   COHERE_API_KEY=your_cohere_api_key
   LLM_PROVIDER=openai
   LLM_MODEL=gpt-4
   ```

## Usage

### 1. Ingest Textbook Data

First, you need to process and store the textbook content in the vector database:

```bash
uv run python ingest.py
```

This will:
- Load all `.txt` files from the `data/` directory
- Clean and chunk the text
- Generate embeddings
- Store everything in ChromaDB

**Note**: Your textbook data should be in the `data/` folder as `.txt` files with naming pattern: `Subject_Level.txt` (e.g., `History_L1.txt`, `Civic_L2.txt`)

### 2. Run the Application

```bash
uv run streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### 3. Using the Interface

#### Student Interface
1. Select "Student Chat" from the sidebar
2. Type your question in the chat input
3. Receive AI-generated answers based on textbook content

#### Teacher Dashboard
1. Select "Teacher Dashboard" from the sidebar
2. View all student questions and statistics
3. Search for specific topics
4. Analyze insights with visualizations

## Project Structure

```
HCG_Helper/
├── data/                      # Textbook data files
│   ├── Civic_L1.txt
│   ├── Civic_L2.txt
│   ├── Geo_L1.txt
│   ├── Geo_L2.txt
│   ├── History_L1.txt
│   └── History_L2.txt
├── chroma_db/                 # ChromaDB vector store (auto-created)
├── config.py                  # Configuration settings
├── database.py                # Query logging database
├── engine.py                  # RAG pipeline engine
├── ingest.py                  # Data ingestion script
├── app.py                     # Streamlit application
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore file
├── SPEC.md                   # Project specification
└── README.md                 # This file
```

## Configuration

Edit `config.py` to customize:

- **Chunk settings**: `CHUNK_SIZE`, `CHUNK_OVERLAP`
- **Retrieval settings**: `TOP_K_RETRIEVAL`, `TOP_K_RERANK`
- **Model settings**: `EMBEDDING_MODEL`, `LLM_MODEL`
- **Paths**: `DATA_DIR`, `CHROMA_DB_DIR`

## Technical Details

### RAG Pipeline

1. **Query Input**: Student question received from frontend
2. **Hybrid Retrieval**: Combines vector similarity and keyword matching
3. **Reranking**: Uses Cohere to re-order results by relevance
4. **Generation**: LLM generates contextual answer
5. **Logging**: Question logged to database for teacher analysis

### Data Processing

- Text chunking with semantic splitting
- OpenAI embeddings for vector representation
- ChromaDB for efficient similarity search
- Metadata preservation (subject, level, source)

## Testing

### Test the RAG Engine

```bash
uv run python engine.py
```

This will run a test query and display statistics.

### Test Data Ingestion

```bash
uv run python ingest.py
```

Verify that ChromaDB is populated correctly.

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure all dependencies are installed
   ```bash
   uv sync
   ```

2. **API key errors**: Verify `.env` file has correct API keys

3. **ChromaDB not found**: Run `uv run python ingest.py` first to create the database

4. **Module not found**: Ensure you're in the correct directory. uv handles the virtual environment automatically

## Future Enhancements

- [ ] Multi-language support
- [ ] Voice input/output
- [ ] More advanced analytics for teachers
- [ ] Integration with learning management systems
- [ ] Mobile application
- [ ] Offline mode with local LLM

## License

This project is for educational purposes.

## Support

For issues or questions, please refer to the project documentation or create an issue in the repository.

---

Built with LlamaIndex, Streamlit, and ChromaDB
