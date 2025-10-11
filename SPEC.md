# Project Specification: HGC Helper - Educational Insights Platform

## 1. Project Overview

**HGC Helper** is an intelligent tutoring system for high school History, Geography, and Civics education. It leverages Retrieval-Augmented Generation (RAG) to provide accurate, context-aware answers to student questions while collecting learning analytics for teachers.

### Key Features
- AI-powered question answering with source citations
- Teacher dashboard with query analytics and insights
- Hybrid retrieval system (vector + keyword search)
- Multi-model LLM support (OpenAI, Gemini)
- Comprehensive evaluation framework

---

## 2. User Personas

### 2.1. Student
- **Primary Goal**: Get accurate answers to social studies questions
- **Needs**:
  - Simple, intuitive chat interface
  - Clear, well-sourced answers
  - Context from textbooks with citations
  - Session-based conversation tracking

### 2.2. Teacher
- **Primary Goal**: Understand student learning difficulties and common misconceptions
- **Needs**:
  - Overview of student questions and patterns
  - Analytics on query frequency and topics
  - Search and filter capabilities
  - Export functionality for further analysis
  - Visualization of trends over time

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     React Frontend                           │
│  ┌──────────────────┐      ┌──────────────────┐            │
│  │  Student Chat    │      │ Teacher Dashboard │            │
│  │  - Q&A Interface │      │ - Analytics       │            │
│  │  - Source Display│      │ - Query Search    │            │
│  └──────────────────┘      └──────────────────┘            │
└─────────────────┬───────────────────┬───────────────────────┘
                  │                   │
            REST API (CORS enabled)   │
                  │                   │
┌─────────────────▼───────────────────▼───────────────────────┐
│                    Flask Backend API                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ /api/chat/*  │  │/api/teacher/*│  │ Query Logger │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────┬───────────────────────────────────────────┘
                  │
       ┌──────────▼──────────┐
       │    RAG Engine       │
       │  ┌───────────────┐  │
       │  │ Hybrid Search │  │
       │  │ Vector + BM25 │  │
       │  └───────┬───────┘  │
       │          │           │
       │  ┌───────▼───────┐  │
       │  │   Reranking   │  │
       │  │ (Cohere/Qwen) │  │
       │  └───────┬───────┘  │
       │          │           │
       │  ┌───────▼───────┐  │
       │  │ LLM Synthesis │  │
       │  │ (GPT-4/Gemini)│  │
       │  └───────────────┘  │
       └─────────────────────┘
                  │
       ┌──────────▼──────────┐
       │     ChromaDB        │
       │  Vector Database    │
       └─────────────────────┘
```

---

## 4. Core Functional Requirements

### 4.1. Student Interface

#### Chat Interface
- **Input**: Text box for question submission
- **Output**:
  - AI-generated answer
  - Source references with:
    - Subject and level (e.g., "History - L1")
    - Source filename
    - Text preview (first 200 characters)
    - Numbered citations
- **Features**:
  - Session-based conversation memory (configurable rounds)
  - Clear chat history option
  - Loading indicators
  - Responsive design

#### Conversation Memory
- Maintains last N conversation rounds per session
- Each round = 1 user message + 1 assistant message
- Configurable via `MAX_MEMORY_ROUNDS` in config
- Automatic cleanup of oldest messages

### 4.2. Teacher Dashboard

#### Query Analytics
- **All Queries View**: Searchable, filterable table of student questions
- **Statistics Overview**:
  - Total queries
  - Recent queries (last 7 days)
  - Queries by subject
  - Unique sessions
- **Visualizations**:
  - Timeline of query frequency
  - Subject distribution charts
  - Common keywords word cloud
- **Search & Filter**:
  - Keyword search
  - Date range filtering
  - Subject filtering
- **Export**: CSV export functionality

### 4.3. Data Ingestion Pipeline

**Offline process** for textbook content preparation:

1. **Load**: Read `.txt` files from `data/` directory
   - Civic L1/L2
   - Geography L1/L2
   - History L1/L2

2. **Clean**: Remove artifacts, headers, footers

3. **Chunk**: Split into semantic chunks
   - Default size: 512 characters
   - Overlap: 50 characters

4. **Embed**: Generate vector embeddings
   - Model: `text-embedding-3-small`

5. **Store**: Ingest into ChromaDB
   - Collection: "textbook_knowledge"
   - Metadata: subject, level, filename

---

## 5. RAG Pipeline Specification

### 5.1. Hybrid Retrieval

**Vector Search (0.7 weight)**:
- Uses semantic similarity
- Model: OpenAI text-embedding-3-small
- Top-K: 10 documents

**BM25 Search (0.3 weight)**:
- Keyword-based sparse retrieval
- Top-K: 10 documents

**Combination**:
- Weighted scoring of both retrievers
- Deduplication of overlapping results
- Final top-K: 10 combined results

### 5.2. Reranking

**Supported Rerankers**:
- **Cohere** (API-based):
  - Model: rerank-v3.5
  - Fast, cloud-based
  - Requires API key
  - Rate limit: 10 calls/minute (free tier)

- **Qwen** (Local):
  - Model: Qwen/Qwen2.5-7B-Instruct
  - No API limits
  - Requires local model download
  - Optional flash attention support

**Configuration**:
- Top-K after reranking: 5 documents
- Configurable via `RERANKER_TYPE` in config

### 5.3. LLM Generation

**Supported Models**:
- **OpenAI GPT-4o** (default)
- **Google Gemini** (alternative)

**Prompt Engineering**:
```
你是一位聰明的高中歷史、地理、公民科家教。
你的角色是幫助學生理解他們課本中的概念。

[Conversation history if available]

上下文資訊如下：
[Retrieved and reranked context]

根據上述的上下文資訊和對話歷史，請回答下列問題：
問題：[User question]
答案：
```

**Output**:
- Natural language answer
- Source metadata for citations

---

## 6. Evaluation Framework

### 6.1. Dataset Generation

**Automatic Generation Pipeline**:
1. Chunk textbooks (size: 2048 characters, overlap: 128)
2. Use GPT-4o to generate Q&A pairs per chunk
3. Ensure answers are faithful to source chunks
4. Store with metadata (chunk_id, source_file, ground_truth)

**Dataset Statistics**:
- Total samples: 258
- Distribution: 6 textbook files
- Format: JSON with question, answer, context, metadata

### 6.2. Evaluation Metrics

**Binary Metrics** (true/false):
- **Faithfulness**: Answer contains only information from context
- **Relevance**: Answer addresses the question
- **Correctness**: Answer matches ground truth meaning

**Aggregate Metrics**:
- Faithfulness Rate: % of faithful answers
- Relevance Rate: % of relevant answers
- Correctness Rate: % of correct answers

### 6.3. Evaluation Pipeline

```bash
# Generate dataset
python eval/run_evaluation.py generate --data-dir ./data --output eval_dataset.json

# Run evaluation
python eval/run_evaluation.py evaluate --dataset eval_dataset.json --output results.json

# Analyze results
python analyze_now.py

# Cross-validate with Gemini
python eval/double_check_with_gemini.py
```

**Tools Available**:
- `chunker.py` - Document chunking
- `qa_generator.py` - Q&A pair generation
- `evaluator.py` - Metric calculation
- `run_evaluation.py` - Full pipeline
- `test_evaluator.py` - Quick testing (10 samples)
- `analyze_results.py` - Error analysis
- `double_check_with_gemini.py` - Cross-validation
- `filter_false_samples.py` - Dataset filtering
- `clean_retrieval_metrics.py` - JSON cleanup

---

## 7. Technical Stack

### 7.1. Backend
- **Language**: Python 3.8+
- **Web Framework**: Flask + Flask-CORS
- **RAG Orchestration**: LlamaIndex
- **Vector Database**: ChromaDB (persistent)
- **Query Logging**: SQLite

### 7.2. Frontend
- **Framework**: React.js
- **Routing**: React Router
- **HTTP Client**: Axios
- **Visualization**: Recharts
- **Styling**: CSS3

### 7.3. AI/ML Services
- **LLM Providers**:
  - OpenAI (GPT-4o, text-embedding-3-small)
  - Google Gemini (optional)
- **Reranking**:
  - Cohere Rerank API
  - Qwen (local alternative)

### 7.4. Development Tools
- **Package Manager**: pip (migrate to uv planned)
- **Environment**: .env file for configuration
- **Version Control**: Git

---

## 8. API Specification

### 8.1. Student Endpoints

#### POST `/api/chat/query`
Submit a question and get an answer with sources.

**Request**:
```json
{
  "question": "公民身分如何演變?",
  "session_id": "optional-session-id"
}
```

**Response**:
```json
{
  "answer": "公民身分的演變...",
  "session_id": "generated-or-provided-session-id",
  "sources": [
    {
      "index": 1,
      "subject": "Civic",
      "level": "L1",
      "filename": "Civic_L1.txt",
      "text_preview": "公民身分的概念起源於..."
    }
  ]
}
```

#### GET `/api/chat/health`
Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-01-10T10:30:00Z"
}
```

### 8.2. Teacher Endpoints

#### GET `/api/teacher/queries`
Get all logged student queries.

**Response**:
```json
{
  "queries": [
    {
      "id": 1,
      "question": "公民身分如何演變?",
      "timestamp": "2025-01-10T10:30:00",
      "session_id": "session-123",
      "response_length": 250
    }
  ]
}
```

#### GET `/api/teacher/statistics`
Get query statistics.

**Response**:
```json
{
  "total_queries": 150,
  "recent_queries_7days": 45,
  "by_subject": {
    "History": 60,
    "Geography": 50,
    "Civic": 40
  }
}
```

#### POST `/api/teacher/search`
Search queries by keyword.

**Request**:
```json
{
  "keyword": "公民"
}
```

**Response**:
```json
{
  "queries": [/* matching queries */]
}
```

#### GET `/api/teacher/keywords?limit=50`
Get common keywords from queries.

**Response**:
```json
{
  "keywords": [
    ["公民", 45],
    ["演變", 32],
    ["權力", 28]
  ]
}
```

