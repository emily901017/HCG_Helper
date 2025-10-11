好的，這是一個根據您的專題研究成果，轉換成給 Coding Agent 的規格文件 (Spec)。

---

### **Project Specification: HGC Helper - Educational Insights Platform**

#### **1. Project Objective**

To build a full-stack web application, "HGC Helper," that functions as an intelligent tutor for high school History, Geography, and Civics. The system will use a Retrieval-Augmented Generation (RAG) pipeline to answer student questions based on a textbook knowledge base. Critically, it will also log all student queries to create a "learning difficulty database," providing teachers with actionable insights for lesson planning.

#### **2. User Personas**

1.  **Student:** Asks academic questions related to social studies subjects and receives credible, context-aware answers.
2.  **Teacher:** Accesses a dashboard to analyze the questions students are asking, in order to identify common areas of confusion or curiosity.

#### **3. Core Functional Requirements**

**3.1. Student-Facing Interface**
* **[UI]** A clean, simple chat interface.
* **[Backend]** User can input a question in a text box.
* **[Backend]** The submitted question triggers the RAG pipeline.
* **[Backend]** The system must log the user's raw question, a timestamp, and potentially a session ID to a persistent database (for the teacher dashboard).
* **[UI]** The generated answer from the LLM is displayed to the student in the chat interface.
* **[NEW FEATURE]** Source references are displayed below each answer, showing:
  * Subject and level (e.g., History - L1, Geography - L2)
  * Filename of the source textbook
  * Preview of the relevant text excerpt (first 200 characters)
  * Each source is numbered for easy reference

**3.2. Teacher-Facing Dashboard**
* **[UI]** A separate, protected page or view for teachers.
* **[Backend]** Fetches all logged student questions from the database.
* **[UI]** Displays the questions in a searchable, filterable, and sortable table or list.
    * Columns should include: `Question`, `Timestamp`, `Frequency (if aggregated)`.
* **[UI Feature - Optional but Recommended]** Basic data visualization, such as a word cloud of common keywords in questions or a bar chart showing question frequency over time.

**3.3. Data Ingestion & Processing (Offline Pipeline)**
* **[Task]** A script or module to process raw text files from high school social studies textbooks.
* **[Processing Steps]**
    1.  **Load:** Read text from data folder which contains civic, geo, history data in txt format.
    2.  **Clean:** Remove irrelevant artifacts, headers, footers, etc.
    3.  **Chunk:** Split the cleaned text into semantically meaningful chunks of a specified size.
    4.  **Embed:** Convert each text chunk into a vector embedding.
    5.  **Store:** Ingest the text chunks and their corresponding embeddings into a ChromaDB vector store.

#### **4. Technical Stack**

* **Programming Language:** `Python`
* **Core Framework:** `LlamaIndex` (for orchestrating the RAG pipeline)
* **Backend API:** `Flask` with Flask-CORS for REST API endpoints
* **Frontend Interface:** `React.js` with React Router for navigation
* **Vector Database:** `ChromaDB`
* **LLM API:** The system must be able to integrate with and switch between:
    * `Google Gemini`
    * `OpenAI GPT-4`
    * (Implement using API keys stored in environment variables)
* **Reranking:** `Cohere Rerank API` for improving retrieval quality

#### **5. RAG Pipeline Architecture (Backend Logic)**

This is the core logic triggered by a student's query. It must be implemented using LlamaIndex.

1.  **Query Input:** Receive the student's question string from the React frontend.

2.  **Hybrid Retrieval:**
    * Configure a retriever in LlamaIndex that combines both dense (vector/semantic) search and sparse (keyword/BM25) search with configurable weights (default: 0.7 vector + 0.3 BM25).
    * The retriever should query the ChromaDB knowledge base and return a list of potentially relevant text nodes (documents).

3.  **Reranking:**
    * Integrate a `Reranker` model (e.g., Cohere Rerank, or a sentence-transformer based cross-encoder).
    * Pass the retrieved nodes from the retrieval step to the reranker.
    * The reranker will re-order the nodes based on their relevance to the original query, outputting the top-k most relevant nodes.

4.  **Contextual Generation:**
    * Construct a prompt for the LLM. This prompt must include:
        * The refined, high-relevance context from the reranking step.
        * The original student question.
    * Send this combined prompt to the selected LLM API (Gemini or GPT-4).
    * **Extract source references** from the reranked nodes, including metadata (subject, level, filename) and text previews.
    * Return both the LLM's generated response and the source references to the frontend.

#### **6. Implementation Plan & Task Breakdown**

1.  **Environment Setup:**
    * Initialize a Python project with `pip` or `conda`.
    * Install all required libraries: `llama-index`, `streamlit`, `chromadb`, `google-generativeai`, `openai`, etc.
    * Set up `.env` file for API keys.

2.  **Module 1: Data Ingestion**
    * Write `ingest.py`.
    * Implement functions for loading, cleaning, chunking, and embedding textbook data.
    * Ensure data is successfully stored in a local ChromaDB instance.

3.  **Module 2: RAG Backend**
    * Create `engine.py` implementing the RAG pipeline.
    * Use LlamaIndex to build the query engine encompassing the Hybrid Retriever (0.7 vector + 0.3 BM25), Reranker, and LLM Synthesizer.
    * Implement the logging mechanism for student questions.
    * Return both answers and source references from queries.

4.  **Module 3: Backend API**
    * Create `api.py` using Flask.
    * Implement REST API endpoints:
        * `/api/chat/query` - Process student questions and return answers with sources
        * `/api/teacher/queries` - Get all logged questions
        * `/api/teacher/statistics` - Get query statistics
        * `/api/teacher/search` - Search questions by keyword
        * `/api/teacher/keywords` - Get common keywords
    * Enable CORS for React frontend communication.

5.  **Module 4: Frontend UI**
    * Create React application with components:
        * `StudentChat.js` - Interactive chat interface with source display
        * `TeacherDashboard.js` - Analytics dashboard with charts and search
        * `App.js` - Main router and navigation
    * Use React Router for navigation between student and teacher views.
    * Use Recharts for data visualization.
    * Implement beautiful UI for displaying source references below answers.

6.  **Integration & Testing:**
    * Connect the React frontend to the Flask backend API.
    * Thoroughly test the end-to-end flow: from asking a question to seeing the answer with sources.
    * Verify the question is logged and appears in the teacher dashboard.
    * Test the teacher dashboard's charts, search, and analytics features.

#### **7. System Architecture & Data Flow**

**7.1. RAG Pipeline Architecture & Data Flow**

This section provides a detailed visual representation of how data flows through the HCG Helper system, from data ingestion to query processing and response generation.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 1: DATA INGESTION (Offline)                    │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────┐
    │  Raw Textbook Data                                              │
    │  ├─ data/civic/*.txt    (公民課本文本)                          │
    │  ├─ data/geo/*.txt      (地理課本文本)                          │
    │  └─ data/history/*.txt  (歷史課本文本)                          │
    └───────────────────┬─────────────────────────────────────────────┘
                        │
                        │ Load & Read Files
                        ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │  Text Preprocessing Pipeline (ingest.py)                         │
    │  ├─ Clean: Remove headers, footers, artifacts                   │
    │  ├─ Normalize: Fix encoding, spacing issues                     │
    │  └─ Extract Metadata: Subject, level, filename                  │
    └───────────────────┬──────────────────────────────────────────────┘
                        │
                        │ Cleaned Text + Metadata
                        ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │  Text Chunking Module                                            │
    │  ├─ Strategy: Semantic chunking (preserve context)              │
    │  ├─ Chunk Size: 512 tokens (configurable)                       │
    │  └─ Overlap: 50 tokens between chunks                           │
    └───────────────────┬──────────────────────────────────────────────┘
                        │
                        │ Text Chunks with Metadata
                        ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │  Dual Index Creation (LlamaIndex)                                │
    │  ├─ Vector Embeddings: OpenAI text-embedding-3-large            │
    │  └─ BM25 Index: Token-based sparse representation               │
    └───────────────────┬──────────────────────────────────────────────┘
                        │
                        │ Store Indexes
                        ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │  ChromaDB Vector Store                                           │
    │  ├─ Collections by subject (civic, geo, history)                │
    │  ├─ Vector embeddings (dense)                                   │
    │  ├─ BM25 indexes (sparse)                                       │
    │  └─ Metadata: {subject, level, filename, chunk_id}              │
    └──────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 2: QUERY PROCESSING (Real-time)                    │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────────┐
    │  Student Question (Frontend)                                     │
    │  "台灣的民主化過程是什麼?"                                       │
    └───────────────────┬──────────────────────────────────────────────┘
                        │
                        │ POST /api/chat/query
                        ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │  Flask API Endpoint (api.py)                                     │
    │  ├─ Parse request                                                │
    │  ├─ Log query to database (timestamp, session_id)               │
    │  └─ Forward to RAG engine                                        │
    └───────────────────┬──────────────────────────────────────────────┘
                        │
                        │ Query string
                        ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │  Hybrid Retriever (engine.py - LlamaIndex)                       │
    │                                                                  │
    │  ┌─────────────────────┐    ┌──────────────────────┐            │
    │  │ Vector Search       │    │ BM25 Search          │            │
    │  │ (Semantic)          │    │ (Keyword)            │            │
    │  │ - Embed query       │    │ - Tokenize query     │            │
    │  │ - Cosine similarity │    │ - TF-IDF scoring     │            │
    │  │ - Top 20 results    │    │ - Top 20 results     │            │
    │  └──────────┬──────────┘    └──────────┬───────────┘            │
    │             │                           │                        │
    │             └───────────┬───────────────┘                        │
    │                         │                                        │
    │              Weighted Fusion (α=0.7, β=0.3)                     │
    │              Score = 0.7*vector + 0.3*BM25                      │
    │                         │                                        │
    │                         ▼                                        │
    │              ┌──────────────────────┐                            │
    │              │ Top 15 Candidates    │                            │
    │              └──────────────────────┘                            │
    └───────────────────┬──────────────────────────────────────────────┘
                        │
                        │ Retrieved nodes with relevance scores
                        ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │  Cohere Reranker                                                 │
    │  ├─ Input: Query + 15 candidate chunks                          │
    │  ├─ Model: rerank-multilingual-v3.0                             │
    │  ├─ Cross-encoder scoring (query-document pairs)                │
    │  └─ Output: Top 5 most relevant chunks                          │
    └───────────────────┬──────────────────────────────────────────────┘
                        │
                        │ Reranked top-K nodes
                        ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │  Context Assembly & Prompt Construction                          │
    │  ├─ Concatenate top 5 chunks with metadata                      │
    │  ├─ Build system prompt with instructions                       │
    │  └─ Construct user prompt: context + question                   │
    └───────────────────┬──────────────────────────────────────────────┘
                        │
                        │ Final prompt
                        ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │  LLM Generation (Gemini/GPT-4)                                   │
    │  ├─ Model: gemini-1.5-pro or gpt-4-turbo                        │
    │  ├─ Temperature: 0.3 (factual responses)                        │
    │  ├─ Max tokens: 1000                                            │
    │  └─ System role: Educational tutor for social studies           │
    └───────────────────┬──────────────────────────────────────────────┘
                        │
                        │ Generated answer
                        ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │  Response Packaging                                              │
    │  {                                                               │
    │    "answer": "台灣的民主化過程...",                              │
    │    "sources": [                                                  │
    │      {                                                           │
    │        "subject": "History",                                     │
    │        "level": "L2",                                            │
    │        "filename": "taiwan_history_ch3.txt",                     │
    │        "preview": "民國76年解除戒嚴，開放黨禁報禁...",            │
    │        "relevance_score": 0.92                                   │
    │      }, ...                                                      │
    │    ]                                                             │
    │  }                                                               │
    └───────────────────┬──────────────────────────────────────────────┘
                        │
                        │ JSON response
                        ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │  Frontend Display (React)                                        │
    │  ├─ Render answer in chat bubble                                │
    │  └─ Display source references below answer                      │
    └──────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 3: ANALYTICS PIPELINE (Teacher)                    │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────────┐
    │  Query Logging Database (SQLite/PostgreSQL)                      │
    │  Table: student_queries                                          │
    │  ├─ id (PRIMARY KEY)                                             │
    │  ├─ question (TEXT)                                              │
    │  ├─ timestamp (DATETIME)                                         │
    │  ├─ session_id (VARCHAR)                                         │
    │  └─ subject_detected (VARCHAR, optional)                         │
    └───────────────────┬──────────────────────────────────────────────┘
                        │
                        │ Teacher accesses dashboard
                        ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │  Teacher Dashboard Backend                                       │
    │  ├─ /api/teacher/queries - Fetch all queries                    │
    │  ├─ /api/teacher/statistics - Aggregate stats                   │
    │  │   └─ Queries per day/week/month                              │
    │  ├─ /api/teacher/keywords - Extract common keywords             │
    │  │   └─ NLP processing (jieba for Chinese)                      │
    │  └─ /api/teacher/search - Search/filter queries                 │
    └───────────────────┬──────────────────────────────────────────────┘
                        │
                        │ Analytics data
                        ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │  Teacher Dashboard UI (React)                                    │
    │  ├─ Query table with sorting & filtering                        │
    │  ├─ Time series chart (queries over time)                       │
    │  ├─ Word cloud (common topics)                                  │
    │  ├─ Subject distribution pie chart                              │
    │  └─ Export to CSV functionality                                 │
    └──────────────────────────────────────────────────────────────────┘
```

**7.2. Key Design Decisions**

1. **Hybrid Retrieval Strategy**
   - **Vector Search (70%):** Captures semantic meaning, handles paraphrasing
   - **BM25 (30%):** Ensures exact keyword matches aren't missed
   - **Rationale:** Combines the best of both approaches for educational content

2. **Reranking Layer**
   - **Purpose:** Cross-encoder models are more accurate than bi-encoders for final ranking
   - **Benefit:** Reduces false positives, improves answer quality
   - **Trade-off:** Adds ~200ms latency but significantly improves relevance

3. **Metadata Preservation**
   - Store subject, level, and filename with each chunk
   - Enables source attribution and credibility
   - Helps students verify information and teachers understand content coverage

4. **Query Logging for Pedagogy**
   - Every question logged in real-time
   - No PII (personally identifiable information) stored
   - Enables data-driven teaching insights

5. **Modular LLM Integration**
   - Abstract LLM layer supports multiple providers
   - Easy to switch between Gemini/GPT-4 based on cost/performance
   - Future-proof for new models

**7.3. Performance Targets**

| Metric | Target | Notes |
|--------|--------|-------|
| Query latency (p95) | < 3 seconds | From question submit to answer display |
| Retrieval precision@5 | > 0.85 | At least 4/5 retrieved chunks relevant |
| Answer accuracy | > 90% | Manual evaluation on test set |
| Concurrent users | 50+ | Without performance degradation |
| Database size | ~500MB | For 3 subjects × 2 years of textbooks |

**7.4. Security & Privacy Considerations**

1. **Data Protection**
   - No student names or personal data collected
   - Session IDs are anonymized UUIDs
   - Teacher dashboard requires authentication (future enhancement)

2. **API Security**
   - API keys stored in environment variables, never in code
   - Rate limiting on endpoints to prevent abuse
   - Input validation and sanitization on all user inputs

3. **CORS Configuration**
   - Restrict allowed origins in production
   - Enable credentials only for authenticated routes