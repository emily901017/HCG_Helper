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


7.1. Objective
To establish a systematic framework for evaluating the RAG pipeline's performance. The evaluation will focus on the accuracy of information retrieval and the quality of the generated answers, ensuring they are relevant, faithful to the source, and free of hallucinations.

7.2. Evaluation Dataset Generation (Chunk-based Method)
A high-quality Question & Answer dataset will be automatically generated for evaluation, overcoming the challenges of content-drift and hallucination common in large-scale generation.

Step 1: Document Chunking:

The original textbook documents will be chunked by size 512. Each chunk will serve as a ground-truth context.

Step 2: Iterative Q&A Generation:

Each chunk will be independently fed to an LLM.

A precise prompt will instruct the LLM to generate a small number (e.g., 2  Q&A pairs based solely on the provided chunk.

Example Prompt: "Please act as a high school student. Based ONLY on the text provided below, generate 3-5 questions a student might ask, along with answers derived directly from the text. The answer must not contain any information outside of this text."

Step 3: Verification and Refinement:

Since each Q&A pair is tied to a specific source chunk, manual or semi-automated verification becomes highly efficient.

Checks:

Is the question relevant to the chunk?

Is the answer factually correct according to the chunk?

Does the answer avoid introducing external information (no hallucination)?

Invalid or low-quality pairs will be discarded to ensure the final evaluation dataset is reliable.

7.3. Key Evaluation Metrics
The generated dataset will be used to run batch tests and measure the following:

Retrieval Evaluation:

Hit Rate: For a given question, did the retrieval step (Hybrid Retrieval + Reranker) successfully fetch the ground-truth chunk from which the question was generated?

Mean Reciprocal Rank (MRR): Measures the average rank of the correct ground-truth chunk in the list of retrieved documents. A higher MRR indicates the retriever is more effective at prioritizing the correct context.

Generation Evaluation (End-to-End):

Faithfulness: How factually accurate is the generated answer compared to the retrieved context? This metric penalizes hallucinations.

Answer Relevance: How well does the generated answer address the user's question? The answer could be faithful but irrelevant if it fails to grasp the user's intent.

(Evaluation Method: These metrics can be assessed using an LLM-as-a-judge approach or manual human scoring.)