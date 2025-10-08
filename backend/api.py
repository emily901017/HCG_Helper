"""
Flask API Backend for HGC Helper
Provides REST endpoints for student chat and teacher dashboard
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import RAGEngine
from database import QueryLogger
import uuid

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Initialize components
rag_engine = None
query_logger = QueryLogger()


@app.before_request
def initialize_engine():
    """Initialize RAG engine on first request"""
    global rag_engine
    if rag_engine is None:
        try:
            print("Initializing RAG engine...")
            rag_engine = RAGEngine()
            print("RAG engine initialized successfully")
        except Exception as e:
            print(f"Error initializing RAG engine: {e}")
            import traceback
            traceback.print_exc()


# Student Chat Endpoints
@app.route('/api/chat/query', methods=['POST'])
def chat_query():
    """
    Handle student chat query

    Request body:
    {
        "question": str,
        "session_id": str (optional)
    }

    Response:
    {
        "answer": str,
        "session_id": str
    }
    """
    try:
        data = request.get_json()
        question = data.get('question')
        session_id = data.get('session_id') or str(uuid.uuid4())

        if not question:
            return jsonify({"error": "Question is required"}), 400

        # Check if RAG engine is initialized
        if rag_engine is None:
            return jsonify({"error": "RAG engine not initialized. Please check backend logs."}), 503

        # Generate answer
        print(f"Processing query: {question}")
        answer = rag_engine.query(question=question, session_id=session_id)
        print(f"Answer generated successfully")

        return jsonify({
            "answer": answer,
            "session_id": session_id
        }), 200

    except Exception as e:
        print(f"Error in chat_query: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/chat/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "engine_initialized": rag_engine is not None
    }), 200


# Teacher Dashboard Endpoints
@app.route('/api/teacher/queries', methods=['GET'])
def get_all_queries():
    """
    Get all student queries

    Response:
    {
        "queries": [
            {
                "id": int,
                "question": str,
                "timestamp": str,
                "session_id": str,
                "response_length": int
            }
        ]
    }
    """
    try:
        queries = query_logger.get_all_queries()
        return jsonify({"queries": queries}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/teacher/statistics', methods=['GET'])
def get_statistics():
    """
    Get query statistics

    Response:
    {
        "total_queries": int,
        "recent_queries_7days": int,
        "by_subject": {
            "History": int,
            "Geography": int,
            "Civics": int
        }
    }
    """
    try:
        stats = query_logger.get_query_statistics()
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/teacher/search', methods=['POST'])
def search_queries():
    """
    Search queries by keyword

    Request body:
    {
        "keyword": str
    }

    Response:
    {
        "results": [
            {
                "id": int,
                "question": str,
                "timestamp": str,
                "session_id": str
            }
        ]
    }
    """
    try:
        data = request.get_json()
        keyword = data.get('keyword')

        if not keyword:
            return jsonify({"error": "Keyword is required"}), 400

        results = query_logger.search_queries(keyword)
        return jsonify({"results": results}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/teacher/keywords', methods=['GET'])
def get_common_keywords():
    """
    Get most common keywords from queries

    Query parameters:
    - limit: int (default: 50)

    Response:
    {
        "keywords": [
            ["keyword", count],
            ...
        ]
    }
    """
    try:
        limit = request.args.get('limit', 50, type=int)
        keywords = query_logger.get_common_keywords(limit=limit)
        return jsonify({"keywords": keywords}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Run in development mode
    app.run(debug=True, host='0.0.0.0', port=3000)
