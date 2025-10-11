"""
Configuration module for HGC Helper
Loads environment variables and sets up configuration constants
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")

# Paths
DATA_DIR = "data"
CHROMA_DB_DIR = "chroma_db"
QUERY_LOG_DB = "query_logs.db"

# RAG Configuration
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K_RETRIEVAL = 10
TOP_K_RERANK = 5

VECTOR_WEIGHT = 0.7
BM25_WEIGHT = 0.3

# Embedding Model
EMBEDDING_MODEL = "text-embedding-3-small"

# Reranker Configuration
RERANKER_TYPE = os.getenv("RERANKER_TYPE", "cohere")  # Options: "cohere" or "qwen"
USE_FLASH_ATTENTION = False  # Set to True if using CUDA GPU with flash attention support (Qwen only)

# Conversation Memory Configuration
MAX_MEMORY_ROUNDS = int(os.getenv("MAX_MEMORY_ROUNDS", "3"))  # Maximum conversation rounds to keep (1 round = user + assistant)
