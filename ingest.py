"""
Data Ingestion Module for HGC Helper
Processes textbook data and stores it in ChromaDB vector store
"""
import os
from pathlib import Path
from typing import List
import chromadb
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini
import config


def setup_llm():
    """Set up the LLM based on configuration"""
    if config.LLM_PROVIDER == "gemini":
        return Gemini(api_key=config.GOOGLE_API_KEY, model=config.LLM_MODEL)
    else:
        return OpenAI(api_key=config.OPENAI_API_KEY, model=config.LLM_MODEL)


def load_textbook_files(data_dir: str = config.DATA_DIR) -> List[Document]:
    """
    Load all textbook text files from the data directory

    Args:
        data_dir: Directory containing textbook text files

    Returns:
        List of Document objects
    """
    documents = []
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Load all .txt files
    for file_path in data_path.glob("*.txt"):
        print(f"Loading {file_path.name}...")

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract subject and level from filename (e.g., "Civic_L1.txt")
        filename = file_path.stem
        parts = filename.split('_')
        subject = parts[0] if len(parts) > 0 else "Unknown"
        level = parts[1] if len(parts) > 1 else "Unknown"

        # Create document with metadata
        doc = Document(
            text=content,
            metadata={
                "filename": file_path.name,
                "subject": subject,
                "level": level,
                "source": str(file_path)
            }
        )
        documents.append(doc)

    print(f"Loaded {len(documents)} documents")
    return documents


def clean_text(text: str) -> str:
    """
    Clean the text by removing irrelevant artifacts

    Args:
        text: Raw text string

    Returns:
        Cleaned text string
    """
    # Remove excessive whitespace
    text = ' '.join(text.split())

    # Remove page numbers (if they follow a pattern)
    # You can add more cleaning rules based on your data

    return text


def create_vector_store(documents: List[Document], persist_dir: str = config.CHROMA_DB_DIR):
    """
    Process documents and create/update ChromaDB vector store

    Args:
        documents: List of documents to process
        persist_dir: Directory to persist ChromaDB
    """
    # Clean document texts - create new Document objects since text is immutable
    cleaned_documents = []
    for doc in documents:
        cleaned_doc = Document(
            text=clean_text(doc.text),
            metadata=doc.metadata
        )
        cleaned_documents.append(cleaned_doc)

    # Use cleaned documents for the rest of the process
    documents = cleaned_documents

    # Set up embedding model
    embed_model = OpenAIEmbedding(
        api_key=config.OPENAI_API_KEY,
        model=config.EMBEDDING_MODEL
    )

    # Set up LLM
    llm = setup_llm()

    # Configure LlamaIndex settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = config.CHUNK_SIZE
    Settings.chunk_overlap = config.CHUNK_OVERLAP

    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path=persist_dir)

    # Create or get collection
    collection = chroma_client.get_or_create_collection("textbook_knowledge")

    # Create vector store
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create text splitter
    text_splitter = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )

    print("Creating embeddings and storing in ChromaDB...")

    # Create index and store embeddings
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[text_splitter],
        show_progress=True
    )

    print(f"Successfully created vector store at {persist_dir}")
    return index


def main():
    """Main function to run the data ingestion pipeline"""
    print("=" * 50)
    print("HGC Helper - Data Ingestion Pipeline")
    print("=" * 50)

    # Load documents
    print("\n1. Loading textbook files...")
    documents = load_textbook_files()

    # Create vector store
    print("\n2. Processing and creating vector store...")
    index = create_vector_store(documents)

    print("\n" + "=" * 50)
    print("Data ingestion completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
