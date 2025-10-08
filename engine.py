"""
RAG Pipeline Engine for HGC Helper
Implements hybrid retrieval, reranking, and LLM generation
"""
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
# from llama_index.llms.gemini import Gemini
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
import cohere
import config
from database import QueryLogger


class RAGEngine:
    """RAG Pipeline Engine with hybrid retrieval and reranking"""

    def __init__(self, persist_dir: str = config.CHROMA_DB_DIR):
        """
        Initialize the RAG engine

        Args:
            persist_dir: Directory where ChromaDB is persisted
        """
        self.persist_dir = persist_dir
        self.query_logger = QueryLogger()
        self._setup_llm()
        self._setup_embeddings()
        self._setup_index()
        self._setup_reranker()

    def _setup_llm(self):
        """Set up the LLM based on configuration"""
        if config.LLM_PROVIDER == "gemini":
            # self.llm = Gemini(
            #     api_key=config.GOOGLE_API_KEY,
            #     model=config.LLM_MODEL
            # )
            self.llm= None
        else:
            self.llm = OpenAI(
                api_key=config.OPENAI_API_KEY,
                model=config.LLM_MODEL
            )

        Settings.llm = self.llm

    def _setup_embeddings(self):
        """Set up the embedding model"""
        try:
            self.embed_model = OpenAIEmbedding(
                api_key=config.OPENAI_API_KEY,
                model=config.EMBEDDING_MODEL
            )
            Settings.embed_model = self.embed_model
        except Exception as e:
            print(f"Error setting up embeddings: {e}")
            print("Try upgrading: pip install --upgrade llama-index-embeddings-openai")
            raise

    def _setup_index(self):
        """Load the existing vector index from ChromaDB"""
        # Initialize ChromaDB client
        chroma_client = chromadb.PersistentClient(path=self.persist_dir)

        # Get collection
        collection = chroma_client.get_or_create_collection("textbook_knowledge")

        # Create vector store
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create index
        self.index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context
        )

    def _setup_reranker(self):
        """Set up Cohere reranker"""
        if config.COHERE_API_KEY:
            self.cohere_client = cohere.Client(config.COHERE_API_KEY)
        else:
            self.cohere_client = None
            print("Warning: COHERE_API_KEY not set. Reranking will be skipped.")

    def _rerank_nodes(self, query: str, nodes: list) -> list:
        """
        Rerank retrieved nodes using Cohere

        Args:
            query: The user's query
            nodes: List of retrieved nodes

        Returns:
            Reranked list of nodes
        """
        if not self.cohere_client or len(nodes) == 0:
            return nodes

        # Extract text from nodes
        documents = [node.node.get_content() for node in nodes]

        # Rerank using Cohere
        results = self.cohere_client.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=documents,
            top_n=config.TOP_K_RERANK
        )

        # Reorder nodes based on reranking results
        reranked_nodes = [nodes[result.index] for result in results.results]

        return reranked_nodes

    def query(self, question: str, session_id: str = None) -> str:
        """
        Process a student query through the RAG pipeline

        Args:
            question: The student's question
            session_id: Optional session identifier

        Returns:
            Generated answer from the LLM
        """
        # Create retriever with hybrid search
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=config.TOP_K_RETRIEVAL
        )

        # Retrieve nodes
        retrieved_nodes = retriever.retrieve(question)

        # Rerank nodes
        if self.cohere_client:
            reranked_nodes = self._rerank_nodes(question, retrieved_nodes)
        else:
            # If no reranker, use top-k from retrieval
            reranked_nodes = retrieved_nodes[:config.TOP_K_RERANK]

        # Create query engine with reranked nodes
        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            llm=self.llm,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=0.5)
            ]
        )

        # Build custom prompt
        system_prompt = """You are an intelligent tutor for high school History, Geography, and Civics.
Your role is to help students understand concepts from their textbooks.

Guidelines:
1. Provide clear, accurate answers based on the context provided
2. If the context doesn't contain enough information, say so honestly
3. Use examples to clarify complex concepts
4. Be encouraging and supportive
5. Keep answers concise but comprehensive
6. Cite specific concepts from the textbook when relevant

Answer the student's question based on the provided context."""

        # Generate response
        response = query_engine.query(question)
        answer = str(response)

        # Log the query
        self.query_logger.log_query(
            question=question,
            session_id=session_id,
            response_length=len(answer)
        )

        return answer

    def get_query_statistics(self):
        """Get query statistics from the logger"""
        return self.query_logger.get_query_statistics()


def test_engine():
    """Test the RAG engine with a sample query"""
    print("Initializing RAG Engine...")
    engine = RAGEngine()

    print("\nTesting with sample query...")
    test_query = "What is democracy?"
    response = engine.query(test_query)

    print(f"\nQuery: {test_query}")
    print(f"Response: {response}")

    print("\nQuery statistics:")
    stats = engine.get_query_statistics()
    print(stats)


if __name__ == "__main__":
    test_engine()
