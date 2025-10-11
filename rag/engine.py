"""
RAG Pipeline Engine for HGC Helper
Implements hybrid retrieval, reranking, and LLM generation
"""
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings, get_response_synthesizer
from llama_index.core.prompts import PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.schema import NodeWithScore
import config
from database import QueryLogger
from typing import List

import cohere


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
        # Conversation memory: session_id -> list of {role, content} messages
        # Maximum rounds from config (1 round = 1 user + 1 assistant message)
        self.conversation_memory = {}
        self.max_memory_rounds = getattr(config, 'MAX_MEMORY_ROUNDS', 3)
        self._setup_llm()
        self._setup_embeddings()
        self._setup_index()
        self._setup_reranker()

    def _setup_llm(self):
        """Set up the LLM based on configuration"""
        if config.LLM_PROVIDER == "gemini":
            self.llm = Gemini(
                api_key=config.GOOGLE_API_KEY,
                model=config.LLM_MODEL
            )
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

        # Get all nodes from the vector store for BM25
        # We need to retrieve all documents to build BM25 index
        all_node_dicts = collection.get(include=['metadatas', 'documents'])
        from llama_index.core.schema import TextNode

        self.all_nodes = []
        for i, doc_id in enumerate(all_node_dicts['ids']):
            node = TextNode(
                text=all_node_dicts['documents'][i],
                id_=doc_id,
                metadata=all_node_dicts['metadatas'][i] if all_node_dicts['metadatas'] else {}
            )
            self.all_nodes.append(node)

    def _setup_reranker(self):
        """Set up reranker based on configuration"""
        reranker_type = getattr(config, 'RERANKER_TYPE', 'qwen').lower()

        if reranker_type == 'cohere':
            # Use Cohere API reranker
            if config.COHERE_API_KEY:
                try:
                    self.cohere_client = cohere.Client(config.COHERE_API_KEY)
                    self.reranker_type = 'cohere'
                    self.reranker = None
                    print("✓ Cohere Reranker initialized successfully")
                except Exception as e:
                    self.cohere_client = None
                    self.reranker = None
                    self.reranker_type = None
                    print(f"Warning: Could not initialize Cohere reranker: {e}")
                    print("Reranking will be skipped.")
            else:
                self.cohere_client = None
                self.reranker = None
                self.reranker_type = None
                print("Warning: COHERE_API_KEY not set. Reranking will be skipped.")
        else:
            print(f"Warning: Unknown reranker type '{reranker_type}'. Valid options: 'cohere'")
            self.reranker = None
            self.cohere_client = None
            self.reranker_type = None
            print("Reranking will be skipped.")

    def _hybrid_retrieve(self, query: str, vector_retriever, bm25_retriever) -> List[NodeWithScore]:
        """
        Perform weighted hybrid retrieval combining vector and BM25 search

        Args:
            query: The user's query
            vector_retriever: Vector-based retriever
            bm25_retriever: BM25-based retriever

        Returns:
            Combined and reranked list of nodes with weighted scores
        """
        # Retrieve from both retrievers
        vector_nodes = vector_retriever.retrieve(query)
        bm25_nodes = bm25_retriever.retrieve(query)

        # Create a dictionary to store combined scores
        node_scores = {}

        # Add vector scores with weight
        for node in vector_nodes:
            node_id = node.node.node_id
            node_scores[node_id] = {
                'node': node.node,
                'score': node.score * config.VECTOR_WEIGHT
            }

        # Add BM25 scores with weight
        for node in bm25_nodes:
            node_id = node.node.node_id
            if node_id in node_scores:
                # Combine scores if node exists in both
                node_scores[node_id]['score'] += node.score * config.BM25_WEIGHT
            else:
                # Add new node from BM25
                node_scores[node_id] = {
                    'node': node.node,
                    'score': node.score * config.BM25_WEIGHT
                }

        # Convert back to NodeWithScore and sort by combined score
        combined_nodes = [
            NodeWithScore(node=data['node'], score=data['score'])
            for data in node_scores.values()
        ]
        combined_nodes.sort(key=lambda x: x.score, reverse=True)

        # Return top-k nodes
        return combined_nodes[:config.TOP_K_RETRIEVAL]

    def _get_conversation_history(self, session_id: str) -> List[dict]:
        """
        Get conversation history for a session

        Args:
            session_id: Session identifier

        Returns:
            List of message dictionaries with 'role' and 'content'
        """
        if session_id not in self.conversation_memory:
            self.conversation_memory[session_id] = []
        return self.conversation_memory[session_id]

    def _add_to_conversation_memory(self, session_id: str, role: str, content: str):
        """
        Add a message to conversation memory

        Args:
            session_id: Session identifier
            role: 'user' or 'assistant'
            content: Message content
        """
        if session_id not in self.conversation_memory:
            self.conversation_memory[session_id] = []

        # Add new message
        self.conversation_memory[session_id].append({
            "role": role,
            "content": content
        })

        # Enforce maximum rounds limit (pop oldest pair if exceeded)
        # Each round = 1 user message + 1 assistant message = 2 messages
        max_messages = self.max_memory_rounds * 2
        while len(self.conversation_memory[session_id]) > max_messages:
            # Remove the oldest pair (user + assistant)
            self.conversation_memory[session_id].pop(0)  # Remove oldest user message
            if len(self.conversation_memory[session_id]) > 0:
                self.conversation_memory[session_id].pop(0)  # Remove oldest assistant message

    def _format_conversation_history(self, history: List[dict]) -> str:
        """
        Format conversation history for inclusion in prompt

        Args:
            history: List of message dictionaries

        Returns:
            Formatted history string
        """
        if not history:
            return ""

        formatted = "\n對話歷史：\n"
        for msg in history:
            if msg['role'] == 'user':
                formatted += f"學生：{msg['content']}\n"
            else:
                formatted += f"家教：{msg['content']}\n"
        formatted += "\n"
        return formatted

    def _rerank_nodes(self, query: str, nodes: list) -> list:
        """
        Rerank retrieved nodes using configured reranker (Cohere or Qwen)

        Args:
            query: The user's query
            nodes: List of retrieved nodes

        Returns:
            Reranked list of nodes with updated scores
        """
        if len(nodes) == 0:
            return nodes

        # Extract text from nodes
        documents = [node.node.get_content() for node in nodes]

        # Use Cohere reranker
        if self.reranker_type == 'cohere' and self.cohere_client:
            try:
                results = self.cohere_client.rerank(
                    model="rerank-v3.5",
                    query=query,
                    documents=documents,
                    top_n=config.TOP_K_RERANK
                )
                # Reorder nodes based on Cohere results
                reranked_nodes = []
                for result in results.results:
                    node_with_score = NodeWithScore(
                        node=nodes[result.index].node,
                        score=result.relevance_score
                    )
                    if result.relevance_score <= 0.4:
                        continue
                    reranked_nodes.append(node_with_score)
                    print(f"Doc idx: {result.index}, Score: {result.relevance_score}")
                return reranked_nodes
            except Exception as e:
                print(f"Warning: Cohere reranking failed: {e}")
                return nodes[:config.TOP_K_RERANK]

        # Use Qwen reranker
        elif self.reranker_type == 'qwen' and self.reranker:
            try:
                results = self.reranker.rerank(
                    query=query,
                    documents=documents,
                    top_n=config.TOP_K_RERANK
                )
                # Reorder nodes based on Qwen results
                reranked_nodes = []
                for idx, score in results:
                    node_with_score = NodeWithScore(
                        node=nodes[idx].node,
                        score=score
                    )
                    print(f"Doc idx: {idx}, Score: {score:.4f}")
                    reranked_nodes.append(node_with_score)
                return reranked_nodes
            except Exception as e:
                print(f"Warning: Qwen reranking failed: {e}")
                import traceback
                traceback.print_exc()
                return nodes[:config.TOP_K_RERANK]

        # No reranker available
        else:
            return nodes[:config.TOP_K_RERANK]

    def query(self, question: str, session_id: str = None) -> dict:
        """
        Process a student query through the RAG pipeline with conversation memory

        Args:
            question: The student's question
            session_id: Optional session identifier

        Returns:
            Dictionary containing:
                - answer: Generated answer from the LLM
                - sources: List of source references used
        """
        # Get conversation history for this session
        if session_id:
            conversation_history = self._get_conversation_history(session_id)
            history_text = self._format_conversation_history(conversation_history)
        else:
            history_text = ""

        # Create vector retriever
        vector_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=config.TOP_K_RETRIEVAL
        )

        # Create BM25 retriever using all nodes
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=self.all_nodes,
            similarity_top_k=config.TOP_K_RETRIEVAL
        )

        # Perform weighted hybrid retrieval (0.7 vector, 0.3 BM25)
        retrieved_nodes = self._hybrid_retrieve(question, vector_retriever, bm25_retriever)
        print(f"Retrieved {len(retrieved_nodes)} nodes (hybrid: {config.VECTOR_WEIGHT} vector + {config.BM25_WEIGHT} BM25).")

        # Rerank nodes
        if self.reranker_type:
            reranked_nodes = self._rerank_nodes(question, retrieved_nodes)
        else:
            # If no reranker, use top-k from retrieval
            reranked_nodes = retrieved_nodes[:config.TOP_K_RERANK]

        print(f"Using {len(reranked_nodes)} reranked nodes for generation.")

        # Build custom prompt template with conversation history
        qa_prompt_str = """
你是一位聰明的高中歷史、地理、公民科家教。
你的角色是幫助學生理解他們課本中的概念。回答要像真人一樣順暢自然。

指南：
1. 根據提供的上下文，提供清晰、準確的答案。
2. 如果上下文沒有包含足夠的資訊，請誠實地告知。
3. 使用例子來闡明複雜的概念。
4. 保持鼓勵與支持的態度。
5. 答案應簡潔而全面。
6. 在適當的時候，引用課本中的具體概念。
7. 如果有對話歷史，請考慮先前的對話內容來回答問題。
{history}
上下文資訊如下：
---------------------
{context_str}
---------------------

根據上述的上下文資訊和對話歷史，請回答下列問題：
問題：{query_str}
答案：
"""

        # Replace history placeholder
        qa_prompt_str = qa_prompt_str.replace("{history}", history_text)
        qa_prompt_template = PromptTemplate(qa_prompt_str)

        # Create response synthesizer with custom prompt
        response_synthesizer = get_response_synthesizer(
            llm=self.llm,
            text_qa_template=qa_prompt_template
        )

        # Generate response using reranked nodes
        response = response_synthesizer.synthesize(question, nodes=reranked_nodes)
        answer = str(response)

        # Add to conversation memory if session_id provided
        if session_id:
            self._add_to_conversation_memory(session_id, "user", question)
            self._add_to_conversation_memory(session_id, "assistant", answer)

        # Extract source references from reranked nodes
        sources = []
        for i, node in enumerate(reranked_nodes, 1):
            metadata = node.node.metadata
            source_info = {
                "index": i,
                "subject": metadata.get("subject", "Unknown"),
                "level": metadata.get("level", "Unknown"),
                "filename": metadata.get("filename", "Unknown"),
                "text_preview": node.node.get_content()
            }
            sources.append(source_info)

        # Log the query
        self.query_logger.log_query(
            question=question,
            session_id=session_id,
            response_length=len(answer)
        )

        return {
            "answer": answer,
            "sources": sources
        }

    def clear_conversation_memory(self, session_id: str = None):
        """
        Clear conversation memory for a specific session or all sessions

        Args:
            session_id: Session identifier. If None, clears all sessions
        """
        if session_id:
            if session_id in self.conversation_memory:
                self.conversation_memory[session_id] = []
                print(f"Cleared conversation memory for session {session_id}")
        else:
            self.conversation_memory = {}
            print("Cleared all conversation memory")

    def get_conversation_history(self, session_id: str) -> List[dict]:
        """
        Get conversation history for a session (public method)

        Args:
            session_id: Session identifier

        Returns:
            List of message dictionaries
        """
        return self._get_conversation_history(session_id)

    def get_query_statistics(self):
        """Get query statistics from the logger"""
        return self.query_logger.get_query_statistics()


def test_engine():
    """Test the RAG engine with a sample query"""
    print("Initializing RAG Engine...")
    engine = RAGEngine()

    print("\nTesting with sample query...")
    test_query = "公民身分如何演變?"
    result = engine.query(test_query)

    print(f"\nQuery: {test_query}")
    print(f"\nAnswer: {result['answer']}")
    print(f"\nSources ({len(result['sources'])} references):")
    for source in result['sources']:
        print(f"\n[{source['index']}] {source['subject']} - {source['level']} ({source['filename']})")
        print(f"Preview: {source['text_preview'][:100]}...")

    print("\nQuery statistics:")
    stats = engine.get_query_statistics()
    print(stats)


if __name__ == "__main__":
    test_engine()