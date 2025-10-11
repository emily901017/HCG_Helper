"""
Qwen3 Reranker Module
Implements local reranking using Qwen3-Reranker-0.6B model
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple
import config


class QwenReranker:
    """
    Local reranker using Qwen3-Reranker-0.6B model
    Replaces Cohere API-based reranking with local inference
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-Reranker-0.6B",
                 device: str = None,
                 use_flash_attention: bool = False):
        """
        Initialize the Qwen reranker

        Args:
            model_name: HuggingFace model name
            device: Device to use ('cuda', 'mps', 'cpu'). Auto-detected if None
            use_flash_attention: Whether to use flash attention 2 (requires GPU)
        """
        self.model_name = model_name
        self.max_length = 8192

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"Loading Qwen3 Reranker on {self.device}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side='left'
        )

        # Load model
        if use_flash_attention and self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                attn_implementation="flash_attention_2"
            ).to(self.device).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name
            ).to(self.device).eval()

        # Get token IDs for yes/no
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")

        # Define prefix and suffix for the prompt format
        self.prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
            "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
            "<|im_start|>user\n"
        )
        self.suffix = (
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
        )

        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

        print(f"✓ Qwen3 Reranker loaded successfully")

    def format_instruction(self, instruction: str, query: str, doc: str) -> str:
        """
        Format the reranking instruction

        Args:
            instruction: Task instruction
            query: User query
            doc: Document to judge

        Returns:
            Formatted prompt string
        """
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'

        return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
            instruction=instruction,
            query=query,
            doc=doc
        )

    def process_inputs(self, pairs: List[str]) -> dict:
        """
        Process input pairs into model inputs

        Args:
            pairs: List of formatted instruction strings

        Returns:
            Tokenized inputs ready for model
        """
        # Tokenize without padding first
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation='longest_first',
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )

        # Add prefix and suffix tokens
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens

        # Pad to max_length and convert to tensors
        inputs = self.tokenizer.pad(
            inputs,
            padding='max_length',
            return_tensors="pt",
            max_length=self.max_length
        )

        # Move to device
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)

        return inputs

    @torch.no_grad()
    def compute_scores(self, inputs: dict) -> List[float]:
        """
        Compute relevance scores for input pairs

        Args:
            inputs: Tokenized model inputs

        Returns:
            List of relevance scores (0-1)
        """
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]

        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()

        return scores

    def rerank(self, query: str, documents: List[str],
               top_n: int = None,
               instruction: str = None) -> List[Tuple[int, float]]:
        """
        Rerank documents based on relevance to query

        Args:
            query: User query
            documents: List of document texts
            top_n: Number of top results to return (None for all)
            instruction: Optional custom instruction

        Returns:
            List of (index, score) tuples sorted by relevance
        """
        if not documents:
            return []

        # Default instruction for educational content
        if instruction is None:
            instruction = (
                'Given a student question about History, Geography, or Civics, '
                'retrieve relevant passages from textbooks that answer the question'
            )

        # Format all query-document pairs
        pairs = [
            self.format_instruction(instruction, query, doc)
            for doc in documents
        ]

        # Process inputs
        inputs = self.process_inputs(pairs)

        # Compute scores
        scores = self.compute_scores(inputs)

        # Create (index, score) pairs and sort by score
        results = [(idx, score) for idx, score in enumerate(scores)]
        results.sort(key=lambda x: x[1], reverse=True)

        # Return top-n if specified
        if top_n is not None:
            results = results[:top_n]

        return results


def test_reranker():
    """Test the Qwen reranker"""
    print("=" * 60)
    print("Testing Qwen3 Reranker")
    print("=" * 60)

    # Initialize reranker
    reranker = QwenReranker()

    # Test data
    query = "公民身分如何演變?"
    documents = [
        "公民身分的演變反映了人類社會對權利和義務的理解不斷深化。在古代社會，公民身分通常僅限於特定族群或階級。",
        "地理環境影響人類的生活方式和文化發展。不同的氣候帶造就了不同的農業型態。",
        "歷史上的重大事件往往改變了社會結構。工業革命帶來了生產方式的根本變化。",
        "現代公民身分逐漸強調平等原則，認為所有人都應享有基本權利。公民權利的範圍持續擴大。"
    ]

    print(f"\nQuery: {query}")
    print(f"\nReranking {len(documents)} documents...")

    # Rerank documents
    results = reranker.rerank(query, documents, top_n=3)

    print("\nTop 3 Results:")
    for rank, (idx, score) in enumerate(results, 1):
        print(f"\n[{rank}] Document {idx} (Score: {score:.4f})")
        print(f"Text: {documents[idx][:100]}...")

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_reranker()
