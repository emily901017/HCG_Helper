"""
Evaluation Metrics Module

Evaluates RAG pipeline performance using various metrics:
- Retrieval accuracy (context relevance)
- Answer quality (faithfulness, relevance)
- Hallucination detection
"""

from typing import List, Dict, Optional
import json
from openai import OpenAI
import os
import time


class RAGEvaluator:
    """Evaluates RAG pipeline performance using automated metrics."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """
        Initialize the evaluator.

        Args:
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
            model: OpenAI model to use for evaluation
        """
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        self.model = model

    def evaluate_retrieval(self, question: str, ground_truth_context: str, retrieved_contexts: List[str]) -> Dict:
        """
        Evaluate retrieval quality by checking if ground truth context is in retrieved contexts.

        Args:
            question: The question being asked
            ground_truth_context: The true relevant context
            retrieved_contexts: List of contexts retrieved by the RAG system

        Returns:
            Dictionary with retrieval metrics
        """
        # Check if ground truth is in top-k retrieved contexts
        retrieval_success = any(
            ground_truth_context.strip() in context or context in ground_truth_context.strip()
            for context in retrieved_contexts
        )

        # Calculate overlap scores
        overlap_scores = []
        for context in retrieved_contexts:
            # Simple word overlap metric
            gt_words = set(ground_truth_context.lower().split())
            ret_words = set(context.lower().split())
            if len(gt_words) > 0:
                overlap = len(gt_words.intersection(ret_words)) / len(gt_words)
                overlap_scores.append(overlap)
            else:
                overlap_scores.append(0.0)

        return {
            'retrieval_success': retrieval_success,
            'max_overlap_score': max(overlap_scores) if overlap_scores else 0.0,
            'top_k': len(retrieved_contexts)
        }

    def evaluate_answer_quality(self, question: str, ground_truth_answer: str,
                                generated_answer: str, context: str) -> Dict:
        """
        Evaluate answer quality using LLM-based evaluation.

        Args:
            question: The question being asked
            ground_truth_answer: The true answer
            generated_answer: The RAG-generated answer
            context: The context used to generate the answer

        Returns:
            Dictionary with answer quality metrics (true/false)
        """
        prompt = f"""你是一位 RAG (檢索增強生成) 系統的專業評估員。

請根據以下標準，以「是/否」(true/false) 來評估生成的答案：

1. 忠實度 (FAITHFULNESS): 答案中的每個論點都忠於原文，即使表達方式不同(是/否)
2. 相關性 (RELEVANCE): 生成的答案是否切合問題？ (是/否)
3. 正確性 (CORRECTNESS): 生成的答案是否使用上下文中的正確資訊，來準確地回答了問題？ (是/否)
4. 解釋 (EXPLANATION): 請簡要說明你的評估理由。

問題：{question}

使用的上下文：{context}

生成的答案：{generated_answer}

請以 JSON 格式提供您的評估：
{{
  "faithfulness": <true/false>,
  "relevance": <true/false>,
  "correctness": <true/false>,
  "explanation": "..."
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            response_text = response.choices[0].message.content

            # Parse JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                evaluation = json.loads(json_str)
                return evaluation
            else:
                print(f"Warning: Could not parse evaluation JSON")
                return {
                    'faithfulness': False,
                    'relevance': False,
                    'correctness': False,
                    'explanation': "Could not parse evaluation JSON"
                }

        except Exception as e:
            print(f"Error evaluating answer: {e}")
            return {
                'faithfulness': False,
                'relevance': False,
                'correctness': False,
                'explanation': f"Error during evaluation: {e}"
            }

    def evaluate_sample(self, sample: Dict, rag_pipeline, evaluate_retrieval: bool = True) -> Dict:
        """
        Evaluate a single sample through the RAG pipeline.

        Args:
            sample: Evaluation sample with question and ground truth
            rag_pipeline: RAG pipeline object with query() method
            evaluate_retrieval: Whether to evaluate retrieval quality (default: True)

        Returns:
            Dictionary with complete evaluation results
        """
        question = sample['question']
        ground_truth_answer = sample['ground_truth_answer']
        ground_truth_context = sample['ground_truth_context']

        # Query the RAG pipeline
        rag_result = rag_pipeline.query(question)

        # Extract generated answer and retrieved contexts
        generated_answer = rag_result.get('answer', '')
        retrieved_contexts = [ref.get('text_preview', ref.get('content', '')) for ref in rag_result.get('sources', [])]

        # Evaluate retrieval (optional)
        if evaluate_retrieval:
            retrieval_metrics = self.evaluate_retrieval(
                question,
                ground_truth_context,
                retrieved_contexts
            )
        else:
            retrieval_metrics = None

        # Evaluate answer quality
        # Use the first retrieved context or empty string
        context_used = retrieved_contexts[0] if retrieved_contexts else ""
        answer_metrics = self.evaluate_answer_quality(
            question,
            ground_truth_answer,
            generated_answer,
            context_used
        )

        result = {
            'sample': sample,
            'generated_answer': generated_answer,
            'retrieved_contexts': retrieved_contexts,
            'answer_metrics': answer_metrics
        }

        # Only include retrieval_metrics if it's not None
        if retrieval_metrics is not None:
            result['retrieval_metrics'] = retrieval_metrics

        return result

    def evaluate_dataset(self, evaluation_samples: List[Dict], rag_pipeline, delay_seconds: float = 6.5) -> Dict:
        """
        Evaluate entire dataset and compute aggregate metrics.

        Args:
            evaluation_samples: List of evaluation samples
            rag_pipeline: RAG pipeline object
            delay_seconds: Delay between samples to respect API rate limits (default: 6.5s for Cohere's 10/min limit)

        Returns:
            Dictionary with aggregate evaluation results
        """
        results = []

        for i, sample in enumerate(evaluation_samples):
            print(f"Evaluating sample {i+1}/{len(evaluation_samples)}...")
            result = self.evaluate_sample(sample, rag_pipeline)
            results.append(result)

            # Add delay between samples to respect Cohere API rate limits
            if i < len(evaluation_samples) - 1:  # Don't delay after the last sample
                print(f"Waiting {delay_seconds} seconds to respect API rate limits...")
                time.sleep(delay_seconds)

        # Compute aggregate metrics
        # Only calculate retrieval success rate if retrieval metrics exist
        results_with_retrieval = [r for r in results if 'retrieval_metrics' in r]
        if results_with_retrieval:
            retrieval_success_rate = sum(
                r['retrieval_metrics']['retrieval_success'] for r in results_with_retrieval
            ) / len(results_with_retrieval)
        else:
            retrieval_success_rate = None

        faithfulness_rate = sum(
            r['answer_metrics']['faithfulness'] for r in results
        ) / len(results)

        relevance_rate = sum(
            r['answer_metrics']['relevance'] for r in results
        ) / len(results)

        correctness_rate = sum(
            r['answer_metrics']['correctness'] for r in results
        ) / len(results)

        aggregate_metrics = {
            'faithfulness_rate': faithfulness_rate,
            'relevance_rate': relevance_rate,
            'correctness_rate': correctness_rate,
            'total_samples': len(results)
        }

        # Only include retrieval_success_rate if it exists
        if retrieval_success_rate is not None:
            aggregate_metrics['retrieval_success_rate'] = retrieval_success_rate

        return {
            'individual_results': results,
            'aggregate_metrics': aggregate_metrics
        }

    def save_evaluation_results(self, results: Dict, output_path: str):
        """
        Save evaluation results to JSON file.

        Args:
            results: Evaluation results dictionary
            output_path: Path to save the results
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Saved evaluation results to {output_path}")
