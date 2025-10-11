"""
Unit test for the evaluator - test with 10 samples
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from eval.evaluator import RAGEvaluator
from eval.qa_generator import QAGenerator
from rag import RAGEngine
import json
import time


def test_multiple_samples():
    """Test evaluation with 10 samples from the dataset"""

    print("=" * 60)
    print("Loading evaluation dataset...")
    print("=" * 60)

    # Load the dataset
    qa_gen = QAGenerator()
    samples = qa_gen.load_evaluation_dataset("eval_dataset.json")

    # Take the first 10 samples
    test_samples = samples

    print(f"\nTesting with {len(test_samples)} samples")

    print("\n" + "=" * 60)
    print("Initializing RAG Pipeline...")
    print("=" * 60)

    # Create RAG pipeline wrapper
    class RAGPipeline:
        def __init__(self):
            self.engine = RAGEngine()

        def query(self, question: str):
            result = self.engine.query(question)
            return result

    rag_pipeline = RAGPipeline()

    print("\n" + "=" * 60)
    print("Evaluating Samples (without retrieval evaluation)...")
    print("=" * 60)

    # Initialize evaluator
    evaluator = RAGEvaluator()

    results = []
    for i, sample in enumerate(test_samples):
        print(f"\n[{i+1}/{len(test_samples)}] Question: {sample['question']}")

        # Evaluate the sample (skip retrieval evaluation)
        result = evaluator.evaluate_sample(sample, rag_pipeline, evaluate_retrieval=False)
        results.append(result)

        # Print metrics
        metrics = result['answer_metrics']
        print(f"  Faithfulness: {metrics['faithfulness']}")
        print(f"  Relevance: {metrics['relevance']}")
        print(f"  Correctness: {metrics['correctness']}")
        print(f"  Explanation: {metrics['explanation']}")

        # Add delay to respect rate limits (except for last sample)
        if i < len(test_samples) - 1:
            print("  Waiting 6.5 seconds...")
            time.sleep(6.5)

    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)

    # Calculate aggregate metrics
    faithfulness_rate = sum(r['answer_metrics']['faithfulness'] for r in results) / len(results)
    relevance_rate = sum(r['answer_metrics']['relevance'] for r in results) / len(results)
    correctness_rate = sum(r['answer_metrics']['correctness'] for r in results) / len(results)

    print(f"Total Samples: {len(results)}")
    print(f"Faithfulness Rate: {faithfulness_rate:.2%}")
    print(f"Relevance Rate: {relevance_rate:.2%}")
    print(f"Correctness Rate: {correctness_rate:.2%}")

    print("\n" + "=" * 60)
    print("Saving test results...")
    print("=" * 60)

    # Save all results
    output = {
        'individual_results': results,
        'aggregate_metrics': {
            'faithfulness_rate': faithfulness_rate,
            'relevance_rate': relevance_rate,
            'correctness_rate': correctness_rate,
            'total_samples': len(results)
        }
    }

    with open("test_evaluation_result_all.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("✓ Test results saved to test_evaluation_result_all.json")


if __name__ == "__main__":
    test_multiple_samples()
