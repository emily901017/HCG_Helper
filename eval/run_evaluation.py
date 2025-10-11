"""
Main Evaluation Pipeline Script

Run the complete evaluation pipeline:
1. Chunk documents
2. Generate Q&A pairs
3. Evaluate RAG pipeline
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path to import from main project
sys.path.append(str(Path(__file__).parent.parent))

from eval.chunker import DocumentChunker
from eval.qa_generator import QAGenerator
from eval.evaluator import RAGEvaluator


def generate_evaluation_dataset(data_dir: str, output_path: str,
                                chunk_size: int = 2048,
                                qa_pairs_per_chunk: int = 1):
    """
    Generate evaluation dataset from documents.

    Args:
        data_dir: Directory containing source documents
        output_path: Path to save the evaluation dataset
        chunk_size: Size of each chunk in characters
        qa_pairs_per_chunk: Number of Q&A pairs to generate per chunk
    """
    print("=" * 60)
    print("STEP 1: Chunking Documents")
    print("=" * 60)

    chunker = DocumentChunker(chunk_size=chunk_size, overlap=128)
    all_chunks = []

    # Chunk all documents in the data directory
    chunks_by_file = chunker.chunk_directory(data_dir, file_extension='.txt')

    for filename, chunks in chunks_by_file.items():
        print(f"\n{filename}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"\nTotal chunks: {len(all_chunks)}")

    print("\n" + "=" * 60)
    print("STEP 2: Generating Q&A Pairs")
    print("=" * 60)

    qa_gen = QAGenerator()
    evaluation_samples = qa_gen.generate_qa_from_chunks(all_chunks, qa_pairs_per_chunk)

    print(f"\nGenerated {len(evaluation_samples)} evaluation samples")

    # Save the dataset
    qa_gen.save_evaluation_dataset(evaluation_samples, output_path)

    return evaluation_samples


def run_evaluation(dataset_path: str, output_path: str):
    """
    Run evaluation on the RAG pipeline.

    Args:
        dataset_path: Path to the evaluation dataset JSON
        output_path: Path to save evaluation results
    """
    print("\n" + "=" * 60)
    print("STEP 3: Running RAG Pipeline Evaluation")
    print("=" * 60)

    # Load evaluation dataset
    qa_gen = QAGenerator()
    evaluation_samples = qa_gen.load_evaluation_dataset(dataset_path)

    print(f"Loaded {len(evaluation_samples)} evaluation samples")

    # Import RAG pipeline
    try:
        from rag import RAGEngine

        # Create a wrapper class for the RAG pipeline
        class RAGPipeline:
            def __init__(self):
                self.engine = RAGEngine()

            def query(self, question: str):
                result = self.engine.query(question)
                return result

        rag_pipeline = RAGPipeline()

    except ImportError as e:
        print(f"Error: Could not import RAG pipeline: {e}")
        print("Make sure the RAG pipeline is properly set up")
        return
    except Exception as e:
        print(f"Error initializing RAG engine: {e}")
        print("Make sure ChromaDB is initialized and dependencies are installed")
        return

    # Run evaluation
    evaluator = RAGEvaluator()
    results = evaluator.evaluate_dataset(evaluation_samples, rag_pipeline)

    # Print aggregate metrics
    metrics = results['aggregate_metrics']
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total Samples: {metrics['total_samples']}")
    if 'retrieval_success_rate' in metrics:
        print(f"\nRetrieval Success Rate: {metrics['retrieval_success_rate']:.2%}")
    print(f"Faithfulness Rate: {metrics['faithfulness_rate']:.2%}")
    print(f"Relevance Rate: {metrics['relevance_rate']:.2%}")
    print(f"Correctness Rate: {metrics['correctness_rate']:.2%}")

    # Save results
    evaluator.save_evaluation_results(results, output_path)


def main():
    parser = argparse.ArgumentParser(
        description='RAG Pipeline Evaluation Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Generate dataset command
    gen_parser = subparsers.add_parser('generate', help='Generate evaluation dataset')
    gen_parser.add_argument('--data-dir', required=True, help='Directory containing source documents')
    gen_parser.add_argument('--output', default='eval_dataset.json', help='Output path for dataset')
    gen_parser.add_argument('--chunk-size', type=int, default=2048, help='Chunk size in characters')
    gen_parser.add_argument('--qa-pairs', type=int, default=1, help='Q&A pairs per chunk')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate RAG pipeline')
    eval_parser.add_argument('--dataset', required=True, help='Path to evaluation dataset')
    eval_parser.add_argument('--output', default='eval_results.json', help='Output path for results')

    # Full pipeline command
    full_parser = subparsers.add_parser('full', help='Run full pipeline (generate + evaluate)')
    full_parser.add_argument('--data-dir', required=True, help='Directory containing source documents')
    full_parser.add_argument('--dataset', default='eval_dataset.json', help='Path for dataset')
    full_parser.add_argument('--results', default='eval_results.json', help='Path for results')
    full_parser.add_argument('--chunk-size', type=int, default=512, help='Chunk size in characters')
    full_parser.add_argument('--qa-pairs', type=int, default=3, help='Q&A pairs per chunk')

    args = parser.parse_args()

    if args.command == 'generate':
        generate_evaluation_dataset(
            args.data_dir,
            args.output,
            args.chunk_size,
            args.qa_pairs
        )

    elif args.command == 'evaluate':
        run_evaluation(args.dataset, args.output)

    elif args.command == 'full':
        # Generate dataset
        generate_evaluation_dataset(
            args.data_dir,
            args.dataset,
            args.chunk_size,
            args.qa_pairs
        )
        # Run evaluation
        run_evaluation(args.dataset, args.results)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
