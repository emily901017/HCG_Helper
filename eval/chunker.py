"""
Document Chunking Module

Handles chunking of textbook documents into fixed-size segments
for Q&A generation and evaluation.
"""

from typing import List, Dict
import os


class DocumentChunker:
    """Chunks documents into fixed-size segments for evaluation."""

    def __init__(self, chunk_size: int = 2048, overlap: int = 128):
        """
        Initialize the document chunker.

        Args:
            chunk_size: Size of each chunk in characters
            overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text into fixed-size segments with overlap.

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks
        """
        if not text:
            return []

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size

            # Extract chunk
            chunk = text[start:end]

            # Try to break at sentence boundary if possible
            if end < text_length:
                # Look for sentence endings in the last 100 characters
                last_period = chunk.rfind('.')
                last_question = chunk.rfind('?')
                last_exclamation = chunk.rfind('!')

                sentence_end = max(last_period, last_question, last_exclamation)

                # If we found a sentence ending in the last portion, use it
                if sentence_end > self.chunk_size - 100:
                    chunk = chunk[:sentence_end + 1]
                    end = start + sentence_end + 1

            chunks.append(chunk.strip())

            # Move to next chunk with overlap
            start = end - self.overlap

        return chunks

    def chunk_file(self, file_path: str) -> List[Dict[str, any]]:
        """
        Chunk a document file into segments.

        Args:
            file_path: Path to the document file

        Returns:
            List of dictionaries containing chunk info
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        chunks = self.chunk_text(text)

        return [
            {
                'chunk_id': i,
                'text': chunk,
                'source_file': os.path.basename(file_path),
                'chunk_size': len(chunk)
            }
            for i, chunk in enumerate(chunks)
        ]

    def chunk_directory(self, directory: str, file_extension: str = '.txt') -> Dict[str, List[Dict]]:
        """
        Chunk all files in a directory.

        Args:
            directory: Path to directory containing documents
            file_extension: File extension to filter (default: .txt)

        Returns:
            Dictionary mapping file names to their chunks
        """
        results = {}

        for filename in os.listdir(directory):
            if filename.endswith(file_extension):
                file_path = os.path.join(directory, filename)
                chunks = self.chunk_file(file_path)
                results[filename] = chunks

        return results
