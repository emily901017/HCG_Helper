"""
Q&A Generation Module

Generates question-answer pairs from document chunks using LLM.
Ensures generated content is faithful to the source material.
"""

from typing import List, Dict, Optional
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

class QAGenerator:
    """Generates Q&A pairs from document chunks for evaluation."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """
        Initialize the Q&A generator.

        Args:
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
            model: OpenAI model to use for generation
        """
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        self.model = model

    def generate_qa_from_chunk(self, chunk_text: str, num_pairs: int = 3) -> List[Dict[str, str]]:
        """
        Generate Q&A pairs from a single chunk.

        Args:
            chunk_text: The text chunk to generate Q&A from
            num_pairs: Number of Q&A pairs to generate (default: 3)

        Returns:
            List of dictionaries with 'question' and 'answer' keys
        """
        prompt = f"""
請扮演一位正在學習公民、地理和歷史的高中生。

請「僅僅」根據下方提供的文本，產生 {num_pairs} 組學生可能會問的問題，以及直接從文本中提取的答案。

重要規則：
1. 問題應力求自然，並反映學生們通常會對這類主題提出的疑問。
2. 答案「必須」只包含所提供文本中出現的資訊。
3. 「請勿」加入任何文本中未提及的外部知識或資訊。
4. 如果一個問題無法單獨從文本中獲得完整解答，請不要將其納入。

請將您的回覆格式化為具有以下結構的 JSON 陣列：
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]

文本：{chunk_text}

請以 JSON 格式產生問答配對：
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # Extract the response text
            response_text = response.choices[0].message.content

            # Try to parse JSON from the response
            # Look for JSON array in the response
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                qa_pairs = json.loads(json_str)
                return qa_pairs
            else:
                print(f"Warning: Could not parse JSON from response: {response_text}")
                return []

        except Exception as e:
            print(f"Error generating Q&A: {e}")
            return []

    def generate_qa_from_chunks(self, chunks: List[Dict], num_pairs_per_chunk: int = 3) -> List[Dict]:
        """
        Generate Q&A pairs from multiple chunks.

        Args:
            chunks: List of chunk dictionaries (from DocumentChunker)
            num_pairs_per_chunk: Number of Q&A pairs per chunk

        Returns:
            List of evaluation samples with chunk info and Q&A pairs
        """
        evaluation_samples = []

        for chunk in chunks:
            chunk_text = chunk['text']
            chunk_id = chunk['chunk_id']
            source_file = chunk['source_file']

            print(f"Generating Q&A for chunk {chunk_id} from {source_file}...")

            qa_pairs = self.generate_qa_from_chunk(chunk_text, num_pairs_per_chunk)

            for qa in qa_pairs:
                evaluation_samples.append({
                    'chunk_id': chunk_id,
                    'source_file': source_file,
                    'ground_truth_context': chunk_text,
                    'question': qa['question'],
                    'ground_truth_answer': qa['answer']
                })

        return evaluation_samples

    def save_evaluation_dataset(self, evaluation_samples: List[Dict], output_path: str):
        """
        Save evaluation dataset to JSON file.

        Args:
            evaluation_samples: List of evaluation samples
            output_path: Path to save the JSON file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_samples, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(evaluation_samples)} evaluation samples to {output_path}")

    def load_evaluation_dataset(self, input_path: str) -> List[Dict]:
        """
        Load evaluation dataset from JSON file.

        Args:
            input_path: Path to the JSON file

        Returns:
            List of evaluation samples
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)
