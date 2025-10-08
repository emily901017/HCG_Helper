"""
Database module for logging student queries
Stores questions with timestamps for teacher dashboard
"""
import sqlite3
from datetime import datetime
from typing import List, Dict
import config


class QueryLogger:
    """Handles logging and retrieval of student queries"""

    def __init__(self, db_path: str = config.QUERY_LOG_DB):
        """
        Initialize the query logger

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                session_id TEXT,
                subject TEXT,
                response_length INTEGER
            )
        ''')

        conn.commit()
        conn.close()

    def log_query(
        self,
        question: str,
        session_id: str = None,
        subject: str = None,
        response_length: int = None
    ):
        """
        Log a student query to the database

        Args:
            question: The question asked by the student
            session_id: Optional session identifier
            subject: Optional subject category
            response_length: Optional length of the response
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO queries (question, timestamp, session_id, subject, response_length)
            VALUES (?, ?, ?, ?, ?)
        ''', (question, datetime.now(), session_id, subject, response_length))

        conn.commit()
        conn.close()

    def get_all_queries(self) -> List[Dict]:
        """
        Retrieve all logged queries

        Returns:
            List of dictionaries containing query data
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, question, timestamp, session_id, subject, response_length
            FROM queries
            ORDER BY timestamp DESC
        ''')

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_query_statistics(self) -> Dict:
        """
        Get statistics about logged queries

        Returns:
            Dictionary containing query statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total queries
        cursor.execute('SELECT COUNT(*) FROM queries')
        total_queries = cursor.fetchone()[0]

        # Queries by subject
        cursor.execute('''
            SELECT subject, COUNT(*) as count
            FROM queries
            WHERE subject IS NOT NULL
            GROUP BY subject
        ''')
        by_subject = dict(cursor.fetchall())

        # Recent activity (last 7 days)
        cursor.execute('''
            SELECT COUNT(*) FROM queries
            WHERE timestamp >= datetime('now', '-7 days')
        ''')
        recent_queries = cursor.fetchone()[0]

        conn.close()

        return {
            'total_queries': total_queries,
            'by_subject': by_subject,
            'recent_queries_7days': recent_queries
        }

    def search_queries(self, keyword: str) -> List[Dict]:
        """
        Search queries by keyword

        Args:
            keyword: Keyword to search for

        Returns:
            List of matching queries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, question, timestamp, session_id, subject, response_length
            FROM queries
            WHERE question LIKE ?
            ORDER BY timestamp DESC
        ''', (f'%{keyword}%',))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_common_keywords(self, limit: int = 50) -> List[tuple]:
        """
        Extract most common keywords from queries

        Args:
            limit: Number of top keywords to return

        Returns:
            List of (keyword, count) tuples
        """
        from collections import Counter
        import re

        queries = self.get_all_queries()

        # Extract words from all questions
        all_words = []
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                      'of', 'with', 'is', 'was', 'are', 'were', 'what', 'how', 'why', 'when',
                      'where', 'who', 'which', '?', '!', '.', ','}

        for query in queries:
            words = re.findall(r'\w+', query['question'].lower())
            all_words.extend([w for w in words if w not in stop_words and len(w) > 2])

        # Count and return most common
        counter = Counter(all_words)
        return counter.most_common(limit)
