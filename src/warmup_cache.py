"""
Warm-up Question Cache Module

Handles caching and retrieval of warm-up question answers to improve
chatbot initialization performance and reduce API calls.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from config import WARMUP_CACHE_CONFIG


class WarmupCache:
    """
    Manages caching of warm-up question answers
    """
    
    def __init__(self, cache_file: str = None):
        """
        Initialize warm-up cache
        
        Args:
            cache_file: Path to cache file
        """
        self.cache_file = cache_file or WARMUP_CACHE_CONFIG["cache_file"]
        self.cache_data = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """
        Load cache data from file
        
        Returns:
            Cache data dictionary
        """
        if not os.path.exists(self.cache_file):
            return {
                "created_at": datetime.now().isoformat(),
                "answers": {}
            }
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data
        except (json.JSONDecodeError, FileNotFoundError):
            return {
                "created_at": datetime.now().isoformat(),
                "answers": {}
            }
    
    def _save_cache(self):
        """
        Save cache data to file
        """
        # Ensure cache directory exists
        cache_dir = os.path.dirname(self.cache_file)
        os.makedirs(cache_dir, exist_ok=True)
        
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache_data, f, indent=2, ensure_ascii=False)
    
    def _is_cache_valid(self) -> bool:
        """
        Check if cache exists and has answers
        
        Returns:
            True if cache has answers, False otherwise
        """
        return len(self.cache_data.get("answers", {})) > 0
    
    def get_answer(self, question: str) -> Optional[str]:
        """
        Get cached answer for a question
        
        Args:
            question: The warm-up question
            
        Returns:
            Cached answer if available and valid, None otherwise
        """
        if not self._is_cache_valid():
            return None
        
        return self.cache_data.get("answers", {}).get(question)
    
    def store_answer(self, question: str, answer: str):
        """
        Store answer for a question
        
        Args:
            question: The warm-up question
            answer: The answer to cache
        """
        if "answers" not in self.cache_data:
            self.cache_data["answers"] = {}
        
        self.cache_data["answers"][question] = answer
        self._save_cache()
    
    def store_answers(self, question_answer_pairs: List[Tuple[str, str]]):
        """
        Store multiple question-answer pairs
        
        Args:
            question_answer_pairs: List of (question, answer) tuples
        """
        if "answers" not in self.cache_data:
            self.cache_data["answers"] = {}
        
        for question, answer in question_answer_pairs:
            self.cache_data["answers"][question] = answer
        
        self._save_cache()
    
    def get_cached_answers(self, questions: List[str]) -> Dict[str, Optional[str]]:
        """
        Get cached answers for multiple questions
        
        Args:
            questions: List of questions to look up
            
        Returns:
            Dictionary mapping questions to their cached answers (None if not cached)
        """
        if not self._is_cache_valid():
            return {q: None for q in questions}
        
        answers = {}
        for question in questions:
            answers[question] = self.cache_data.get("answers", {}).get(question)
        
        return answers
    
    def clear_cache(self):
        """
        Clear all cached data
        """
        self.cache_data = {
            "created_at": datetime.now().isoformat(),
            "answers": {}
        }
        self._save_cache()
    
    def get_cache_stats(self) -> Dict:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        total_answers = len(self.cache_data.get("answers", {}))
        
        return {
            "valid": total_answers > 0,
            "total_answers": total_answers,
            "created_at": self.cache_data.get("created_at"),
            "age_days": None  # No longer tracking age
        }
    
    def is_available(self) -> bool:
        """
        Check if cache is available
        
        Returns:
            True if cache has answers
        """
        return self._is_cache_valid()


def create_warmup_cache() -> WarmupCache:
    """
    Create a warm-up cache instance with default configuration
    
    Returns:
        WarmupCache instance
    """
    return WarmupCache(
        cache_file=WARMUP_CACHE_CONFIG["cache_file"]
    )