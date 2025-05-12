"""
Text Processor Module

This module provides functionality for processing and analyzing text
for the purpose of matching with song titles.
"""

import re
from typing import List, Dict, Optional, Set
import logging

logger = logging.getLogger("text-to-spotify.text_processor")

class TextProcessor:
    """
    Processes text for matching with song titles.
    
    This class provides methods for cleaning and segmenting text
    to optimize matching with Spotify track titles.
    """
    
    def __init__(self):
        """Initialize the text processor."""
        logger.info("TextProcessor initialized")

    def process_text(self, text: str, max_phrase_len: int = 4) -> List[str]:
        """
        Process the input text and return a list of phrases using greedy phrase matching.
        
        Args:
            text: Input text to process
            max_phrase_len: Maximum phrase length to consider
            
        Returns:
            List of extracted phrases from the text
        """
        # Clean the text
        text = self._clean_text(text)
        words = text.split()
        phrases = []
        i = 0
        
        logger.debug(f"Processing text with {len(words)} words, max phrase length: {max_phrase_len}")
        
        while i < len(words):
            matched = False
            for l in range(max_phrase_len, 0, -1):
                if i + l <= len(words):
                    phrase = ' '.join(words[i:i+l])
                    phrases.append(phrase)
                    i += l
                    matched = True
                    logger.debug(f"Added phrase: '{phrase}' (length: {l})")
                    break
            if not matched:
                i += 1
                
        logger.debug(f"Processed text into {len(phrases)} phrases")
        return phrases
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text by lowercasing and removing punctuation.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation except apostrophes (important for song titles)
        text = re.sub(r'[^\w\s\']', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_keywords(self, text: str, common_words: Set[str]) -> List[str]:
        """
        Extract keywords from text by removing common words.
        
        Args:
            text: Text to extract keywords from
            common_words: Set of common words to exclude
            
        Returns:
            List of keywords
        """
        # Clean the text
        cleaned_text = self._clean_text(text)
        
        # Split into words
        words = cleaned_text.split()
        
        # Filter out common words
        keywords = [word for word in words if word not in common_words]
        
        logger.debug(f"Extracted {len(keywords)} keywords from {len(words)} words")
        return keywords
    
    def segment_text(self, text: str, min_segment_length: int = 3, max_segment_length: int = 6) -> List[str]:
        """
        Segment text into meaningful chunks for song matching.
        
        Args:
            text: Text to segment
            min_segment_length: Minimum segment length in words
            max_segment_length: Maximum segment length in words
            
        Returns:
            List of text segments
        """
        # Clean the text
        cleaned_text = self._clean_text(text)
        
        # Split into words
        words = cleaned_text.split()
        
        # Initialize segments list
        segments = []
        
        # Create segments
        for i in range(len(words) - min_segment_length + 1):
            for j in range(min_segment_length, min(max_segment_length + 1, len(words) - i + 1)):
                segment = ' '.join(words[i:i+j])
                segments.append(segment)
        
        logger.debug(f"Generated {len(segments)} segments from text")
        return segments
