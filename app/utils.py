"""
Utilities Module

This module provides various utility functions and constants
used throughout the application.
"""

import time
import functools
import logging
import asyncio
from typing import Set, List, Callable, Any, TypeVar, Union, cast, Dict

logger = logging.getLogger("text-to-spotify.utils")

# Type variables for generic function typing
T = TypeVar('T')
R = TypeVar('R')

# Common words that are often difficult to match with song titles
COMMON_WORDS: Set[str] = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'to', 'of', 'for', 
    'in', 'on', 'at', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should',
    'can', 'could', 'may', 'might', 'must', 'that', 'which', 'who', 'whom', 'whose',
    'this', 'these', 'those', 'am', 'is', 'are', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
}

def filter_common_words(text: str) -> str:
    """
    Filter out common words from a text string.
    
    Args:
        text: The input text string
        
    Returns:
        A string with common words removed
    """
    words = text.lower().split()
    filtered_words = [word for word in words if word not in COMMON_WORDS]
    return ' '.join(filtered_words)

def is_meaningful_word(word: str) -> bool:
    """
    Check if a word is meaningful (not a common word).
    
    Args:
        word: The word to check
        
    Returns:
        True if the word is meaningful, False otherwise
    """
    return word.lower() not in COMMON_WORDS and len(word) > 1

def filter_meaningful_words(words: List[str]) -> List[str]:
    """
    Filter a list of words to keep only meaningful ones.
    
    Args:
        words: List of words to filter
        
    Returns:
        List of meaningful words
    """
    return [word for word in words if is_meaningful_word(word)]

# Define function types for the timer decorator
AsyncFunc = Callable[..., Any]
SyncFunc = Callable[..., Any]
AnyFunc = Union[AsyncFunc, SyncFunc]

def timer(func: AnyFunc) -> AnyFunc:
    """
    Decorator to time function execution.
    
    This decorator can be used on both synchronous and asynchronous functions.
    It logs the execution time of the decorated function.
    
    Args:
        func: The function to time
        
    Returns:
        Wrapped function that logs execution time
    """
    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        """Wrapper for asynchronous functions."""
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"{func.__name__} took {end_time - start_time:.2f} seconds to execute")
        return result
        
    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        """Wrapper for synchronous functions."""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"{func.__name__} took {end_time - start_time:.2f} seconds to execute")
        return result
    
    # Return the appropriate wrapper based on the function type
    if asyncio.iscoroutinefunction(func):
        return cast(AnyFunc, async_wrapper)
    else:
        return cast(AnyFunc, sync_wrapper)

def truncate_text(text: str, max_length: int = 50) -> str:
    """
    Truncate text to a maximum length and add ellipsis if needed.
    
    Args:
        text: The text to truncate
        max_length: Maximum length before truncation
        
    Returns:
        Truncated text with ellipsis if needed
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts using a simple word overlap approach.
    
    Args:
        text1: First text to compare
        text2: Second text to compare
        
    Returns:
        Similarity score between 0 and 1
    """
    # Convert to lowercase and split into words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Calculate Jaccard similarity
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)

def safe_get(obj: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Safely get a nested value from a dictionary.
    
    Args:
        obj: Dictionary to extract value from
        *keys: Keys to navigate through the dictionary
        default: Default value to return if key is not found
        
    Returns:
        Value at the specified location or default if not found
    """
    try:
        for key in keys:
            obj = obj[key]
        return obj
    except (KeyError, TypeError):
        return default 