"""
Text to Spotify Playlist Generator - Main Application Module

This module contains the main FastAPI application and endpoints
for converting text to Spotify playlists.
"""

from fastapi import FastAPI, Request, Form, HTTPException, Body, Depends
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, JSONResponse
import os
from dotenv import load_dotenv
from .spotify_client import SpotifyClient
from .text_processor import TextProcessor
from .utils import (
    filter_common_words, 
    is_meaningful_word, 
    filter_meaningful_words, 
    timer, 
    truncate_text, 
    COMMON_WORDS
)
from typing import List, Dict, Optional, Any, Union
import re
import random
import requests
import json
import logging
import time
from logging.handlers import RotatingFileHandler
from pydantic import BaseModel

# Configure logging
def setup_logging() -> logging.Logger:
    """
    Configure application logging with console and file handlers.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_dir = os.getenv("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger("text-to-spotify")
    logger.setLevel(getattr(logging, log_level))
    
    # Only add handlers if they don't exist to prevent duplicate logging
    if not logger.handlers:
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level))

        # Create file handler
        file_handler = RotatingFileHandler(
            f"{log_dir}/app.log", 
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(getattr(logging, log_level))

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger

# Load environment variables
load_dotenv()

# Setup logger
logger = setup_logging()

# Define models for request/response validation
class SongItem(BaseModel):
    """Song item data model for playlist creation"""
    id: str
    name: str
    artist: str
    source_phrase: Optional[str] = None

class PlaylistRequest(BaseModel):
    """Playlist creation request model"""
    songs: List[SongItem]
    text: str

# Create FastAPI app
app = FastAPI(
    title="Text to Spotify Playlist Generator",
    description="Convert text into Spotify playlists with matching song titles",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Templates setup
templates = Jinja2Templates(directory="app/templates")

# Initialize a global spotify client
# This ensures the same instance is used across requests
spotify_client = SpotifyClient(
    client_id=os.getenv("SPOTIPY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
    redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI")
)

# Initialize text processor
text_processor = TextProcessor()

# Application configuration
PARAPHRASE_PROVIDER = os.getenv("PARAPHRASE_PROVIDER", "none").lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

# Initialize spaCy if available
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
    logger.info("spaCy model loaded successfully")
except Exception as e:
    logger.error(f"spaCy not available or model not loaded: {e}")
    SPACY_AVAILABLE = False

@timer
def generate_paraphrases(text: str, n: int = 3) -> List[str]:
    """
    Generate paraphrases using Ollama or OpenAI, with specific guidance for song-friendly phrases.
    
    Args:
        text: The text to paraphrase
        n: Number of paraphrases to generate
        
    Returns:
        List of paraphrased texts, or the original text if paraphrasing fails
    """
    logger.info(f"Generating paraphrases for: '{truncate_text(text, 50)}'")
    
    # If text is very short, just return it without paraphrasing
    if len(text.split()) <= 3:
        logger.info(f"Text is too short, skipping paraphrasing")
        return [text]
    
    # Calculate original text length for prompt constraints
    original_word_count = len(text.split())
    min_words = max(3, original_word_count - 2)
    max_words = original_word_count + 2
    
    prompt = (
        f"Generate {n} different ways to express the same sentiment as this text. "
        f"Each paraphrase should express the same meaning, but with different wording. "
        f"These will be used to create a Spotify playlist where song titles will be matched to words/phrases, "
        f"so use common words and phrases that might appear in song titles.\n\n"
        f"Original text: {text}\n\n"
        f"Requirements:\n"
        f"- Return ONLY the paraphrases, one per line, with no numbering, explanations, or other text\n"
        f"- Each paraphrase MUST be similar in length to the original ({original_word_count} words)\n"
        f"- Each paraphrase should have between {min_words} and {max_words} words\n"
        f"- Keep the same emotional tone and meaning\n"
        f"- Use natural, everyday language\n"
        f"- Use words and phrases commonly found in song titles like 'love', 'heart', 'baby', etc.\n"
        f"- Maintain the same level of formality\n"
        f"- Each paraphrase should be a complete, grammatically correct sentence\n"
        f"- Vary the structure and vocabulary across the different paraphrases\n"
        f"- Include some phrases like 'I want', 'you are', 'love is', etc. that are common in songs\n"
    )
    
    # Better fallback variations to use if API calls fail
    fallback_variations = [
        text,
        text.replace("I", "You").replace("my", "your"),
        "I " + text if not text.startswith("I") else text
    ]
    
    if PARAPHRASE_PROVIDER == "ollama":
        return _generate_paraphrases_ollama(prompt, text, n, fallback_variations)
    elif PARAPHRASE_PROVIDER == "openai" and OPENAI_API_KEY:
        return _generate_paraphrases_openai(prompt, text, n, fallback_variations)
    else:
        logger.info(f"No paraphrasing provider configured, using original text")
        return [text]

def _generate_paraphrases_ollama(prompt: str, text: str, n: int, fallbacks: List[str]) -> List[str]:
    """
    Generate paraphrases using Ollama.
    
    Args:
        prompt: The prompt to send to Ollama
        text: The original text (for logging)
        n: Number of paraphrases to generate
        fallbacks: Fallback paraphrases if generation fails
        
    Returns:
        List of paraphrased texts, or fallbacks if generation fails
    """
    logger.info(f"Using Ollama for paraphrasing with model: {OLLAMA_MODEL}")
    try:
        logger.debug(f"Sending request to Ollama at: {OLLAMA_BASE_URL}")
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.8,
                "top_p": 0.95,
                "timeout": 20000  # 20 seconds timeout
            },
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        raw = result.get("response", "").strip()
        logger.debug(f"Ollama raw response length: {len(raw)} chars")
        
        return _extract_paraphrases(raw, n, fallbacks)
    except Exception as e:
        logger.error(f"Ollama paraphrasing failed: {str(e)}")
        return fallbacks[:n]

def _generate_paraphrases_openai(prompt: str, text: str, n: int, fallbacks: List[str]) -> List[str]:
    """
    Generate paraphrases using OpenAI.
    
    Args:
        prompt: The prompt to send to OpenAI
        text: The original text (for logging)
        n: Number of paraphrases to generate
        fallbacks: Fallback paraphrases if generation fails
        
    Returns:
        List of paraphrased texts, or fallbacks if generation fails
    """
    logger.info(f"Using OpenAI for paraphrasing with model: {OPENAI_MODEL}")
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        logger.debug(f"Sending request to OpenAI")
        
        completion = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates natural, conversational paraphrases while maintaining the original meaning and emotional tone. Your paraphrases should use words and phrases commonly found in song titles to help with matching. Return ONLY the paraphrases, one per line, with no numbering, explanations, or other text."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=256,
            temperature=0.8,
            top_p=0.95,
            request_timeout=20
        )
        
        content = completion.choices[0].message.content
        logger.debug(f"OpenAI response length: {len(content)} chars")
        
        return _extract_paraphrases(content, n, fallbacks)
    except Exception as e:
        logger.error(f"OpenAI paraphrasing failed: {str(e)}")
        return fallbacks[:n]

def _extract_paraphrases(raw_text: str, n: int, fallbacks: List[str]) -> List[str]:
    """
    Extract and clean paraphrases from raw API response text.
    
    Args:
        raw_text: Raw text response from API
        n: Maximum number of paraphrases to return
        fallbacks: Fallback paraphrases if extraction fails
        
    Returns:
        List of cleaned and filtered paraphrases
    """
    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
    logger.debug(f"Extracted {len(lines)} raw lines from response")
    
    # Remove duplicates and clean lines
    seen = set()
    unique_lines = []
    
    for line in lines:
        # Clean up any numbering or bullet points
        clean_line = re.sub(r'^[\d\-\.\*]+\s*', '', line).strip()
        
        # Skip lines that look like instructions or explanations
        if any(word in clean_line.lower() for word in ["variation", "paraphrase", "original text", "here are", "example"]):
            continue
            
        if clean_line and clean_line not in seen:
            unique_lines.append(clean_line)
            seen.add(clean_line)
            
    logger.debug(f"Final unique paraphrases: {len(unique_lines)}")
    return unique_lines[:n] if unique_lines else fallbacks[:n]

@timer
async def greedy_sentence_to_playlist(
    paraphrase: str, 
    max_phrase_len: int = 6, 
    max_results: int = 5, 
    allow_overlap: bool = True
) -> Dict:
    """
    Convert a sentence to a playlist using a greedy algorithm.
    
    Args:
        paraphrase: Text to convert to playlist
        max_phrase_len: Maximum phrase length to consider
        max_results: Maximum results per phrase
        allow_overlap: Whether to allow overlapping phrases
        
    Returns:
        Dictionary with playlist data and coverage information
    """
    global spotify_client
    logger.info(f"Converting text to playlist: '{truncate_text(paraphrase, 50)}'")
    
    # Clean the input text
    text = paraphrase.lower()
    words = text.split()
    
    # Track which words are covered
    word_coverage = [False] * len(words)
    
    # Track which parts are matched with which songs
    matched_songs = []
    
    # Variables to track progress
    i = 0
    skipped_indices = set()
    
    # Process input with decreasing phrase length, from max_phrase_len down to 1
    while i < len(words):
        if i in skipped_indices:
            i += 1
            continue
            
        found_match = False
        
        # Try phrases of decreasing length
        for phrase_len in range(min(max_phrase_len, len(words) - i), 0, -1):
            # Skip if we've already processed this exact position with this length
            if all(word_coverage[i:i+phrase_len]):
                continue
                
            # Extract the phrase
            current_phrase = ' '.join(words[i:i+phrase_len])
            
            # Skip common single words
            if phrase_len == 1 and current_phrase in COMMON_WORDS:
                skipped_indices.add(i)
                i += 1
                break
                
            logger.debug(f"Searching for phrase: '{current_phrase}'")
            
            # Search for songs matching the phrase
            song_matches = await spotify_client.search_multiple_tracks(
                current_phrase, 
                max_results=max_results
            )
            
            if song_matches:
                logger.debug(f"Found {len(song_matches)} matches for '{current_phrase}'")
                
                # Add song info with the matched phrase
                for song in song_matches:
                    # Add source phrase to the song data
                    song["source_phrase"] = current_phrase
                    song["source_indices"] = list(range(i, i+phrase_len))
                
                # Mark these words as covered
                for j in range(i, i+phrase_len):
                    word_coverage[j] = True
                
                # Add songs to results
                matched_songs.append({
                    "phrase": current_phrase,
                    "songs": song_matches,
                    "phrase_len": phrase_len,
                    "indices": list(range(i, i+phrase_len))
                })
                
                # Advance index by the phrase length
                i += phrase_len
                found_match = True
                break
        
        # If no match found for any phrase length, move to the next word
        if not found_match:
            i += 1
    
    # Calculate coverage statistics
    total_words = len(words)
    covered_words = sum(word_coverage)
    coverage_percentage = round((covered_words / total_words) * 100) if total_words > 0 else 0
    
    logger.info(f"Playlist generation complete - Covered {covered_words}/{total_words} words ({coverage_percentage}%)")
    
    # Return the playlist data and coverage information
    return {
        "matched_songs": matched_songs,
        "word_coverage": word_coverage,
        "total_words": total_words,
        "covered_words": covered_words,
        "coverage_percentage": coverage_percentage,
        "words": words,
        "paraphrase": paraphrase
    }

@timer
async def find_additional_matches(paraphrase: str, existing_playlist: list, max_results: int = 3):
    """
    Find additional song matches for words/phrases regardless of whether they are already covered.
    This ensures we have multiple song options for each part of the text.
    
    Args:
        paraphrase: The text to find matches for
        existing_playlist: The playlist of songs already matched
        max_results: Maximum number of results per word/phrase
        
    Returns:
        List of additional song matches
    """
    global spotify_client
    logger.info(f"Finding additional matches for: '{truncate_text(paraphrase, 30)}'")
    
    # Normalize the text but preserve apostrophes for song matching
    normalized_paraphrase = paraphrase.lower().replace(',', ' ').replace('.', ' ').replace('!', ' ').replace('?', ' ')
    normalized_paraphrase = re.sub(r'\s+', ' ', normalized_paraphrase).strip()
    
    # Split into words, treating apostrophes as part of words
    words = re.findall(r"[\w']+", normalized_paraphrase)
    
    # Keep track of existing song IDs to avoid duplicates
    existing_song_ids = {item['song']['id'] for item in existing_playlist if 'song' in item}
    
    # Keep track of phrases we've already matched in the existing playlist
    existing_phrases = {item['phrase'].lower() for item in existing_playlist}
    
    additional_matches = []
    
    # Try combinations of 2-3 words that might not have been matched yet
    phrase_lengths = [3, 2, 1]
    max_additional_per_phrase = 2  # Limit number of additional matches per phrase
    
    for phrase_len in phrase_lengths:
        if phrase_len > len(words):
            continue
            
        for i in range(len(words) - phrase_len + 1):
            phrase = ' '.join(words[i:i+phrase_len])
            
            # Skip if not a meaningful phrase (only common words)
            if phrase_len == 1 and not is_meaningful_word(phrase):
                continue
                
            # Skip exact matches to existing phrases to avoid duplicating work
            if phrase.lower() in existing_phrases:
                continue
                
            # For single words, only search for meaningful ones to reduce API calls
            if phrase_len == 1 and not is_meaningful_word(phrase):
                continue
                
            logger.debug(f"Searching for additional matches for '{phrase}'")
            
            matches = await spotify_client.search_multiple_tracks(phrase, max_results=max_results + 2)
            
            # Filter out songs we've already used
            matches = [match for match in matches if match['id'] not in existing_song_ids]
            
            # Take only top matches up to the limit
            good_matches = []
            for match in matches[:max_additional_per_phrase]:
                if match['score'] > 65:  # Lower threshold for additional matches
                    logger.debug(f"Additional match for '{phrase}': '{match['name']}' (score: {match['score']})")
                    good_matches.append({
                        "phrase": phrase,
                        "song": match,
                        "is_additional": True  # Mark as an additional match
                    })
                    existing_song_ids.add(match['id'])
            
            additional_matches.extend(good_matches)
    
    logger.info(f"Found {len(additional_matches)} additional matches")
    return additional_matches

@app.post("/generate-playlist")
@timer
async def generate_playlist(text: str = Form(...)):
    """Generate playlist candidates for multiple paraphrased inputs using improved algorithm."""
    global spotify_client
    try:
        logger.info(f"Starting playlist generation for text: '{truncate_text(text, 30)}...'")
        
        # Clean text but preserve important punctuation as spaces and apostrophes
        text_clean = text.lower()
        # Replace punctuation except apostrophes with spaces
        text_clean = re.sub(r'[^\w\s\']', ' ', text_clean)
        # Replace multiple spaces with single space
        text_clean = re.sub(r'\s+', ' ', text_clean).strip()
        
        # Properly handle contractions and possessives with apostrophes
        text_clean = re.sub(r'\s\'', '\'', text_clean)  # Fix cases like "don ' t" to "don't"
        
        logger.info(f"Cleaned text: '{truncate_text(text_clean, 50)}'")
        
        # Generate paraphrases
        logger.info(f"Generating paraphrases using provider: {PARAPHRASE_PROVIDER}")
        paraphrases = generate_paraphrases(text_clean, n=2)
        logger.info(f"Generated {len(paraphrases)} paraphrases: {paraphrases}")
        
        # Always place the original text first in the list
        all_paraphrases = [text_clean] + [p for p in paraphrases if p != text_clean]
        all_paraphrases = list(dict.fromkeys(all_paraphrases))  # Remove any remaining duplicates
        logger.info(f"Total unique paraphrases: {len(all_paraphrases)}, with original text first")
        
        all_playlist_candidates = []
        
        # Try different phrase lengths for different paraphrases
        phrase_lengths = [4, 3, 2]
        
        for i, paraphrase in enumerate(all_paraphrases):
            logger.info(f"Processing paraphrase {i+1}/{len(all_paraphrases)}: '{truncate_text(paraphrase, 50)}'")
            
            # Try only one phrase length per paraphrase to reduce API calls
            max_len = phrase_lengths[i % len(phrase_lengths)]
            logger.info(f"Using max phrase length: {max_len}")
            
            # Generate playlist with strict matching first
            playlist_data = await greedy_sentence_to_playlist(
                paraphrase, 
                max_phrase_len=max_len, 
                max_results=5,
                allow_overlap=False
            )
            
            # Extract matched songs and calculate hard-to-match words
            playlist = playlist_data.get("matched_songs", [])
            word_coverage = playlist_data.get("word_coverage", [])
            words = playlist_data.get("words", [])
            
            # Determine hard to match words (words that aren't covered)
            hard_to_match = [words[i] for i, covered in enumerate(word_coverage) if not covered]
            
            # Then find additional matches for individual words/phrases regardless of coverage
            additional_matches = await find_additional_matches(
                paraphrase,
                existing_playlist=playlist,
                max_results=3
            )
            
            if additional_matches:
                playlist.extend(additional_matches)
                logger.info(f"Added {len(additional_matches)} additional song matches")
            
            logger.info(f"Found total of {len(playlist)} songs with {len(hard_to_match)} hard-to-match words")
            
            # Calculate coverage (percentage of words matched)
            meaningful_words = [word for i, word in enumerate(words) if is_meaningful_word(word)]
            meaningful_word_count = len(meaningful_words) or 1  # Avoid division by zero
            
            # Calculate coverage based on meaningful words only
            matched_meaningful_words = sum(1 for i, word in enumerate(words) if word_coverage[i] and is_meaningful_word(word))
            coverage = matched_meaningful_words / meaningful_word_count
            logger.info(f"Coverage: {coverage:.2f} ({matched_meaningful_words}/{meaningful_word_count} meaningful words)")
            
            # Generate advice for hard-to-match words
            advice = ""
            if hard_to_match:
                meaningful_hard_words = [w for w in hard_to_match if is_meaningful_word(w)]
                if len(meaningful_hard_words) > 5:
                    advice = f"Try rephrasing your text to avoid hard-to-match words."
                elif meaningful_hard_words:
                    advice = f"Try rephrasing these words for better matches: {', '.join(meaningful_hard_words)}."
                logger.info(f"Advice: {advice}")
            
            # Mark if this is the original text
            is_original = (i == 0)
            
            all_playlist_candidates.append({
                "paraphrase": paraphrase,
                "playlist": playlist,
                "coverage": coverage,
                "advice": advice,
                "is_original": is_original
            })
        
        # Sort candidates by putting original first, then by coverage
        all_playlist_candidates.sort(key=lambda x: (not x["is_original"], -x["coverage"]))
        logger.info(f"Sorted {len(all_playlist_candidates)} playlist candidates, original text first")
        
        return JSONResponse({
            "playlist_candidates": all_playlist_candidates
        })
    except Exception as e:
        logger.error(f"Exception in generate_playlist: {str(e)}")
        if "not initialized" in str(e):
            return RedirectResponse(url="/login")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create-playlist")
@timer
async def create_playlist(songs: List[Dict] = Body(...), text: str = Body(...)):
    """Create a Spotify playlist from selected songs."""
    global spotify_client
    try:
        if not spotify_client.sp:
            logger.warning("Spotify client not initialized, redirecting to login")
            return RedirectResponse(url="/login")
        
        logger.info(f"Creating playlist with {len(songs)} songs for text: '{truncate_text(text, 30)}'")
        
        # Create the playlist
        playlist_url = await spotify_client.create_playlist(songs, text[:50])
        logger.info(f"Playlist created successfully: {playlist_url}")
        
        return JSONResponse({"playlist_url": playlist_url})
    except Exception as e:
        logger.error(f"Error creating playlist: {str(e)}")
        if "not initialized" in str(e):
            return RedirectResponse(url="/login")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    global spotify_client
    try:
        # Check if Spotify client is properly configured
        spotify_configured = bool(os.getenv("SPOTIPY_CLIENT_ID")) and bool(os.getenv("SPOTIPY_CLIENT_SECRET"))
        
        # Check if Spotify client is authenticated
        spotify_authenticated = bool(spotify_client.sp)
        
        # Check if paraphrasing is configured
        paraphrase_configured = PARAPHRASE_PROVIDER != "none"
        if PARAPHRASE_PROVIDER == "ollama":
            paraphrase_details = f"Ollama ({OLLAMA_MODEL})"
        elif PARAPHRASE_PROVIDER == "openai":
            paraphrase_details = f"OpenAI ({OPENAI_MODEL})"
        else:
            paraphrase_details = "None"
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "spotify_configured": spotify_configured,
            "spotify_authenticated": spotify_authenticated,
            "paraphrase_provider": paraphrase_details,
            "spacy_available": SPACY_AVAILABLE,
            "debug_mode": DEBUG_MODE
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }

async def get_best_song_candidates(phrase: str, max_phrase_len: int = 4, max_results: int = 5):
    """
    DEPRECATED - This function is no longer used as it makes too many API calls.
    
    Args:
        phrase: The phrase to search for
        max_phrase_len: Maximum phrase length to consider
        max_results: Maximum number of results to return
        
    Returns:
        List of song matches for the phrase
    """
    global spotify_client
    logger.warning(f"Deprecated function get_best_song_candidates called with phrase: {phrase}")
    return await spotify_client.search_multiple_tracks(phrase, max_results=max_results)

def split_into_ngrams(text: str, ngram_max: int = 1) -> List[str]:
    """
    Split text into n-grams up to ngram_max, preserving order.
    
    Args:
        text: Text to split into n-grams
        ngram_max: Maximum n-gram size
        
    Returns:
        List of n-grams extracted from text
    """
    words = text.split()
    ngrams = []
    i = 0
    
    while i < len(words):
        found = False
        for n in range(ngram_max, 0, -1):
            if i + n <= len(words):
                ngram = ' '.join(words[i:i+n])
                ngrams.append(ngram)
                i += n
                found = True
                break
        if not found:
            i += 1
            
    return ngrams

@app.get("/")
async def home(request: Request):
    """
    Render the home page.
    
    Args:
        request: FastAPI request object
        
    Returns:
        TemplateResponse with the rendered index.html
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/login")
async def login():
    """
    Redirect to Spotify login page.
    
    Returns:
        RedirectResponse to Spotify authorization URL
    """
    global spotify_client
    auth_url = spotify_client.get_auth_url()
    return RedirectResponse(url=auth_url)

@app.get("/callback")
async def callback(code: str):
    """
    Handle Spotify OAuth callback.
    
    Args:
        code: Authorization code from Spotify
        
    Returns:
        RedirectResponse to the home page
        
    Raises:
        HTTPException: If authentication fails
    """
    global spotify_client
    try:
        await spotify_client.handle_callback(code)
        return RedirectResponse(url="/")
    except Exception as e:
        logger.error(f"Authentication callback failed: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Authentication failed: {str(e)}"
        )

@app.get("/auth-status")
async def auth_status():
    """
    Check if user is authenticated with Spotify.
    
    Returns:
        JSON response with authentication status
    """
    global spotify_client
    try:
        if spotify_client.sp:
            return {"authenticated": True}
        else:
            return {"authenticated": False}
    except Exception as e:
        logger.error(f"Error checking auth status: {str(e)}")
        return {"authenticated": False, "error": str(e)}
