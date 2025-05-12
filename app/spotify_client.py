"""
Spotify Client Module

This module provides a client for interacting with the Spotify API
to search for tracks and create playlists.
"""

import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
from typing import List, Optional, Dict, Any, Union
import json
from fuzzywuzzy import fuzz
import logging
import asyncio
from .utils import COMMON_WORDS

logger = logging.getLogger("text-to-spotify.spotify_client")

class SpotifyClient:
    """
    Client for interacting with the Spotify API.
    
    This class provides methods for authenticating with Spotify,
    searching for tracks, and creating playlists.
    """
    
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        """
        Initialize the Spotify client.
        
        Args:
            client_id: Spotify API client ID
            client_secret: Spotify API client secret
            redirect_uri: Redirect URI for OAuth flow
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.sp = None
        self.token_info = None
        logger.info("SpotifyClient initialized")

    def get_auth_url(self) -> str:
        """
        Get the Spotify authorization URL.
        
        Returns:
            Spotify authorization URL for user login
        """
        auth_manager = SpotifyOAuth(
            client_id=self.client_id,
            client_secret=self.client_secret,
            redirect_uri=self.redirect_uri,
            scope="playlist-modify-public playlist-modify-private",
            cache_handler=None
        )
        logger.debug("Generated Spotify authorization URL")
        return auth_manager.get_authorize_url()

    async def handle_callback(self, code: str) -> None:
        """
        Handle the OAuth callback and initialize the Spotify client.
        
        Args:
            code: Authorization code from Spotify
            
        Raises:
            Exception: If authentication fails
        """
        try:
            auth_manager = SpotifyOAuth(
                client_id=self.client_id,
                client_secret=self.client_secret,
                redirect_uri=self.redirect_uri,
                scope="playlist-modify-public playlist-modify-private",
                cache_handler=None
            )
            
            # Get the token
            self.token_info = auth_manager.get_access_token(code)
            
            # Initialize Spotify client with the token
            self.sp = spotipy.Spotify(auth=self.token_info['access_token'])
            logger.info("Spotify authentication successful")
        except Exception as e:
            logger.error(f"Spotify authentication failed: {str(e)}")
            raise Exception(f"Spotify authentication failed: {str(e)}")

    async def search_track(self, title: str) -> Optional[Dict[str, Any]]:
        """
        Search for a track on Spotify with fuzzy matching.
        
        Args:
            title: Track title to search for
            
        Returns:
            Dictionary with track information or None if no match found
            
        Raises:
            Exception: If Spotify client is not initialized
        """
        if not self.sp:
            logger.error("Spotify client not initialized")
            raise Exception("Spotify client not initialized. Please authenticate first.")
        
        try:
            # Split the title into words
            words = title.split()
            best_match = None
            best_score = 0

            # Search for each word and combinations
            for word in words:
                # Search for tracks with the word as the title
                results = self.sp.search(q=f'track:"{word}"', type='track', limit=10)
                for track in results['tracks']['items']:
                    # Check for exact match first
                    if track['name'].strip().lower() == word.strip().lower():
                        logger.debug(f"Found exact match for '{word}': {track['name']}")
                        return {
                            'id': track['id'],
                            'name': track['name'],
                            'artist': track['artists'][0]['name'],
                            'score': 100
                        }
                    # Fuzzy match if exact match is not found
                    score = fuzz.partial_ratio(track['name'].strip().lower(), word.strip().lower())
                    if score > best_score:
                        best_score = score
                        best_match = {
                            'id': track['id'],
                            'name': track['name'],
                            'artist': track['artists'][0]['name'],
                            'score': score
                        }

            # If no exact match found, return the best fuzzy match
            if best_match and best_score > 70:  # Adjust threshold as needed
                logger.debug(f"Best fuzzy match for '{title}': {best_match['name']} with score {best_score}")
                return best_match

            logger.debug(f"No good match found for '{title}'")
            return None
        except Exception as e:
            logger.error(f"Error searching for track '{title}': {str(e)}")
            return None

    async def create_playlist(self, tracks: List[Dict[str, Any]], name: str) -> str:
        """
        Create a new playlist and add tracks to it.
        
        Args:
            tracks: List of track dictionaries (with id, name, artist)
            name: Name for the new playlist
            
        Returns:
            URL of the created playlist
            
        Raises:
            Exception: If Spotify client is not initialized or playlist creation fails
        """
        if not self.sp:
            logger.error("Spotify client not initialized")
            raise Exception("Spotify client not initialized. Please authenticate first.")
        
        try:
            logger.info(f"Creating playlist with name: 'Text to Playlist: {name}'")
            # Get current user
            user = self.sp.current_user()
            logger.debug(f"Current Spotify user: {user['id']}")
            
            # Create playlist
            playlist = self.sp.user_playlist_create(
                user=user['id'],
                name=f"Text to Playlist: {name}",
                public=True,
                description=f"Generated from text: {name}"
            )
            logger.info(f"Created playlist with ID: {playlist['id']}")
            
            # Add tracks to playlist
            track_uris = [f"spotify:track:{track['id']}" for track in tracks]
            logger.info(f"Adding {len(track_uris)} tracks to playlist")
            
            # Add tracks in batches of 100 (Spotify API limit)
            for i in range(0, len(track_uris), 100):
                batch = track_uris[i:i+100]
                self.sp.playlist_add_items(playlist['id'], batch)
            
            logger.info(f"Playlist created successfully: {playlist['external_urls']['spotify']}")
            return playlist['external_urls']['spotify']
        except Exception as e:
            logger.error(f"Error creating playlist: {str(e)}")
            raise Exception(f"Failed to create playlist: {str(e)}")

    async def search_multiple_tracks(self, word: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """
        Return up to max_results song matches for a word, using enhanced matching strategies.
        
        Args:
            word: Word or phrase to search for
            max_results: Maximum number of results to return
            
        Returns:
            List of track dictionaries with match information
            
        Raises:
            Exception: If Spotify client is not initialized
        """
        if not self.sp:
            logger.error("Spotify client not initialized")
            raise Exception("Spotify client not initialized. Please authenticate first.")
        
        # Skip very common words unless they're part of a longer phrase
        word = word.strip()
        if word.lower() in COMMON_WORDS and len(word.split()) == 1:
            logger.debug(f"Skipping common word: '{word}'")
            return []
            
        try:
            logger.debug(f"Spotify search for: '{word}'")
            matches = []
            
            # Clean the search term while preserving apostrophes
            search_term = word.lower()
            # Normalize search term to handle cases like "don't" and "dont"
            normalized_term = search_term.replace("'", "")
            
            # Only do one search with quotes to minimize API calls
            logger.debug(f"Performing quoted search")
            
            # First try with exact phrase including apostrophes
            results = self.sp.search(q=f'track:"{search_term}"', type='track', limit=30)
            logger.debug(f"Found {len(results['tracks']['items'])} tracks in search with apostrophes")
            
            # If we get very few results and the term has apostrophes, try without them
            if len(results['tracks']['items']) < 5 and "'" in search_term:
                normalized_results = self.sp.search(q=f'track:"{normalized_term}"', type='track', limit=30)
                logger.debug(f"Found {len(normalized_results['tracks']['items'])} tracks in normalized search")
                
                # Combine results
                for track in normalized_results['tracks']['items']:
                    results['tracks']['items'].append(track)
            
            # Process all results in one go
            for track in results['tracks']['items']:
                track_name = track['name'].strip().lower()
                
                # Normalize track name for matching
                track_name_normalized = track_name.replace("'", "")
                
                # Calculate scores with both original and normalized versions
                exact_score = 100 if track_name == search_term or track_name_normalized == normalized_term else 0
                contains_score = 90 if search_term in track_name or normalized_term in track_name_normalized else 0
                
                # Fuzzy matching both with and without apostrophes
                partial_score_original = fuzz.partial_ratio(track_name, search_term)
                partial_score_normalized = fuzz.partial_ratio(track_name_normalized, normalized_term)
                partial_score = max(partial_score_original, partial_score_normalized)
                
                token_sort_score_original = fuzz.token_sort_ratio(track_name, search_term)
                token_sort_score_normalized = fuzz.token_sort_ratio(track_name_normalized, normalized_term)
                token_sort_score = max(token_sort_score_original, token_sort_score_normalized)
                
                token_set_score_original = fuzz.token_set_ratio(track_name, search_term)
                token_set_score_normalized = fuzz.token_set_ratio(track_name_normalized, normalized_term)
                token_set_score = max(token_set_score_original, token_set_score_normalized)
                
                # Combine scores
                combined_score = max(
                    exact_score,
                    contains_score,
                    partial_score * 0.7,
                    token_sort_score * 0.6,
                    token_set_score * 0.5
                )
                
                # Only add high-scoring matches
                if combined_score > 70:
                    if exact_score == 100:
                        logger.debug(f"Perfect match found: '{track['name']}' by {track['artists'][0]['name']}")
                    elif contains_score > 0:
                        logger.debug(f"Contains match found: '{track['name']}' contains '{word}'")
                    else:
                        logger.debug(f"Fuzzy match: '{track['name']}' score={combined_score:.1f}")
                        
                    matches.append({
                        'id': track['id'],
                        'name': track['name'],
                        'artist': track['artists'][0]['name'],
                        'score': combined_score,
                        'popularity': track.get('popularity', 0)
                    })
            
            # Sort by score and popularity
            matches.sort(key=lambda x: (-x['score'], -x['popularity']))
            
            # Remove duplicates while preserving order
            unique_matches = []
            seen_ids = set()
            for match in matches:
                if match['id'] not in seen_ids:
                    unique_matches.append(match)
                    seen_ids.add(match['id'])
            
            logger.debug(f"Final matches for '{word}': {len(unique_matches)} tracks")
            return unique_matches[:max_results]
        except Exception as e:
            logger.error(f"Error searching for track '{word}': {str(e)}")
            return []
