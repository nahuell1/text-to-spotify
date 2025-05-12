# Text to Spotify Playlist Generator

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

An interactive web application that converts text into a Spotify playlist by finding songs whose titles match words and phrases from your input text.

> Demo (without AI paraphrasing) https://text-to-spotify.onrender.com/

## 🎵 Features

- **Text Conversion**: Turn any text into a Spotify playlist
- **Smart Matching**: Uses advanced algorithms to find song matches for phrases, not just individual words
- **Paraphrasing**: Generates alternative versions of your text to improve song matching
- **Interactive UI**: Color-coded visualization shows which parts of your text are covered
- **Multiple Match Options**: Provides multiple song options for each part of your text
- **Selective Playlist Creation**: Choose which songs to include in your final playlist
- **API Integration**: Seamless Spotify authentication and playlist creation


## 💻 Installation

### Prerequisites

- Python 3.8+
- Ollama or OpenAI API key for paraphrasing

### Option 1: Local Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/nahuell1/text-to-spotify.git
   cd text-to-spotify
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm  # Optional, for improved text processing
   ```

4. Create a Spotify Developer application:
   - Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
   - Create a new application
   - Set the redirect URI to `http://localhost:8000/callback`
   - Note your Client ID and Client Secret

5. Create a `.env` file:
   ```
   # Spotify API credentials (required)
   SPOTIPY_CLIENT_ID=your_spotify_client_id
   SPOTIPY_CLIENT_SECRET=your_spotify_client_secret
   SPOTIPY_REDIRECT_URI=http://localhost:8000/callback

   # Paraphrasing options (optional)
   # Options: "none", "ollama", "openai"
   PARAPHRASE_PROVIDER=none

   # Ollama configuration (required if PARAPHRASE_PROVIDER=ollama)
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama3

   # OpenAI configuration (required if PARAPHRASE_PROVIDER=openai)
   OPENAI_API_KEY=your_openai_api_key
   OPENAI_MODEL=gpt-3.5-turbo

   # Logging configuration
   # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
   LOG_LEVEL=INFO
   LOG_DIR=logs

   # Debug mode (enables additional logging and features)
   # Options: true, false
   DEBUG_MODE=false
   ```

6. Run the application:
   ```bash
   uvicorn app.main:app --reload
   ```

7. Open your browser and go to `http://localhost:8000`

### Option 2: Docker Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/nahuell1/text-to-spotify.git
   cd text-to-spotify
   ```

2. Create a `.env` file as described above

3. Build and run with Docker Compose:
   ```bash
   docker-compose up --build
   ```

4. Open your browser and go to `http://localhost:8000`

## 🎮 Usage

1. Enter your text in the input field
2. Click "Generate Playlist Candidates"
3. Review the playlist candidates:
   - The original text is always shown first
   - Each candidate shows coverage percentage
   - Words are color-coded based on coverage (red = not covered, yellow = partially, green = covered)
   - Multiple song options are provided for each phrase
4. Select the songs you want to include:
   - Use the checkboxes to select individual songs
   - Use "Select all" to toggle all songs for a candidate
   - See how your selections affect text coverage in real-time
5. Click "Create Playlist from Selected"
6. Authenticate with Spotify if prompted
7. Your playlist will be created in your Spotify account

## 🔄 Advanced Features

### Paraphrasing Providers

The application can use different providers to generate paraphrases of your text to improve song matching:

- **None**: Uses only the original text
- **Ollama**: Uses a local Ollama instance with models like llama3
- **OpenAI**: Uses OpenAI's API for high-quality paraphrases (requires API key)

### Debugging and Logging

- Comprehensive logging system with configurable log levels
- Debug mode for detailed diagnostics
- Health check endpoint at `/health` for monitoring

## 🏗️ Project Structure

```
text-to-spotify/
├── app/
│   ├── main.py             # Main application logic
│   ├── spotify_client.py   # Spotify API integration
│   ├── text_processor.py   # Text processing utilities
│   ├── utils.py            # Utility functions and helpers
│   └── templates/
│       └── index.html      # Web interface
├── logs/                   # Log files directory
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose configuration
├── .dockerignore           # Docker ignore file
├── .env                    # Environment variables (create from .env.example)
└── README.md               # This file
```

## 🛣️ Roadmap

- [ ] Add user accounts for saving favorite playlists
- [ ] Improve matching algorithm with machine learning
- [ ] Support for multiple languages
- [ ] Mobile application

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Spotify Web API](https://developer.spotify.com/documentation/web-api/) for providing the API
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Spotipy](https://spotipy.readthedocs.io/) for the Spotify API wrapper
- [Tailwind CSS](https://tailwindcss.com/) for styling
