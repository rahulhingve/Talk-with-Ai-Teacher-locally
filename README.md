﻿# Talk-with-Ai-Teacher-locally

## Installation Guide for Mistral 7B with Ollama

### Windows Installation

1. Download and install Ollama from the [official website](https://ollama.ai/download)
2. Open Command Prompt and run:

```bash
ollama pull mistral
```

### macOS Installation

1. Install Ollama using Homebrew:

```bash
brew install ollama
```

2. Start Ollama service:

```bash
ollama serve
```

3. Pull Mistral 7B model:

```bash
ollama pull mistral
```

The model is approximately 4GB, so ensure you have sufficient storage and a stable internet connection. Once installed, you can run Mistral using:

```bash
ollama run mistral
```

```bash
# Clone this repository
git clone https://github.com/yourusername/Talk-with-Ai-Teacher-locally.git

# Navigate into the repository
cd Talk-with-Ai-Teacher-locally

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install the required packages
pip install -U openai-whisper ollama pyttsx3 pyaudio keyboard numpy

# Run the AI
python ai.py
```
