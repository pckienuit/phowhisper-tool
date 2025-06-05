# Audio Transcription Tool

A powerful tool for transcribing and processing audio lectures using PhoWhisper and Gemini AI.

## Features

- Transcribe audio files using PhoWhisper (Vietnamese speech recognition)
- Process transcripts with Gemini AI to create structured lecture notes
- Support for various audio formats (WAV, MP3, etc.)
- YouTube video audio extraction
- Automatic audio speed optimization for long recordings
- Parallel processing for multiple files
- GUI interface for easy use

## Requirements

- Python 3.8 or higher
- FFmpeg installed on your system
- NVIDIA GPU with CUDA support (recommended)

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install required Python packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

## Usage

### Command Line Interface

1. Place your audio files in the `audio` folder
2. Run the script:
```bash
python phowhisper.py
```

For YouTube videos:
```bash
python phowhisper.py "youtube-url"
```

### GUI Interface

Run the GUI version:
```bash
python gui.py
```

## Project Structure

- `audio/` - Directory for input audio files
- `output/` - Directory for transcription results
- `phowhisper.py` - Main script
- `gui.py` - GUI interface
- `systemprompt` - Gemini AI processing instructions
- `requirements.txt` - Python dependencies

## Output Files

For each processed audio file, the following files are generated in the `output` folder:
- `[filename].txt` - Raw transcription
- `[filename]_processed.txt` - Processed and structured content

## License


## Contributing
