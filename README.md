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

# PhoWhisper

PhoWhisper is an automatic speech recognition and transcript processing tool, optimized for both GPU and CPU, suitable for batch processing and bash script integration.

## Features
- **Automatic device selection:** Detects and uses GPU if available, otherwise falls back to CPU.
- **Batch audio processing:** Processes all audio files in the `audio` folder automatically.
- **Auto-cleanup:** Automatically deletes audio files that have already been processed.
- **Speed file handling:** If a file with `_speed` in its name exists, only processes the `_speed` file and skips the original.
- **Optimized speed:** For long files, automatically finds and creates an optimal speed version for faster processing (unless a `_speed` file already exists).
- **YouTube audio support:** Download and transcribe audio directly from YouTube links.
- **Flexible CLI:** Supports both automatic and manual modes, with full CLI argument support for scripting and automation.
- **Manual file management:** In manual mode, allows you to select, skip, or delete files interactively.
- **Seamless Gemini integration:** Processes transcripts with Gemini for advanced post-processing.
- **Output management:** Saves both raw and processed transcripts to the `output` folder.

## Usage

### 0. Prepare Required Folders
Before running the script, make sure you have the following folders in your project directory:
```bash
mkdir -p audio output
```
- `audio/` : Place your audio (wav/mp4) files here for processing
- `output/` : Transcripts and Gemini-processed results are saved here

### 1. Quick Start (Auto Mode, Recommended)
```bash
python phowhisper.py
```
Or:
```bash
python phowhisper.py --mode auto
```
- Automatically selects GPU/CPU, processes and cleans up files.
- Suitable for batch, cronjob, or bash script usage.

### 2. Manual Mode
```bash
python phowhisper.py --mode manual
```
- Lets you choose device and manage files interactively.

Or specify device directly:
```bash
python phowhisper.py --mode manual --device cpu
python phowhisper.py --mode manual --device cuda
```

### 3. Process YouTube Audio
```bash
python phowhisper.py https://youtube.com/watch?v=xxxx
```

## File Handling Rules
- If a file with `_speed` in its name exists in the `audio` folder, only the `_speed` file is processed; the original is skipped.
- If a file has already been processed (a corresponding `_processed.txt` exists in `output`), it is automatically deleted from `audio`.
- If the file is a `_speed` file, it is processed directly without optimal speed detection.

## CLI Arguments
- `--mode auto|manual` : Select automatic or manual mode (default: auto)
- `--device cpu|cuda` : Specify device in manual mode

## Folders
- `audio/` : Place your audio (wav/mp4) files here for processing
- `output/` : Transcripts and Gemini-processed results are saved here

## Example Bash Script
```bash
# Process all audio files automatically, no manual input
python phowhisper.py

# Manual mode, choose CPU
python phowhisper.py --mode manual --device cpu
```

## Modes
- **Auto mode:**
  - Detects and uses the best available device (GPU/CPU)
  - Processes all files automatically
  - Cleans up processed files
  - Skips original files if a `_speed` version exists
- **Manual mode:**
  - Lets you choose device and file operations interactively
  - Useful for step-by-step control or debugging

## Contributing & Contact
- Contribute code, report issues: create an issue or pull request on Github
- Contact author: [your-email@example.com]
