# What the F is this tool?

This tool is your intelligent assistant for listening to audio recordings, YouTube videos, lectures, and more. It automatically transcribes speech and summarizes the content into clear, structured notes—making it easy to review, study, or share important information from any audio or video source.

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

## Requirements
- Python 3.8 or higher
- FFmpeg installed on your system
- NVIDIA GPU with CUDA support (recommended)
- PyTorch

## Installation Guide

### 1. Install Python Dependencies

#### Installing PyTorch
##### Windows
1. Visit [PyTorch's official website](https://pytorch.org/get-started/locally/)
2. Select your preferences:
   - PyTorch Build: Stable
   - Your OS: Windows
   - Package: Pip
   - Language: Python
   - Compute Platform: CUDA (if you have NVIDIA GPU) or CPU
3. Copy and run the generated command. For example:
   ```bash
   # For CUDA (GPU) support:
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

   # For CPU only:
   pip3 install torch torchvision torchaudio
   ```

##### macOS
```bash
# For CPU only:
pip3 install torch torchvision torchaudio

# For MPS (Apple Silicon) support:
pip3 install torch torchvision torchaudio
```

##### Linux
```bash
# For CUDA (GPU) support:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
pip3 install torch torchvision torchaudio
```

#### Installing FFmpeg
##### Windows
1. Download FFmpeg from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html) (choose Windows build).
2. Extract the downloaded zip file.
3. Add the `bin` folder (inside the extracted FFmpeg folder) to your system PATH environment variable.
4. Open a new Command Prompt and run:
   ```bash
   ffmpeg -version
   ```
   You should see FFmpeg version info if installed correctly.

##### macOS (with Homebrew)
```bash
brew install ffmpeg
```

##### Ubuntu/Debian Linux
```bash
sudo apt update
sudo apt install ffmpeg
```

##### Other Linux
Refer to your distribution's package manager or see [FFmpeg official instructions](https://ffmpeg.org/download.html).

### 2. Install the Tool
1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```
2. Install required Python packages:
```bash
pip install -r requirements.txt
```

### 3. Configure Gemini API
1. Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) and sign in with your Google account.
2. Click "Create API key" and follow the instructions.
3. Copy the generated API key.
4. Create a file named `.env` in the project root (if it doesn't exist).
5. Add the following line to your `.env` file:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```
6. Save the file. You're ready to use Gemini features!

## Usage Guide

### 1. Prepare Required Folders
Before running the script, make sure you have the following folders in your project directory:
```bash
mkdir -p audio output
```
- `audio/` : Place your audio (wav/mp4) files here for processing
- `output/` : Transcripts and Gemini-processed results are saved here

### 2. Running the Tool

#### Quick Start (Auto Mode, Recommended)
```bash
python phowhisper.py
```
Or:
```bash
python phowhisper.py --mode auto
```
- Automatically selects GPU/CPU, processes and cleans up files.
- Suitable for batch, cronjob, or bash script usage.

#### Manual Mode
```bash
python phowhisper.py --mode manual
```
- Lets you choose device and manage files interactively.

Or specify device directly:
```bash
python phowhisper.py --mode manual --device cpu
python phowhisper.py --mode manual --device cuda
```

#### Process YouTube Audio
```bash
python phowhisper.py https://youtube.com/watch?v=xxxx
```

### 3. File Handling Rules
- If a file with `_speed` in its name exists in the `audio` folder, only the `_speed` file is processed; the original is skipped.
- If a file has already been processed (a corresponding `_processed.txt` exists in `output`), it is automatically deleted from `audio`.
- If the file is a `_speed` file, it is processed directly without optimal speed detection.

### 4. CLI Arguments
- `--mode auto|manual` : Select automatic or manual mode (default: auto)
- `--device cpu|cuda` : Specify device in manual mode

### 5. Output Files
For each processed audio file, the following files are generated in the `output` folder:
- `[filename].txt` - Raw transcription
- `[filename]_processed.txt` - Processed and structured content

## Project Structure
- `audio/` - Directory for input audio files
- `output/` - Directory for transcription results
- `phowhisper.py` - Main script
- `systemprompt` - Gemini AI processing instructions
- `requirements.txt` - Python dependencies

## Contributing & Contact
- Contribute code, report issues: create an issue or pull request on Github
- Contact author: 23520804@gm.uit.edu.vn
