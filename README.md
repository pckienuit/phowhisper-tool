# What the F is this tool?

This tool is your intelligent assistant for listening to audio recordings, YouTube videos, lectures, and more. It automatically transcribes speech and summarizes the content into clear, structured notes—making it easy to review, study, or share important information from any audio or video source.

## Features
- **Automatic device selection:** Detects and uses GPU if available, otherwise falls back to CPU.
- **Batch audio processing:** Processes all audio files in the `audio` folder automatically.
- **Auto-cleanup:** Automatically deletes audio files that have already been processed.
- **Speed file handling:** If a file with `_speed` in its name exists, only processes the `_speed` file and skips the original.
- **Optimized speed:** For long files, automatically finds and creates an optimal speed version for faster processing (unless a `_speed` file already exists).
- **YouTube and Google Drive support:** Download and transcribe audio directly from YouTube and Google Drive links.
- **Advanced noise reduction:** Apply adaptive noise reduction algorithms to improve transcription quality.
- **Flexible CLI:** Supports both automatic and manual modes, with full CLI argument support for scripting and automation.
- **Manual file management:** In manual mode, allows you to select, skip, or delete files interactively.
- **Seamless Gemini integration:** Processes transcripts with Gemini AI for advanced post-processing and content structuring.
- **GUI interface:** User-friendly graphical interface with Q&A capabilities for lecture content.
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

#### Process YouTube/Google Drive Audio
```bash
python phowhisper.py https://youtube.com/watch?v=xxxx
python phowhisper.py https://drive.google.com/file/d/xxxx/view
```

#### GUI Interface
```bash
python gui.py
```
- User-friendly graphical interface
- Supports drag & drop files
- Integrated Q&A functionality for lecture content

### 3. File Handling Rules
- If a file with `_speed` in its name exists in the `audio` folder, only the `_speed` file is processed; the original is skipped.
- If a file has already been processed (a corresponding `_processed.txt` exists in `output`), it is automatically deleted from `audio`.
- If the file is a `_speed` file, it is processed directly without optimal speed detection.

### 4. CLI Arguments
- `--mode auto|manual` : Select automatic or manual mode (default: auto)
- `--device cpu|cuda` : Specify device in manual mode
- `--noise-reduction` : Apply noise reduction preprocessing
- `--reduction-strength` : Noise reduction strength (0.0-1.0, default: 0.5)
- `--skip-speed` : Skip speed optimization for faster processing
- `--asr-model` : ASR model selection: `auto` (auto-detect language), `phowhisper` (Vietnamese only), `whisper` (multilingual)

#### Examples with noise reduction:
```bash
python phowhisper.py --noise-reduction --reduction-strength 0.7
python phowhisper.py --mode manual --device cuda --noise-reduction
```

#### Examples with ASR model selection:
```bash
# Auto-detect language (default)
python phowhisper.py --asr-model auto

# Force Vietnamese model (PhoWhisper) even for English audio
python phowhisper.py --asr-model phowhisper

# Force multilingual model (Whisper) even for Vietnamese audio
python phowhisper.py --asr-model whisper

# Combine with other options
python phowhisper.py --asr-model whisper --device cuda --noise-reduction
```

### 5. Output Files
For each processed audio file, the following files are generated in the `output` folder:
- `[filename].txt` - Raw transcription
- `[filename]_processed.txt` - Processed and structured content

## Project Structure
- `audio/` - Directory for input audio files
- `output/` - Directory for transcription results
- `phowhisper.py` - Main script
- `gui.py` - Graphical user interface
- `systemprompt` - Gemini AI processing instructions
- `requirements.txt` - Python dependencies

## Contributing & Contact
- Contribute code, report issues: create an issue or pull request on Github
- Contact author: 23520804@gm.uit.edu.vn

## Google Colab Setup

### 1. Open Google Colab
1. Go to [Google Colab](https://colab.research.google.com)
2. Create a new notebook or open an existing one

### 2. Clone the Repository
```python
!git clone [repository-url]
%cd [repository-name]
```

### 3. Install Dependencies
```python
# First, install PyTorch with CUDA support
!pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install huggingface-hub first to resolve dependency conflicts
!pip install huggingface-hub>=0.28.1

# Install transformers and other dependencies
!pip install transformers>=4.41.0
!pip install sentence-transformers>=4.1.0
!pip install accelerate>=1.7.0
!pip install peft>=0.15.2
!pip install gradio>=5.31.0
!pip install diffusers>=0.33.1

# Install remaining requirements
!pip install -r requirements.txt
```

### 4. Install FFmpeg
```python
!apt-get update
!apt-get install -y ffmpeg
```

### 5. Configure Gemini API
1. Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Create an API key
3. Add the API key to your Colab notebook:
```python
import os
os.environ['GEMINI_API_KEY'] = 'your_api_key_here'
```

### 6. Create Required Folders
```python
!mkdir -p audio output
```

### 7. Upload Audio Files
- Use the Colab file browser to upload audio files to the `audio` folder
- Or use the following code to upload files:
```python
from google.colab import files
uploaded = files.upload()
# Move uploaded files to audio folder
!mv *.mp3 *.wav *.mp4 audio/
```

### 8. Run the Tool
```python
!python phowhisper.py
```

### Notes for Colab Users
- Colab provides free GPU access, but sessions are time-limited
- For long audio files, consider using the speed optimization feature
- Save your output files before the session ends
- You can download processed files using:
```python
from google.colab import files
files.download('output/your_file.txt')
```

### Troubleshooting
If you encounter any package compatibility issues:
1. Restart the Colab runtime (Runtime > Restart runtime)
2. Make sure you're using a GPU runtime (Runtime > Change runtime type > Hardware accelerator > GPU)
3. Run the installation commands in order as shown above
4. If issues persist, try clearing the Colab cache and reinstalling:
```python
!pip cache purge
!pip uninstall -y torch torchvision torchaudio transformers huggingface-hub sentence-transformers accelerate peft gradio diffusers
```
Then reinstall the packages using the commands in step 3.

5. If you still encounter issues, try installing packages one by one and check for errors:
```python
!pip install huggingface-hub>=0.28.1
!pip install transformers>=4.41.0
!pip install sentence-transformers>=4.1.0
!pip install accelerate>=1.7.0
!pip install peft>=0.15.2
!pip install gradio>=5.31.0
!pip install diffusers>=0.33.1
```
