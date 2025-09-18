# PhoWhisper Tool

Công cụ thông minh để chuyển đổi âm thanh thành văn bản với khả năng xử lý tiếng ồn nền và tối ưu hóa hiệu suất. Hỗ trợ transcription audio từ file, YouTube, và nhiều nguồn khác với AI model Whisper.

## ✨ Tính năng chính

### 🎯 Core Features
- **Auto transcription**: Tự động chuyển đổi speech thành text với độ chính xác cao
- **Noise reduction**: Xử lý và giảm tiếng ồn nền trước khi transcribe
- **Smart chunking**: Tự động chia audio thành các đoạn tối ưu (≤ 30s) để xử lý hiệu quả
- **GPU optimization**: Tự động phát hiện và sử dụng GPU (CUDA) nếu có sẵn
- **Batch processing**: Xử lý hàng loạt tất cả file trong thư mục `audio/`

### 🔧 Advanced Features  
- **Speed optimization**: Option để bỏ qua speed processing cho xử lý nhanh
- **Adaptive processing**: Phân tích noise level và tự động điều chỉnh strategy
- **YouTube support**: Download và transcribe trực tiếp từ YouTube links
- **Auto cleanup**: Tự động xóa file đã xử lý
- **Output management**: Lưu kết quả vào thư mục `output/` với format rõ ràng

### 🎛️ CLI Options
- `--noise-reduction`: Bật chức năng giảm tiếng ồn
- `--reduction-strength`: Điều chỉnh cường độ giảm noise (0.5-2.0)
- `--skip-speed`: Bỏ qua speed optimization để xử lý nhanh hơn
- `--auto`: Chế độ tự động không cần interaction
- `--manual`: Chế độ thủ công cho phép chọn file

## 🚀 Cài đặt

### 1. Yêu cầu hệ thống
- Python 3.8+
- FFmpeg
- NVIDIA GPU với CUDA support (khuyến nghị)
- 4GB+ RAM cho việc xử lý audio dài

### 2. Cài đặt PyTorch

#### Windows (với CUDA)
```bash
# Cho GPU NVIDIA
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Cho CPU only
pip3 install torch torchvision torchaudio
```

#### macOS
```bash
# Cho CPU only
pip3 install torch torchvision torchaudio

# Cho Apple Silicon (M1/M2)
pip3 install torch torchvision torchaudio
```

#### Linux
```bash
# Cho CUDA
pip3 install torch torchvision torchaudio

# Cho CPU only  
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 4. Cài đặt FFmpeg

#### Windows
1. Download từ [FFmpeg official site](https://ffmpeg.org/download.html)
2. Extract và thêm vào PATH
3. Hoặc sử dụng Chocolatey: `choco install ffmpeg`

#### macOS
```bash
# Sử dụng Homebrew
brew install ffmpeg
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install ffmpeg
```

## 📖 Hướng dẫn sử dụng

### Cách sử dụng cơ bản

1. **Chuẩn bị file audio**: Đặt file audio vào thư mục `audio/`
2. **Chạy tool**: 
   ```bash
   python phowhisper.py
   ```
3. **Kết quả**: Xem file transcription trong thư mục `output/`

### Các chế độ nâng cao

#### Với noise reduction
```bash
# Bật giảm tiếng ồn với cường độ mặc định
python phowhisper.py --noise-reduction

# Điều chỉnh cường độ giảm noise
python phowhisper.py --noise-reduction --reduction-strength 1.5
```

#### Chế độ tốc độ cao
```bash
# Bỏ qua speed optimization
python phowhisper.py --skip-speed

# Kết hợp với noise reduction
python phowhisper.py --noise-reduction --skip-speed
```

#### Chế độ tự động
```bash
# Xử lý tất cả file mà không cần interaction
python phowhisper.py --auto

# Tự động với noise reduction
python phowhisper.py --auto --noise-reduction
```

### YouTube transcription
```bash
# Tool sẽ tự động phát hiện YouTube URLs trong input
python phowhisper.py
# Nhập YouTube URL khi được yêu cầu
```

## 📁 Cấu trúc thư mục

```
phowhisper-tool/
├── audio/              # Đặt file audio input vào đây
├── output/             # Kết quả transcription được lưu ở đây
├── phowhisper.py       # Script chính
├── gui.py              # Giao diện GUI (optional)
├── requirements.txt    # Dependencies
├── .env               # Config file (optional)
└── README.md          # Tài liệu này
```

## ⚙️ Cấu hình nâng cao

### Noise Reduction Settings
- **Default strength**: 1.0 (balanced)
- **Light noise**: 0.5-0.8 
- **Heavy noise**: 1.5-2.0
- **Algorithm**: Spectral subtraction + High-pass filtering

### Chunk Processing
- **Maximum chunk size**: 30 seconds
- **Minimum chunk size**: 2 seconds  
- **Smart chunking**: Phân tích noise level để tối ưu
- **Fallback**: Fixed 30s chunks khi algorithm thất bại

### Performance Tips
- Sử dụng GPU để tăng tốc độ xử lý đáng kể
- Với file > 10 phút: khuyến nghị dùng `--noise-reduction`
- Với audio chất lượng thấp: dùng `--reduction-strength 1.5`
- Để xử lý nhanh: dùng `--skip-speed`

## 🎛️ Environment Variables

Tạo file `.env` để cấu hình:
```env
# GPU settings
CUDA_VISIBLE_DEVICES=0

# Model settings  
WHISPER_MODEL=base
DEVICE=auto

# Processing settings
MAX_CHUNK_SIZE=30
NOISE_REDUCTION_DEFAULT=1.0
```

## 🔧 Troubleshooting

### Lỗi thường gặp

#### "CUDA out of memory"
```bash
# Giảm batch size hoặc sử dụng CPU
python phowhisper.py --skip-speed
```

#### "FFmpeg not found"
- Đảm bảo FFmpeg đã được cài đặt và trong PATH
- Windows: Thêm FFmpeg bin folder vào System PATH

#### Audio quality thấp
```bash
# Tăng cường độ noise reduction
python phowhisper.py --noise-reduction --reduction-strength 2.0
```

#### Xử lý chậm
```bash
# Sử dụng skip speed mode
python phowhisper.py --skip-speed
```

### Performance Benchmarks
- **GPU (RTX 3080)**: ~10x nhanh hơn CPU
- **CPU (Intel i7)**: 1 phút audio ≈ 2-3 phút xử lý
- **Memory usage**: ~2-4GB cho file audio 1 giờ

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

## 📜 License

Project này sử dụng MIT License. Xem file `LICENSE` để biết thêm chi tiết.

## 🙏 Credits

- **Whisper**: OpenAI's speech recognition model
- **PyTorch**: Deep learning framework
- **FFmpeg**: Audio/video processing
- **Scipy**: Signal processing for noise reduction

## 📞 Support

Nếu gặp vấn đề hoặc có câu hỏi:
1. Kiểm tra [Troubleshooting](#🔧-troubleshooting) section
2. Tạo issue trên GitHub
3. Email: [your-email@example.com]

---

**Made with ❤️ for Vietnamese transcription needs**
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
