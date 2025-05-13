from transformers import pipeline
import os
from pydub import AudioSegment
from pydub.utils import make_chunks
import torch
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
from typing import List, Dict
import numpy as np
from tqdm import tqdm
import logging
import warnings
from pytube import YouTube
import re
import yt_dlp
import google.generativeai as genai
import time
from multiprocessing import cpu_count
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Suppress all warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('numpy').setLevel(logging.ERROR)
logging.getLogger('pydub').setLevel(logging.ERROR)

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Load API key from environment variable
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in .env file")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Initialize model with optimized settings
transcriber = pipeline(
    "automatic-speech-recognition",
    model="vinai/PhoWhisper-medium",
    device="cuda",
    return_timestamps=True,
    framework="pt",
    torch_dtype=torch.float16,  # Use half precision for faster processing
    model_kwargs={"use_cache": True}  # Enable model caching
)

def split_audio(audio_path: str, chunk_length_ms: int = 30000) -> List[str]:
    """
    Split audio into optimized chunks at silent or very quiet points.
    Returns paths to temporary files containing the chunks.
    """
    audio = AudioSegment.from_file(audio_path)
    temp_files = []
    
    print("\nSplitting audio into chunks...")
    
    # Calculate frame length for analysis (20ms frames for faster processing)
    frame_length = 20
    frames = make_chunks(audio, frame_length)
    
    # Find optimal split points
    split_points = []
    current_chunk_start = 0
    min_silence_threshold = -50  # Increased threshold for better chunking
    min_silence_duration = 300   # Reduced minimum silence duration for more natural splits
    
    # Process frames in batches for better performance
    batch_size = 50
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        batch_dBFS = [frame.dBFS for frame in batch]
        
        for j, dBFS in enumerate(batch_dBFS):
            frame_idx = i + j
            if dBFS <= min_silence_threshold:
                if frame_idx - current_chunk_start >= chunk_length_ms / frame_length:
                    split_points.append(frame_idx * frame_length)
                    current_chunk_start = frame_idx
    
    # Add the end of the audio as the final split point
    split_points.append(len(audio))
    
    # Create chunks based on split points
    for i in range(len(split_points) - 1):
        start = split_points[i]
        end = split_points[i + 1]
        chunk = audio[start:end]
        
        if chunk.dBFS > min_silence_threshold:  # Only save chunks with audible sound
            temp_file = f"temp_chunk_{i}.wav"
            chunk.export(temp_file, format="wav", parameters=["-ac", "1", "-ar", "16000"])
            temp_files.append(temp_file)
    
    return temp_files

def process_chunk(chunk_path: str) -> str:
    """Process a single audio chunk with optimized settings."""
    try:
        # Use cached transcriber instance
        if not hasattr(process_chunk, 'transcriber'):
            process_chunk.transcriber = pipeline(
                "automatic-speech-recognition",
                model="vinai/PhoWhisper-medium",
                device="cuda" if torch.cuda.is_available() else "cpu",
                return_timestamps=True,
                framework="pt",
                torch_dtype=torch.float16,
                model_kwargs={
                    "use_cache": True,
                    "low_cpu_mem_usage": True
                }
            )
        
        result = process_chunk.transcriber(chunk_path)
        return result['text']
    except Exception as e:
        print(f"\nError processing chunk {chunk_path}: {str(e)}")
        return ""

def transcribe_audio(audio_path: str, max_workers: int = None) -> str:
    """Transcribe audio using optimized parallel processing."""
    if not os.path.exists(audio_path):
        return "File not found"
    
    audio = AudioSegment.from_file(audio_path)
    if len(audio) > 30000:  # If audio is longer than 30 seconds
        temp_files = split_audio(audio_path)
        transcriptions = []
        
        print("\nTranscribing audio chunks...")
        
        # Determine optimal number of workers
        if max_workers is None:
            if torch.cuda.is_available():
                max_workers = torch.cuda.device_count()
            else:
                max_workers = min(cpu_count(), 4)  # Limit CPU workers to prevent overload
        
        # Process chunks in parallel with optimized settings
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_chunk, temp_file): temp_file 
                      for temp_file in temp_files}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
                chunk_path = futures[future]
                try:
                    text = future.result()
                    transcriptions.append(text)
                except Exception as e:
                    print(f"\nError processing {chunk_path}: {str(e)}")
                finally:
                    # Clean up temporary file
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)
        
        # Clear CUDA cache after processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return " ".join(transcriptions).strip()
    else:
        print("\nTranscribing audio...")
        result = process_chunk(audio_path)
        return result

def extract_audio_from_video(video_path: str, output_folder: str = "audio") -> str:
    """Extract audio from video file using ffmpeg directly."""
    try:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_folder, f"{base_name}.wav")
        
        print(f"\nExtracting audio from {os.path.basename(video_path)}...")
        # Use ffmpeg directly to extract audio
        command = [
            'ffmpeg', '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite output file if exists
            output_path
        ]
        
        subprocess.run(command, check=True, capture_output=True)
        print(f"✓ Audio extracted successfully")
        return output_path
    except Exception as e:
        print(f"\nError extracting audio from {video_path}: {str(e)}")
        return None

def convert_to_wav(input_folder: str) -> None:
    """Convert files to WAV format with optimized settings."""
    files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    if not files:
        print("\nNo files found in the audio folder.")
        return

    print("\nConverting files to WAV format...")
    for file_name in tqdm(files, desc="Converting files"):
        file_path = os.path.join(input_folder, file_name)
        if file_name.endswith(".mp4"):
            extract_audio_from_video(file_path)
        elif not file_name.endswith(".wav"):
            audio = AudioSegment.from_file(file_path)
            wav_file_path = os.path.splitext(file_path)[0] + ".wav"
            audio.export(
                wav_file_path,
                format="wav",
                parameters=["-ac", "1", "-ar", "16000"]
            )

def cleanup_audio_folder(folder_path: str) -> None:
    """Delete all files in the specified audio folder after processing."""
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    if not files:
        return

    print("\nCleaning up audio files...")
    for file_name in tqdm(files, desc="Deleting files"):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

def download_youtube_audio(url: str, output_folder: str = "audio") -> str:
    """
    Download audio from a YouTube URL and convert it to WAV format.
    Returns the path to the downloaded audio file.
    """
    try:
        print(f"\nDownloading audio from YouTube: {url}")
        # Create YouTube object
        yt = YouTube(url)
        
        # Get the video title and clean it for filename
        video_title = re.sub(r'[^\w\s-]', '', yt.title)
        video_title = re.sub(r'[-\s]+', '-', video_title).strip('-_')
        
        # Get the audio stream
        audio_stream = yt.streams.filter(only_audio=True).first()
        
        if not audio_stream:
            print("No audio stream found in the video.")
            return None
            
        # Download the audio
        print("Downloading audio stream...")
        audio_file = audio_stream.download(
            output_path=output_folder,
            filename=f"{video_title}.mp4"
        )
        
        # Convert to WAV format
        print("Converting to WAV format...")
        wav_path = os.path.join(output_folder, f"{video_title}.wav")
        command = [
            'ffmpeg', '-i', audio_file,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite output file if exists
            wav_path
        ]
        
        subprocess.run(command, check=True, capture_output=True)
        
        # Remove the original MP4 file
        if os.path.exists(audio_file):
            os.remove(audio_file)
            
        print(f"✓ Audio downloaded and converted successfully: {wav_path}")
        return wav_path
        
    except Exception as e:
        print(f"\nError downloading YouTube audio: {str(e)}")
        return None

def download_youtube_audio_ytdlp(url: str, output_folder: str = "audio") -> str:
    """
    Download audio from a YouTube URL using yt-dlp and convert it to WAV format.
    Returns the path to the downloaded audio file.
    """
    try:
        print(f"\nDownloading audio from YouTube (yt-dlp): {url}")
        os.makedirs(output_folder, exist_ok=True)
        
        # Clean up any existing files in the output folder
        for file in os.listdir(output_folder):
            if file.endswith('.wav') or file.endswith('.mp3'):
                os.remove(os.path.join(output_folder, file))
        
        # Use yt-dlp to download best audio
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(output_folder, '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'noplaylist': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get('title', 'audio')
            # Clean the title to make it safe for filenames
            title = re.sub(r'[^\w\s-]', '', title)
            title = re.sub(r'[-\s]+', '-', title).strip('-_')
            
            # Look for the WAV file in the output folder
            wav_files = [f for f in os.listdir(output_folder) if f.endswith('.wav')]
            if wav_files:
                wav_path = os.path.join(output_folder, wav_files[0])
                print(f"✓ Audio downloaded and converted successfully: {wav_path}")
                return wav_path
            else:
                print("Failed to find the downloaded WAV file.")
                return None
                
    except Exception as e:
        print(f"\nError downloading YouTube audio with yt-dlp: {str(e)}")
        return None

def process_transcript_with_gemini(transcript_text: str, max_retries=3, retry_delay=30) -> str:
    """
    Process transcribed text using Gemini AI with optimized settings.
    """
    for attempt in range(max_retries):
        try:
            # Initialize Gemini model with optimized configuration
            model = genai.GenerativeModel(
                model_name='gemini-2.0-flash',
                generation_config={
                    "temperature": 0.5,
                    "top_p": 0.7,
                    "top_k": 40,
                    "max_output_tokens": 8192,
                }
            )
            
            # Create the prompt with optimized structure
            prompt = f"""Tôi có một bản ghi âm bài giảng và muốn bạn cải thiện nó để tạo ra một tài liệu tham khảo tốt hơn cho sinh viên. Vui lòng thực hiện các bước sau:

1. *Tóm tắt nội dung:* Tạo một bản tóm tắt ngắn gọn nhưng đầy đủ thông tin về các chủ đề chính được đề cập trong bản ghi âm. Bản tóm tắt thể hiện dưới dạng một đoạn văn.

2. *Chỉnh sửa và cấu trúc lại nội dung:*
   * Loại bỏ các từ đệm, câu nói ấp úng và các phần không liên quan.
   * Sắp xếp lại thông tin một cách logic, sử dụng tiêu đề và dấu đầu dòng để phân chia các chủ đề.
   * Diễn giải rõ ràng các khái niệm phức tạp hoặc các đoạn khó hiểu.
   * Đảm bảo văn phong trang trọng, phù hợp với tài liệu học thuật. Không chứa phần liệt kê các điểm chính.

3. *Liệt kê các điểm chính:* Tạo một danh sách các điểm chính hoặc các khái niệm quan trọng mà sinh viên cần ghi nhớ.

4. *Liệt kê các dặn dò:* Nếu trong quá trình giảng dạy người nói có dặn dò gì thì liệt kê ra

5. *Các câu hỏi:* Nếu trong quá trình giảng dạy người nói có đặt câu hỏi gì thì liệt kê ra, đồng thời trả lời cho câu trả lời đó, ưu tiên trả lời theo gợi ý người nói

6. *Nội dung ôn tập:* Nếu có

Đảm bảo câu trả lời là tiếng việt và chỉ gồm những nội dung được yêu cầu. Không cần câu mở đầu và câu kết thúc. Phần nào có ví dụ từ người nói thì thêm vào để làm rõ khái niệm hoặc vấn đề

Bản ghi âm:
{transcript_text}"""

            print(f"\nProcessing with Gemini AI (Attempt {attempt + 1}/{max_retries})...")
            
            # Generate response with optimized settings
            response = model.generate_content(prompt)
            
            if response.text:
                print("\n✓ Gemini AI processing completed successfully!")
                return response.text
            else:
                print("\nWarning: Empty response from Gemini API")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                continue
                
        except Exception as e:
            print(f"\nError processing transcript with Gemini: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            continue
    
    print("\nFalling back to original transcript after all retries failed")
    return transcript_text

# Main execution
if __name__ == "__main__":
    audio_folder = "audio"
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(audio_folder, exist_ok=True)

    print("\n=== Audio Transcription Process Started ===")
    
    try:
        # Check if YouTube URL is provided as command line argument
        if len(sys.argv) > 1 and "youtube.com" in sys.argv[1]:
            youtube_url = sys.argv[1]
            print(f"\nProcessing YouTube URL: {youtube_url}")
            audio_path = download_youtube_audio_ytdlp(youtube_url, audio_folder)
            if audio_path:
                print("\nTranscribing audio...")
                output_text = transcribe_audio(audio_path)
                # Save raw transcript
                base_name = os.path.splitext(os.path.basename(audio_path))[0]
                transcript_file = os.path.join(output_folder, f"{base_name}.txt")
                with open(transcript_file, "w", encoding="utf-8") as f:
                    f.write(output_text)
                print(f"\n✓ Raw transcript saved to: {transcript_file}")
                
                print("\nProcessing with Gemini AI...")
                processed_text = process_transcript_with_gemini(output_text)
                # Save processed output to a separate file
                processed_file = os.path.join(output_folder, f"{base_name}_processed.txt")
                with open(processed_file, "w", encoding="utf-8") as f:
                    f.write(processed_text)
                print(f"\n✓ Processed output saved to: {processed_file}")
        else:
            # Process all files in the audio folder
            convert_to_wav(audio_folder)

            wav_files = [f for f in os.listdir(audio_folder) if f.endswith(".wav")]
            if not wav_files:
                print("\nNo WAV files found in the audio folder.")
                youtube_url = input("\nPlease enter a YouTube URL (or press Enter to exit): ").strip()
                
                if youtube_url and "youtube.com" in youtube_url:
                    print(f"\nProcessing YouTube URL: {youtube_url}")
                    audio_path = download_youtube_audio_ytdlp(youtube_url, audio_folder)
                    if audio_path:
                        print("\nTranscribing audio...")
                        output_text = transcribe_audio(audio_path)
                        # Save raw transcript
                        base_name = os.path.splitext(os.path.basename(audio_path))[0]
                        transcript_file = os.path.join(output_folder, f"{base_name}.txt")
                        with open(transcript_file, "w", encoding="utf-8") as f:
                            f.write(output_text)
                        print(f"\n✓ Raw transcript saved to: {transcript_file}")
                        
                        print("Processing with Gemini AI...")
                        processed_text = process_transcript_with_gemini(output_text)
                        # Save processed output to a separate file
                        processed_file = os.path.join(output_folder, f"{base_name}_processed.txt")
                        with open(processed_file, "w", encoding="utf-8") as f:
                            f.write(processed_text)
                        print(f"\n✓ Processed output saved to: {processed_file}")
                else:
                    print("\nNo valid YouTube URL provided. Exiting...")
                    sys.exit(0)
            else:
                print(f"\nFound {len(wav_files)} WAV files to transcribe.")
                for audio_name in tqdm(wav_files, desc="Transcribing files"):
                    audio_path = os.path.join(audio_folder, audio_name)
                    print(f"\nProcessing file: {audio_name}")
                    print("Transcribing audio...")
                    output_text = transcribe_audio(audio_path)
                    # Save raw transcript
                    base_name = os.path.splitext(audio_name)[0]
                    transcript_file = os.path.join(output_folder, f"{base_name}.txt")
                    with open(transcript_file, "w", encoding="utf-8") as f:
                        f.write(output_text)
                    print(f"✓ Raw transcript saved to: {transcript_file}")
                    
                    print("Processing with Gemini AI...")
                    processed_text = process_transcript_with_gemini(output_text)
                    # Save processed output to a separate file
                    processed_file = os.path.join(output_folder, f"{base_name}_processed.txt")
                    with open(processed_file, "w", encoding="utf-8") as f:
                        f.write(processed_text)
                    print(f"✓ Processed output saved to: {processed_file}")

        # Clean up audio files after processing
        cleanup_audio_folder(audio_folder)
        
        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("\n=== Audio Transcription Process Completed ===")
        
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        print("Please check the error message above and try again.")
