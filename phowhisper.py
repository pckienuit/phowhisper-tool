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
import requests
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

genai.configure(api_key=GEMINI_API_KEY)

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
    try:
        audio = AudioSegment.from_file(audio_path)
        # Normalize audio volume before splitting
        audio = normalize_audio(audio)
        temp_files = []
        
        print("\nSplitting audio into chunks...")
        
        # Calculate frame length for analysis (20ms frames for faster processing)
        frame_length = 20
        frames = make_chunks(audio, frame_length)
        
        # Find optimal split points
        split_points = []
        current_chunk_start = 0
        min_silence_threshold = -40  # Adjusted threshold for better chunking
        min_silence_duration = 500   # Increased minimum silence duration for more reliable splits
        min_chunk_length = 5000      # Minimum chunk length in milliseconds
        
        # Process frames in batches for better performance
        batch_size = 100  # Increased batch size for faster processing
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            batch_dBFS = [frame.dBFS for frame in batch]
            
            for j, dBFS in enumerate(batch_dBFS):
                frame_idx = i + j
                current_time = frame_idx * frame_length
                
                # Check if we've reached minimum chunk length and found silence
                if (current_time - current_chunk_start >= min_chunk_length and 
                    dBFS <= min_silence_threshold):
                    # Look ahead to ensure this is a good split point
                    look_ahead = min(10, len(frames) - frame_idx - 1)
                    if look_ahead > 0:
                        next_frames = frames[frame_idx:frame_idx + look_ahead]
                        next_dBFS = [frame.dBFS for frame in next_frames]
                        if all(dB <= min_silence_threshold for dB in next_dBFS):
                            split_points.append(current_time)
                            current_chunk_start = current_time
        
        # Add the end of the audio as the final split point if needed
        if len(audio) - current_chunk_start >= min_chunk_length:
            split_points.append(len(audio))
        
        # If no good split points found, create chunks of fixed length
        if not split_points:
            print("No suitable split points found, using fixed-length chunks...")
            for i in range(0, len(audio), chunk_length_ms):
                split_points.append(i)
            split_points.append(len(audio))
        
        # Create chunks based on split points
        for i in range(len(split_points) - 1):
            start = split_points[i]
            end = split_points[i + 1]
            chunk = audio[start:end]
            
            # Only save chunks that are long enough and have audible sound
            if len(chunk) >= min_chunk_length and chunk.dBFS > min_silence_threshold:
                temp_file = f"temp_chunk_{i}.wav"
                chunk.export(
                    temp_file,
                    format="wav",
                    parameters=["-ac", "1", "-ar", "16000"]
                )
                temp_files.append(temp_file)
        
        if not temp_files:
            print("Warning: No valid chunks created. Using original audio file.")
            temp_file = "temp_chunk_0.wav"
            audio.export(
                temp_file,
                format="wav",
                parameters=["-ac", "1", "-ar", "16000"]
            )
            temp_files.append(temp_file)
        
        print(f"Created {len(temp_files)} audio chunks")
        return temp_files
        
    except Exception as e:
        print(f"\nError splitting audio: {str(e)}")
        # Fallback: return the original file as a single chunk
        temp_file = "temp_chunk_0.wav"
        try:
            audio.export(
                temp_file,
                format="wav",
                parameters=["-ac", "1", "-ar", "16000"]
            )
            return [temp_file]
        except:
            raise Exception(f"Failed to process audio file: {str(e)}")

def process_chunk(chunk_path: str) -> str:
    """Process a single audio chunk with optimized settings."""
    try:
        # Use cached transcriber instance with optimized settings
        if not hasattr(process_chunk, 'transcriber'):
            process_chunk.transcriber = pipeline(
                "automatic-speech-recognition",
                model="vinai/PhoWhisper-medium",
                device="cuda" if torch.cuda.is_available() else "cpu",
                return_timestamps=True,
                framework="pt",
                torch_dtype=torch.float16,  # Use half precision for faster processing
                model_kwargs={
                    "use_cache": True,
                    "low_cpu_mem_usage": True
                }
            )
            # Enable model optimization
            if hasattr(process_chunk.transcriber.model, 'enable_model_cpu_offload'):
                process_chunk.transcriber.model.enable_model_cpu_offload()
            
            # Optimize model for inference
            if torch.cuda.is_available():
                process_chunk.transcriber.model = process_chunk.transcriber.model.eval()
                with torch.cuda.amp.autocast():
                    process_chunk.transcriber.model = torch.compile(process_chunk.transcriber.model)
        
        # Process the chunk
        result = process_chunk.transcriber(chunk_path)
        return result['text']
    except Exception as e:
        print(f"\nError processing chunk {chunk_path}: {str(e)}")
        return ""

def transcribe_audio(audio_path: str, max_workers: int = None) -> str:
    """Transcribe audio using optimized parallel processing."""
    if not os.path.exists(audio_path):
        return "File not found"
    
    try:
        # Validate WAV file format
        audio = AudioSegment.from_file(audio_path)
        
        # Check if audio is valid
        if len(audio) == 0:
            return "Error: Empty audio file"
            
        # Ensure correct audio format
        if audio.channels != 1 or audio.frame_rate != 16000:
            print("\nConverting audio to required format (16kHz mono)...")
            audio = audio.set_frame_rate(16000).set_channels(1)
            temp_path = "temp_converted.wav"
            audio.export(temp_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
            audio_path = temp_path
        
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
                gc.collect()  # Add garbage collection
            
            # Clean up temporary converted file if it exists
            if os.path.exists("temp_converted.wav"):
                os.remove("temp_converted.wav")
                
            return " ".join(transcriptions).strip()
        else:
            print("\nTranscribing audio...")
            result = process_chunk(audio_path)
            
            # Clean up temporary converted file if it exists
            if os.path.exists("temp_converted.wav"):
                os.remove("temp_converted.wav")
                
            return result
            
    except Exception as e:
        print(f"\nError reading audio file: {str(e)}")
        return f"Error: Could not process audio file - {str(e)}"

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
        
        # Process the extracted audio for volume
        audio = AudioSegment.from_file(output_path)
        audio = normalize_audio(audio)
        audio.export(output_path, format="wav")
        
        print(f"✓ Audio extracted and normalized successfully")
        return output_path
    except Exception as e:
        print(f"\nError extracting audio from {video_path}: {str(e)}")
        return None

def check_and_adjust_volume(audio: AudioSegment, min_dBFS: float = -30.0) -> AudioSegment:
    """
    Check if audio volume is too low and adjust it if necessary.
    
    Args:
        audio (AudioSegment): The audio segment to check
        min_dBFS (float): Minimum acceptable dBFS level (default: -30.0)
        
    Returns:
        AudioSegment: Adjusted audio segment if needed
    """
    try:
        current_dBFS = audio.dBFS
        if current_dBFS < min_dBFS:
            print(f"\nAudio volume too low ({current_dBFS:.1f} dBFS). Adjusting to minimum level...")
            # Calculate how much we need to increase the volume
            increase_by = min_dBFS - current_dBFS
            # Apply the increase
            adjusted_audio = audio.apply_gain(increase_by)
            print(f"Volume adjusted to {adjusted_audio.dBFS:.1f} dBFS")
            return adjusted_audio
        return audio
    except Exception as e:
        print(f"\nError checking audio volume: {str(e)}")
        return audio

def normalize_audio(audio: AudioSegment, target_dBFS: float = -20.0) -> AudioSegment:
    """
    Normalize audio to a target dBFS level.
    
    Args:
        audio (AudioSegment): The audio segment to normalize
        target_dBFS (float): Target dBFS level (default: -20.0, which is a good level for speech)
        
    Returns:
        AudioSegment: Normalized audio segment
    """
    try:
        # First check if volume is too low and adjust if necessary
        audio = check_and_adjust_volume(audio)
        
        # Calculate the difference between current and target dBFS
        change_in_dBFS = target_dBFS - audio.dBFS
        
        # Apply the change
        normalized_audio = audio.apply_gain(change_in_dBFS)
        
        # Ensure the audio doesn't clip
        if normalized_audio.max_dBFS > 0:
            # If audio would clip, reduce gain to prevent clipping
            reduction = normalized_audio.max_dBFS
            normalized_audio = normalized_audio.apply_gain(-reduction)
        
        return normalized_audio
    except Exception as e:
        print(f"\nError normalizing audio: {str(e)}")
        return audio

def convert_to_wav(input_folder: str, output_folder: str) -> list[str]:
    """
    Convert files to WAV format with optimized settings.
    Returns list of files that need processing.
    """
    files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    if not files:
        print("\nNo files found in the audio folder.")
        return []

    print("\nChecking files for conversion...")
    files_to_process = []
    
    for file_name in tqdm(files, desc="Checking files"):
        # Check if file has already been processed
        is_converted, is_transcribed = check_file_status(file_name, output_folder)
        
        if is_transcribed:
            print(f"\nFile {file_name} has already been fully processed.")
            # Still process the file to allow user interaction
            audio_path = os.path.join(input_folder, file_name)
            process_single_file(audio_path, file_name, output_folder)
            continue
            
        file_path = os.path.join(input_folder, file_name)
        if file_name.endswith(".mp4"):
            print(f"\nConverting video file: {file_name}")
            audio_path = extract_audio_from_video(file_path)
            if audio_path:
                files_to_process.append(os.path.basename(audio_path))
        elif not file_name.endswith(".wav"):
            print(f"\nConverting audio file: {file_name}")
            audio = AudioSegment.from_file(file_path)
            # Normalize audio volume
            audio = normalize_audio(audio)
            wav_file_path = os.path.splitext(file_path)[0] + ".wav"
            audio.export(
                wav_file_path,
                format="wav",
                parameters=["-ac", "1", "-ar", "16000"]
            )
            files_to_process.append(os.path.basename(wav_file_path))
        else:
            # If it's already a WAV file, just add it to the list
            files_to_process.append(file_name)
    
    return files_to_process

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
                
                # Process the downloaded audio for volume
                audio = AudioSegment.from_file(wav_path)
                audio = normalize_audio(audio)
                audio.export(wav_path, format="wav")
                
                print(f"✓ Audio downloaded, converted and normalized successfully: {wav_path}")
                return wav_path
            else:
                print("Failed to find the downloaded WAV file.")
                return None
                
    except Exception as e:
        print(f"\nError downloading YouTube audio with yt-dlp: {str(e)}")
        return None

def check_gemini_availability() -> bool:
    """
    Check if Gemini API is available and working.
    
    Returns:
        bool: True if Gemini is available and working, False otherwise
    """
    try:
        # Initialize Gemini model with generation config
        generation_config = {
            "temperature": 0.5,
            "top_p": 0.7,
            "top_k": 40,
            "max_output_tokens": 128000,
        }
        
        model = genai.GenerativeModel(
            model_name='gemini-2.5-flash-preview-05-20',
            generation_config=generation_config
        )
        
        # Try a simple test prompt
        response = model.generate_content("Test connection")
        return True
    except Exception as e:
        print(f"\nError checking Gemini availability: {str(e)}")
        return False

def process_transcript_with_gemini(transcript_text: str, max_retries=3, retry_delay=30) -> str:
    """
    Process transcribed text using Gemini with streaming output.
    """
    # Check Gemini availability first
    if not check_gemini_availability():
        print("\nGemini API is not available. Using original transcript.")
        return transcript_text

    prompt = f"""Tôi có một bản ghi âm bài giảng và muốn bạn cải thiện nó để tạo ra một tài liệu tham khảo tốt hơn cho sinh viên. Vui lòng thực hiện các bước sau:\n\n1. *Tóm tắt nội dung:* Tạo một bản tóm tắt ngắn gọn nhưng đầy đủ thông tin về các chủ đề chính được đề cập trong bản ghi âm. Bản tóm tắt thể hiện dưới dạng một đoạn văn.\n\n2. *Chỉnh sửa và cấu trúc lại nội dung:*\n   * Loại bỏ các từ đệm, câu nói ấp úng và các phần không liên quan.\n   * Sắp xếp lại thông tin một cách logic, sử dụng tiêu đề và dấu đầu dòng để phân chia các chủ đề.\n   * Diễn giải rõ ràng các khái niệm phức tạp hoặc các đoạn khó hiểu.\n   * Đảm bảo văn phong trang trọng, phù hợp với tài liệu học thuật. Không chứa phần liệt kê các điểm chính.\n\n3. *Liệt kê các điểm chính:* Tạo một danh sách các điểm chính hoặc các khái niệm quan trọng mà sinh viên cần ghi nhớ.\n\n4. *Liệt kê các dặn dò:* Nếu trong quá trình giảng dạy người nói có dặn dò gì thì liệt kê ra\n\n5. *Các câu hỏi:* Nếu trong quá trình giảng dạy người nói có đặt câu hỏi gì thì liệt kê ra, đồng thời trả lời cho câu trả lời đó, ưu tiên trả lời theo gợi ý người nói\n\n6. *Nội dung ôn tập:* Nếu có\n\nĐảm bảo câu trả lời là tiếng việt và chỉ gồm những nội dung được yêu cầu. Không cần câu mở đầu và câu kết thúc. Phần nào có ví dụ từ người nói thì thêm vào để làm rõ khái niệm hoặc vấn đề\n\nBản ghi âm:\n{transcript_text}"""

    for attempt in range(max_retries):
        print(f"\nProcessing with Gemini (Attempt {attempt + 1}/{max_retries})...\n")
        try:
            # Initialize Gemini model with generation config
            generation_config = {
                "temperature": 0.5,
                "top_p": 0.7,
                "top_k": 40,
                "max_output_tokens": 128000,
            }
            
            model = genai.GenerativeModel(
                model_name='gemini-2.5-flash-preview-05-20',
                generation_config=generation_config
            )
            
            # Generate response with streaming
            response = model.generate_content(prompt, stream=True)
            
            full_response = ""
            for chunk in response:
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    full_response += chunk.text
            
            if full_response:
                print("\n\n✓ Gemini processing completed successfully!\n")
                return full_response
            else:
                print("\nEmpty response from Gemini")
                
        except Exception as e:
            print(f"\nError from Gemini: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                import time
                time.sleep(retry_delay)
    
    print("\nFalling back to original transcript after all retries failed")
    return transcript_text

def ask_gemini_question(transcript_text: str, question: str, max_retries=3, retry_delay=30) -> str:
    """
    Ask a question about the transcript using Gemini.
    
    Args:
        transcript_text (str): The transcript text to ask questions about
        question (str): The question to ask
        max_retries (int): Maximum number of retry attempts
        retry_delay (int): Delay between retries in seconds
        
    Returns:
        str: Gemini's response to the question
    """
    # Check Gemini availability first
    if not check_gemini_availability():
        return "Gemini API is not available. Please try again later."

    prompt = f"""Dựa trên bản ghi âm bài giảng sau, hãy trả lời câu hỏi của tôi một cách chi tiết và chính xác nhất có thể. Nếu câu trả lời không có trong bản ghi âm, hãy nói rõ điều đó.

Bản ghi âm:
{transcript_text}

Câu hỏi:
{question}

Hãy trả lời bằng tiếng Việt và đảm bảo câu trả lời:
1. Chính xác và dựa trên thông tin từ bản ghi âm
2. Đầy đủ và chi tiết
3. Dễ hiểu và có cấu trúc rõ ràng
4. Nếu có ví dụ từ bản ghi âm, hãy trích dẫn để làm rõ câu trả lời"""

    for attempt in range(max_retries):
        print(f"\nAsking Gemini (Attempt {attempt + 1}/{max_retries})...\n")
        try:
            # Initialize Gemini model with generation config
            generation_config = {
                "temperature": 0.5,
                "top_p": 0.7,
                "top_k": 40,
                "max_output_tokens": 128000,
            }
            
            model = genai.GenerativeModel(
                model_name='gemini-2.5-flash-preview-05-20',
                generation_config=generation_config
            )
            
            # Generate response with streaming
            response = model.generate_content(prompt, stream=True)
            
            full_response = ""
            for chunk in response:
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    full_response += chunk.text
            
            if full_response:
                print("\n\n✓ Gemini response completed successfully!\n")
                return full_response
            else:
                print("\nEmpty response from Gemini")
                
        except Exception as e:
            print(f"\nError from Gemini: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                import time
                time.sleep(retry_delay)
    
    print("\nFalling back to original transcript after all retries failed")
    return "Không thể nhận được câu trả lời từ Gemini. Vui lòng thử lại sau."

def convert_youtube_url(url: str) -> str:
    """
    Convert a shortened YouTube URL (youtu.be) to its full format (youtube.com/watch?v=).
    
    Args:
        url (str): The YouTube URL to convert
        
    Returns:
        str: The converted URL in full format
    """
    if "youtu.be/" in url:
        # Extract the video ID from the shortened URL
        video_id = url.split("youtu.be/")[1]
        # Convert to full format
        return f"https://youtube.com/watch?v={video_id}"
    return url

def check_file_status(audio_name: str, output_folder: str) -> tuple[bool, bool]:
    """
    Check if a file has been converted and transcribed.
    
    Args:
        audio_name (str): Name of the audio file
        output_folder (str): Path to the output folder
        
    Returns:
        tuple[bool, bool]: (is_converted, is_transcribed)
    """
    base_name = os.path.splitext(audio_name)[0]
    transcript_file = os.path.join(output_folder, f"{base_name}.txt")
    processed_file = os.path.join(output_folder, f"{base_name}_processed.txt")
    
    is_converted = os.path.exists(transcript_file)
    is_transcribed = os.path.exists(processed_file)
    
    return is_converted, is_transcribed

def process_single_file(audio_path: str, audio_name: str, output_folder: str) -> None:
    """
    Process a single audio file with user interaction.
    
    Args:
        audio_path (str): Path to the audio file
        audio_name (str): Name of the audio file
        output_folder (str): Path to the output folder
    """
    is_converted, is_transcribed = check_file_status(audio_name, output_folder)
    
    if is_transcribed:
        print(f"\nFile {audio_name} has already been processed.")
        while True:
            choice = input("\nDo you want to:\n1. Process again from the beginning\n2. Only reprocess with Gemini\n3. Skip this file\nEnter your choice (1/2/3): ").strip()
            
            if choice == "1":
                print(f"\nProcessing {audio_name} from the beginning...")
                print("Transcribing audio...")
                output_text = transcribe_audio(audio_path)
                # Save raw transcript
                base_name = os.path.splitext(audio_name)[0]
                transcript_file = os.path.join(output_folder, f"{base_name}.txt")
                with open(transcript_file, "w", encoding="utf-8") as f:
                    f.write(output_text)
                print(f"✓ Raw transcript saved to: {transcript_file}")
                
                print("Processing with Gemini...")
                processed_text = process_transcript_with_gemini(output_text)
                # Save processed output
                processed_file = os.path.join(output_folder, f"{base_name}_processed.txt")
                with open(processed_file, "w", encoding="utf-8") as f:
                    f.write(processed_text)
                print(f"✓ Processed output saved to: {processed_file}")
                break
                
            elif choice == "2":
                print(f"\nReprocessing {audio_name} with Gemini...")
                # Read existing transcript
                base_name = os.path.splitext(audio_name)[0]
                transcript_file = os.path.join(output_folder, f"{base_name}.txt")
                with open(transcript_file, "r", encoding="utf-8") as f:
                    output_text = f.read()
                
                print("Processing with Gemini...")
                processed_text = process_transcript_with_gemini(output_text)
                # Save processed output
                processed_file = os.path.join(output_folder, f"{base_name}_processed.txt")
                with open(processed_file, "w", encoding="utf-8") as f:
                    f.write(processed_text)
                print(f"✓ Processed output saved to: {processed_file}")
                break
                
            elif choice == "3":
                print(f"\nSkipping {audio_name}")
                break
                
            else:
                print("\nInvalid choice. Please enter 1, 2, or 3.")
    
    elif is_converted:
        print(f"\nFile {audio_name} has been transcribed but not processed with Gemini.")
        while True:
            choice = input("\nDo you want to:\n1. Process with Gemini\n2. Skip this file\nEnter your choice (1/2): ").strip()
            
            if choice == "1":
                print(f"\nProcessing {audio_name} with Gemini...")
                # Read existing transcript
                base_name = os.path.splitext(audio_name)[0]
                transcript_file = os.path.join(output_folder, f"{base_name}.txt")
                with open(transcript_file, "r", encoding="utf-8") as f:
                    output_text = f.read()
                
                processed_text = process_transcript_with_gemini(output_text)
                # Save processed output
                processed_file = os.path.join(output_folder, f"{base_name}_processed.txt")
                with open(processed_file, "w", encoding="utf-8") as f:
                    f.write(processed_text)
                print(f"✓ Processed output saved to: {processed_file}")
                break
                
            elif choice == "2":
                print(f"\nSkipping {audio_name}")
                break
                
            else:
                print("\nInvalid choice. Please enter 1 or 2.")
    
    else:
        print(f"\nProcessing new file: {audio_name}")
        print("Transcribing audio...")
        output_text = transcribe_audio(audio_path)
        # Save raw transcript
        base_name = os.path.splitext(audio_name)[0]
        transcript_file = os.path.join(output_folder, f"{base_name}.txt")
        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write(output_text)
        print(f"✓ Raw transcript saved to: {transcript_file}")
        
        print("Processing with Gemini...")
        processed_text = process_transcript_with_gemini(output_text)
        # Save processed output
        processed_file = os.path.join(output_folder, f"{base_name}_processed.txt")
        with open(processed_file, "w", encoding="utf-8") as f:
            f.write(processed_text)
        print(f"✓ Processed output saved to: {processed_file}")

# Main execution
if __name__ == "__main__":
    audio_folder = "audio"
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(audio_folder, exist_ok=True)

    print("\n=== Audio Transcription Process Started ===")
    
    try:
        # Check if YouTube URL is provided as command line argument
        if len(sys.argv) > 1 and ("youtube.com" in sys.argv[1] or "youtu.be" in sys.argv[1]):
            youtube_url = convert_youtube_url(sys.argv[1])
            print(f"\nProcessing YouTube URL: {youtube_url}")
            audio_path = download_youtube_audio_ytdlp(youtube_url, audio_folder)
            if audio_path:
                audio_name = os.path.basename(audio_path)
                process_single_file(audio_path, audio_name, output_folder)
        else:
            # Process all files in the audio folder
            wav_files = convert_to_wav(audio_folder, output_folder)
    
            if not wav_files:
                print("\nNo new files to process.")
                youtube_url = input("\nPlease enter a YouTube URL (or press Enter to exit): ").strip()
                
                if youtube_url and ("youtube.com" in youtube_url or "youtu.be" in youtube_url):
                    youtube_url = convert_youtube_url(youtube_url)
                    print(f"\nProcessing YouTube URL: {youtube_url}")
                    audio_path = download_youtube_audio_ytdlp(youtube_url, audio_folder)
                    if audio_path:
                        audio_name = os.path.basename(audio_path)
                        process_single_file(audio_path, audio_name, output_folder)
                else:
                    print("\nNo valid YouTube URL provided. Exiting...")
                    sys.exit(0)
            else:
                print(f"\nFound {len(wav_files)} files to process.")
                for audio_name in wav_files:
                    audio_path = os.path.join(audio_folder, audio_name)
                    process_single_file(audio_path, audio_name, output_folder)

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
