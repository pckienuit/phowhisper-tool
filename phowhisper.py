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
import random
from difflib import SequenceMatcher
import tempfile
import time

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

def display_info(message: str, level: str = "info") -> None:
    """
    Display information directly on screen with different styles.
    
    Args:
        message (str): Message to display
        level (str): Level of information ('info', 'success', 'warning', 'error')
    """
    if level == "info":
        print(f"\nℹ️  {message}")
    elif level == "success":
        print(f"\n✅ {message}")
    elif level == "warning":
        print(f"\n⚠️  {message}")
    elif level == "error":
        print(f"\n❌ {message}")

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
        
        display_info("\nSplitting audio into chunks...", "info")
        
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
            display_info("No suitable split points found, using fixed-length chunks...", "warning")
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
            display_info("Warning: No valid chunks created. Using original audio file.", "warning")
            temp_file = "temp_chunk_0.wav"
            audio.export(
                temp_file,
                format="wav",
                parameters=["-ac", "1", "-ar", "16000"]
            )
            temp_files.append(temp_file)
        
        display_info(f"Created {len(temp_files)} audio chunks", "info")
        return temp_files
        
    except Exception as e:
        display_info(f"\nError splitting audio: {str(e)}", "error")
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
        display_info(f"\nError processing chunk {chunk_path}: {str(e)}", "error")
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
            display_info("\nConverting audio to required format (16kHz mono)...", "info")
            audio = audio.set_frame_rate(16000).set_channels(1)
            temp_path = "temp_converted.wav"
            audio.export(temp_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
            audio_path = temp_path
        
        if len(audio) > 30000:  # If audio is longer than 30 seconds
            temp_files = split_audio(audio_path)
            transcriptions = []
            
            display_info("\nTranscribing audio chunks...", "info")
            
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
                        display_info(f"\nError processing {chunk_path}: {str(e)}", "error")
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
            display_info("\nTranscribing audio...", "info")
            result = process_chunk(audio_path)
            
            # Clean up temporary converted file if it exists
            if os.path.exists("temp_converted.wav"):
                os.remove("temp_converted.wav")
                
            return result
            
    except Exception as e:
        display_info(f"\nError reading audio file: {str(e)}", "error")
        return f"Error: Could not process audio file - {str(e)}"

def extract_audio_from_video(video_path: str, output_folder: str = "audio") -> str:
    """Extract audio from video file using ffmpeg directly."""
    try:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        temp_wav_path = os.path.join(output_folder, f"{base_name}_converted_temp.wav")
        final_wav_path = os.path.join(output_folder, f"{base_name}.wav")
        
        display_info(f"\nExtracting audio from {os.path.basename(video_path)}...", "info")
        # Use ffmpeg directly to extract audio
        command = [
            'ffmpeg', '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite output file if exists
            temp_wav_path
        ]
        
        subprocess.run(command, check=True, capture_output=True)
        
        # Process the extracted audio for volume
        audio = AudioSegment.from_file(temp_wav_path)
        audio = normalize_audio(audio)
        audio.export(temp_wav_path, format="wav")
        
        # Check if speed optimization is needed
        duration_ms = len(audio)
        if duration_ms > 10 * 60 * 1000:  # If longer than 10 minutes
            display_info("File is longer than 10 minutes. Attempting speed optimization...", "info")
            optimal_speed = find_optimal_audio_speed(temp_wav_path)
            
            if optimal_speed > 1.001:  # Use small tolerance for floating point comparison
                display_info(f"Optimal speed {optimal_speed:.2f}x identified", "success")
                # Create sped-up version
                try:
                    ffmpeg_command = [
                        'ffmpeg', '-i', temp_wav_path,
                        '-filter:a', f'atempo={optimal_speed}',
                        '-vn', '-y', final_wav_path
                    ]
                    subprocess.run(ffmpeg_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    display_info(f"Created sped-up version: {os.path.basename(final_wav_path)}", "success")
                    
                    # Clean up temp file
                    if os.path.exists(temp_wav_path):
                        os.remove(temp_wav_path)
                    
                    return final_wav_path
                except Exception as e:
                    display_info(f"Error creating sped-up version: {str(e)}", "error")
                    display_info("Falling back to 1.0x version", "warning")
            else:
                display_info(f"No speed increase needed (optimal speed: {optimal_speed:.2f}x)", "info")
        
        # If no speed optimization needed or failed, rename temp file to final
        os.rename(temp_wav_path, final_wav_path)
        display_info("Audio extracted and normalized successfully", "success")
        return final_wav_path
        
    except Exception as e:
        display_info(f"\nError extracting audio from {video_path}: {str(e)}", "error")
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
            display_info(f"\nAudio volume too low ({current_dBFS:.1f} dBFS). Adjusting to minimum level...", "info")
            # Calculate how much we need to increase the volume
            increase_by = min_dBFS - current_dBFS
            # Apply the increase
            adjusted_audio = audio.apply_gain(increase_by)
            display_info(f"Volume adjusted to {adjusted_audio.dBFS:.1f} dBFS", "info")
            return adjusted_audio
        return audio
    except Exception as e:
        display_info(f"\nError checking audio volume: {str(e)}", "error")
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
        display_info(f"\nError normalizing audio: {str(e)}", "error")
        return audio

def calculate_text_similarity(text1: str, text2: str) -> dict:
    """
    Calculate various similarity metrics between two texts.
    
    Args:
        text1 (str): First text
        text2 (str): Second text
        
    Returns:
        dict: Dictionary containing various similarity metrics and an average score
    """
    # Clean and normalize texts
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    text1 = clean_text(text1)
    text2 = clean_text(text2)
    
    words1 = text1.split()
    words2 = text2.split()
    
    word_count1 = len(words1)
    word_count2 = len(words2)
    word_count_diff = abs(word_count1 - word_count2)
    word_count_ratio = min(word_count1, word_count2) / max(word_count1, word_count2) if max(word_count1, word_count2) > 0 else 0
    
    char_count1 = len(text1)
    char_count2 = len(text2)
    char_count_diff = abs(char_count1 - char_count2)
    char_count_ratio = min(char_count1, char_count2) / max(char_count1, char_count2) if max(char_count1, char_count2) > 0 else 0
    
    char_similarity = SequenceMatcher(None, text1, text2).ratio()
    word_similarity = SequenceMatcher(None, ' '.join(words1), ' '.join(words2)).ratio()
    
    def get_similar_words(word1, word2, threshold=0.8):
        return SequenceMatcher(None, word1, word2).ratio() >= threshold
    
    similar_words = []
    for w1 in words1:
        for w2 in words2:
            if get_similar_words(w1, w2):
                similar_words.append((w1, w2))
    
    similar_word_pairs = len(similar_words)
    similar_word_ratio = similar_word_pairs / max(len(words1), len(words2)) if max(len(words1), len(words2)) > 0 else 0
    
    # Calculate average similarity score with adjusted weight for word_similarity
    average_similarity = (word_count_ratio + char_count_ratio + char_similarity + (0.5 * word_similarity)) / 3.5
    
    return {
        "word_count": {
            "text1": word_count1,
            "text2": word_count2,
            "difference": word_count_diff,
            "ratio": word_count_ratio
        },
        "char_count": {
            "text1": char_count1,
            "text2": char_count2,
            "difference": char_count_diff,
            "ratio": char_count_ratio
        },
        "similarity": {
            "character_level": char_similarity,
            "word_level": word_similarity,
            "similar_word_pairs": similar_word_pairs,
            "similar_word_ratio": similar_word_ratio
        },
        "average_similarity_score": average_similarity
    }

def extract_random_segments(audio: AudioSegment, segment_lengths: list[int], num_segments: int = 3, min_dBFS: float = -50.0) -> list[tuple[AudioSegment, int]]:
    """
    Extract random segments of different lengths from the audio.
    
    Args:
        audio (AudioSegment): The audio to extract from
        segment_lengths (list[int]): List of segment lengths in milliseconds
        num_segments (int): Number of segments to extract for each length
        min_dBFS (float): Minimum audio level in dBFS to consider as valid audio
        
    Returns:
        list[tuple[AudioSegment, int]]: List of (segment, start_time) tuples
    """
    segments = []
    for length in segment_lengths:
        # Calculate possible start positions
        max_start = len(audio) - length
        if max_start <= 0:
            continue
            
        # Extract num_segments random segments
        attempts = 0
        max_attempts = num_segments * 10  # Allow more attempts to find valid segments
        
        while len(segments) < num_segments and attempts < max_attempts:
            start_time = random.randint(0, max_start)
            segment = audio[start_time:start_time + length]
            
            # Check if segment has actual audio content
            if segment.dBFS > min_dBFS:
                segments.append((segment, start_time))
                display_info(f"Found valid segment at {start_time/1000:.1f}s with level {segment.dBFS:.1f} dBFS", "info")
            else:
                display_info(f"Skipping silent segment at {start_time/1000:.1f}s (level: {segment.dBFS:.1f} dBFS)", "info")
            
            attempts += 1
            
        if len(segments) < num_segments:
            display_info(f"Warning: Could only find {len(segments)} valid segments of length {length/1000:.1f}s", "warning")
            
    return segments

def track_temp_files(action: str, file_path: str = None) -> None:
    """
    Track temporary files created and processed.
    
    Args:
        action (str): Action being performed ('create', 'process', 'cleanup')
        file_path (str): Path to the temporary file
    """
    if not hasattr(track_temp_files, 'temp_files'):
        track_temp_files.temp_files = set()
        track_temp_files.processed_files = set()
    
    if action == 'create' and file_path:
        track_temp_files.temp_files.add(file_path)
        display_info(f"Created temp file: {os.path.basename(file_path)}", "info")
    elif action == 'process' and file_path:
        track_temp_files.processed_files.add(file_path)
        display_info(f"Processed temp file: {os.path.basename(file_path)}", "success")
    elif action == 'cleanup':
        remaining = track_temp_files.temp_files - track_temp_files.processed_files
        if remaining:
            display_info(f"{len(remaining)} temporary files were not processed:", "warning")
            for file in remaining:
                display_info(f"- {os.path.basename(file)}", "warning")
        display_info("Temp file statistics:", "info")
        display_info(f"Total temp files created: {len(track_temp_files.temp_files)}", "info")
        display_info(f"Total temp files processed: {len(track_temp_files.processed_files)}", "info")
        display_info(f"Unprocessed temp files: {len(remaining)}", "info")
        # Reset tracking
        track_temp_files.temp_files = set()
        track_temp_files.processed_files = set()

def convert_to_wav(input_folder: str, output_folder: str) -> list[str]:
    """
    Convert files to WAV format with optimized settings.
    Returns list of files that need processing.
    """
    files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    if not files:
        display_info("\nNo files found in the audio folder.", "warning")
        return []

    display_info(f"\nFound {len(files)} files in '{input_folder}'", "info")
    display_info("Checking files for conversion...", "info")
    files_to_process = []
    
    for file_name in tqdm(files, desc="Checking files"):
        # Check if file has already been processed
        is_converted, is_transcribed = check_file_status(file_name, output_folder)
        
        if is_transcribed:
            display_info(f"\nFile {file_name} has already been fully processed.", "info")
            # Still process the file to allow user interaction
            audio_path = os.path.join(input_folder, file_name)
            process_single_file(audio_path, file_name, output_folder)
            continue
            
        file_path = os.path.join(input_folder, file_name)
        if file_name.endswith(".mp4"):
            display_info(f"\nConverting video file: {file_name}", "info")
            audio_path = extract_audio_from_video(file_path)
            if audio_path:
                track_temp_files('create', audio_path)
                files_to_process.append(os.path.basename(audio_path))
        elif not file_name.endswith(".wav"):
            display_info(f"\nConverting audio file: {file_name}", "info")
            audio = AudioSegment.from_file(file_path)
            # Normalize audio volume
            audio = normalize_audio(audio)
            wav_file_path = os.path.splitext(file_path)[0] + ".wav"
            audio.export(
                wav_file_path,
                format="wav",
                parameters=["-ac", "1", "-ar", "16000"]
            )
            track_temp_files('create', wav_file_path)
            files_to_process.append(os.path.basename(wav_file_path))
        else:
            # If it's already a WAV file, just add it to the list
            files_to_process.append(file_name)
    
    display_info(f"\nFound {len(files_to_process)} files to process", "info")
    return files_to_process

def cleanup_audio_folder(folder_path: str) -> None:
    """Delete all files in the specified audio folder after processing."""
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    if not files:
        return

    display_info("\nCleaning up audio files...", "info")
    for file_name in tqdm(files, desc="Deleting files"):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
            display_info(f"Removed: {file_name}", "success")

def download_youtube_audio(url: str, output_folder: str = "audio") -> str:
    """
    Download audio from a YouTube URL and convert it to WAV format.
    Returns the path to the downloaded audio file.
    """
    try:
        display_info(f"\nDownloading audio from YouTube: {url}", "info")
        # Create YouTube object
        yt = YouTube(url)
        
        # Get the video title and clean it for filename
        video_title = re.sub(r'[^\w\s-]', '', yt.title)
        video_title = re.sub(r'[-\s]+', '-', video_title).strip('-_')
        
        # Get the audio stream
        audio_stream = yt.streams.filter(only_audio=True).first()
        
        if not audio_stream:
            display_info("No audio stream found in the video.", "warning")
            return None
            
        # Download the audio
        display_info("Downloading audio stream...", "info")
        audio_file = audio_stream.download(
            output_path=output_folder,
            filename=f"{video_title}.mp4"
        )
        
        # Convert to WAV format
        display_info("Converting to WAV format...", "info")
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
            
        display_info(f"✓ Audio downloaded and converted successfully: {wav_path}", "success")
        return wav_path
        
    except Exception as e:
        display_info(f"\nError downloading YouTube audio: {str(e)}", "error")
        return None

def download_youtube_audio_ytdlp(url: str, output_folder: str = "audio") -> str:
    """
    Download audio from a YouTube URL using yt-dlp and convert it to WAV format.
    Returns the path to the downloaded audio file.
    """
    try:
        display_info(f"\nDownloading audio from YouTube (yt-dlp): {url}", "info")
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
                
                display_info(f"✓ Audio downloaded, converted and normalized successfully: {wav_path}", "success")
                return wav_path
            else:
                display_info("Failed to find the downloaded WAV file.", "warning")
                return None
                
    except Exception as e:
        display_info(f"\nError downloading YouTube audio with yt-dlp: {str(e)}", "error")
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
        display_info(f"\nError checking Gemini availability: {str(e)}", "error")
        return False

def process_transcript_with_gemini(transcript_text: str, max_retries=3, retry_delay=30) -> str:
    """
    Process transcribed text using Gemini with streaming output.
    """
    # Check Gemini availability first
    if not check_gemini_availability():
        display_info("\nGemini API is not available. Using original transcript.", "info")
        return transcript_text

    prompt = f"""Tôi có một bản ghi âm bài giảng và muốn bạn cải thiện nó để tạo ra một tài liệu tham khảo tốt hơn cho sinh viên. Vui lòng thực hiện các bước sau:\n\n1. *Tóm tắt nội dung:* Tạo một bản tóm tắt ngắn gọn nhưng đầy đủ thông tin về các chủ đề chính được đề cập trong bản ghi âm. Bản tóm tắt thể hiện dưới dạng một đoạn văn.\n\n2. *Chỉnh sửa và cấu trúc lại nội dung:*\n   * Loại bỏ các từ đệm, câu nói ấp úng và các phần không liên quan.\n   * Sắp xếp lại thông tin một cách logic, sử dụng tiêu đề và dấu đầu dòng để phân chia các chủ đề.\n   * Diễn giải rõ ràng các khái niệm phức tạp hoặc các đoạn khó hiểu.\n   * Đảm bảo văn phong trang trọng, phù hợp với tài liệu học thuật. Không chứa phần liệt kê các điểm chính.\n\n3. *Liệt kê các điểm chính:* Tạo một danh sách các điểm chính hoặc các khái niệm quan trọng mà sinh viên cần ghi nhớ.\n\n4. *Liệt kê các dặn dò:* Nếu trong quá trình giảng dạy người nói có dặn dò gì thì liệt kê ra\n\n5. *Các câu hỏi:* Nếu trong quá trình giảng dạy người nói có đặt câu hỏi gì thì liệt kê ra, đồng thời trả lời cho câu trả lời đó, ưu tiên trả lời theo gợi ý người nói\n\n6. *Nội dung ôn tập:* Nếu có\n\nĐảm bảo câu trả lời là tiếng việt và chỉ gồm những nội dung được yêu cầu. Không cần câu mở đầu và câu kết thúc. Phần nào có ví dụ từ người nói thì thêm vào để làm rõ khái niệm hoặc vấn đề\n\nBản ghi âm:\n{transcript_text}"""

    for attempt in range(max_retries):
        display_info(f"\nProcessing with Gemini (Attempt {attempt + 1}/{max_retries})...\n", "info")
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
                    display_info(chunk.text, end="", flush=True)
                    full_response += chunk.text
            
            if full_response:
                display_info("\n\n✓ Gemini processing completed successfully!\n", "success")
                return full_response
            else:
                display_info("\nEmpty response from Gemini", "warning")
                
        except Exception as e:
            display_info(f"\nError from Gemini: {str(e)}", "error")
            if attempt < max_retries - 1:
                display_info(f"Retrying in {retry_delay} seconds...", "info")
                time.sleep(retry_delay)
    
    display_info("\nFalling back to original transcript after all retries failed", "warning")
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
        display_info(f"\nAsking Gemini (Attempt {attempt + 1}/{max_retries})...\n", "info")
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
                    display_info(chunk.text, end="", flush=True)
                    full_response += chunk.text
            
            if full_response:
                display_info("\n\n✓ Gemini response completed successfully!\n", "success")
                return full_response
            else:
                display_info("\nEmpty response from Gemini", "warning")
                
        except Exception as e:
            display_info(f"\nError from Gemini: {str(e)}", "error")
            if attempt < max_retries - 1:
                display_info(f"Retrying in {retry_delay} seconds...", "info")
                time.sleep(retry_delay)
    
    display_info("\nFalling back to original transcript after all retries failed", "warning")
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
        display_info(f"\nFile {audio_name} has already been processed.", "info")
        while True:
            choice = input("\nDo you want to:\n1. Process again from the beginning\n2. Only reprocess with Gemini\n3. Skip this file\nEnter your choice (1/2/3): ").strip()
            
            if choice == "1":
                display_info(f"\nProcessing {audio_name} from the beginning...", "info")
                # First optimize speed if needed
                audio = AudioSegment.from_file(audio_path)
                duration_ms = len(audio)
                
                if duration_ms > 10 * 60 * 1000:  # If longer than 10 minutes
                    display_info("File is longer than 10 minutes. Attempting speed optimization...", "info")
                    optimal_speed = find_optimal_audio_speed(audio_path)
                    
                    if optimal_speed > 1.001:  # Use small tolerance for floating point comparison
                        display_info(f"Optimal speed {optimal_speed:.2f}x identified", "success")
                        # Create sped-up version
                        sped_up_wav_path = os.path.join(output_folder, f"{os.path.splitext(audio_name)[0]}_speed_{optimal_speed:.2f}x.wav")
                        try:
                            ffmpeg_command = [
                                'ffmpeg', '-i', audio_path,
                                '-filter:a', f'atempo={optimal_speed}',
                                '-vn', '-y', sped_up_wav_path
                            ]
                            subprocess.run(ffmpeg_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            display_info(f"Created sped-up version: {os.path.basename(sped_up_wav_path)}", "success")
                            audio_path = sped_up_wav_path  # Use the sped-up version for transcription
                        except Exception as e:
                            display_info(f"Error creating sped-up version: {str(e)}", "error")
                            display_info("Falling back to original speed", "warning")
                
                display_info("Transcribing audio...", "info")
                output_text = transcribe_audio(audio_path)
                # Save raw transcript
                base_name = os.path.splitext(audio_name)[0]
                transcript_file = os.path.join(output_folder, f"{base_name}.txt")
                with open(transcript_file, "w", encoding="utf-8") as f:
                    f.write(output_text)
                display_info(f"✓ Raw transcript saved to: {transcript_file}", "success")
                
                display_info("Processing with Gemini...", "info")
                processed_text = process_transcript_with_gemini(output_text)
                # Save processed output
                processed_file = os.path.join(output_folder, f"{base_name}_processed.txt")
                with open(processed_file, "w", encoding="utf-8") as f:
                    f.write(processed_text)
                display_info(f"✓ Processed output saved to: {processed_file}", "success")
                break
                
            elif choice == "2":
                display_info(f"\nReprocessing {audio_name} with Gemini...", "info")
                # Read existing transcript
                base_name = os.path.splitext(audio_name)[0]
                transcript_file = os.path.join(output_folder, f"{base_name}.txt")
                with open(transcript_file, "r", encoding="utf-8") as f:
                    output_text = f.read()
                
                display_info("Processing with Gemini...", "info")
                processed_text = process_transcript_with_gemini(output_text)
                # Save processed output
                processed_file = os.path.join(output_folder, f"{base_name}_processed.txt")
                with open(processed_file, "w", encoding="utf-8") as f:
                    f.write(processed_text)
                display_info(f"✓ Processed output saved to: {processed_file}", "success")
                break
                
            elif choice == "3":
                display_info(f"\nSkipping {audio_name}", "info")
                break
                
            else:
                display_info("\nInvalid choice. Please enter 1, 2, or 3.", "warning")
    
    elif is_converted:
        display_info(f"\nFile {audio_name} has been transcribed but not processed with Gemini.", "info")
        while True:
            choice = input("\nDo you want to:\n1. Process with Gemini\n2. Skip this file\nEnter your choice (1/2): ").strip()
            
            if choice == "1":
                display_info(f"\nProcessing {audio_name} with Gemini...", "info")
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
                display_info(f"✓ Processed output saved to: {processed_file}", "success")
                break 
                
            elif choice == "2":
                display_info(f"\nSkipping {audio_name}", "info")
                break
                
            else:
                display_info("\nInvalid choice. Please enter 1 or 2.", "warning")
    
    else:
        display_info(f"\nProcessing new file: {audio_name}", "info")
        # First optimize speed if needed
        audio = AudioSegment.from_file(audio_path)
        duration_ms = len(audio)
        
        if duration_ms > 10 * 60 * 1000:  # If longer than 10 minutes
            display_info("File is longer than 10 minutes. Attempting speed optimization...", "info")
            optimal_speed = find_optimal_audio_speed(audio_path)
            
            if optimal_speed > 1.001:  # Use small tolerance for floating point comparison
                display_info(f"Optimal speed {optimal_speed:.2f}x identified", "success")
                # Create sped-up version
                sped_up_wav_path = os.path.join(output_folder, f"{os.path.splitext(audio_name)[0]}_speed_{optimal_speed:.2f}x.wav")
                try:
                    ffmpeg_command = [
                        'ffmpeg', '-i', audio_path,
                        '-filter:a', f'atempo={optimal_speed}',
                        '-vn', '-y', sped_up_wav_path
                    ]
                    subprocess.run(ffmpeg_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    display_info(f"Created sped-up version: {os.path.basename(sped_up_wav_path)}", "success")
                    audio_path = sped_up_wav_path  # Use the sped-up version for transcription
                except Exception as e:
                    display_info(f"Error creating sped-up version: {str(e)}", "error")
                    display_info("Falling back to original speed", "warning")
        
        display_info("Transcribing audio...", "info")
        output_text = transcribe_audio(audio_path)
        # Save raw transcript
        base_name = os.path.splitext(audio_name)[0]
        transcript_file = os.path.join(output_folder, f"{base_name}.txt")
        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write(output_text)
        display_info(f"✓ Raw transcript saved to: {transcript_file}", "success")
        
        display_info("Processing with Gemini...", "info")
        processed_text = process_transcript_with_gemini(output_text)
        # Save processed output
        processed_file = os.path.join(output_folder, f"{base_name}_processed.txt")
        with open(processed_file, "w", encoding="utf-8") as f:
            f.write(processed_text)
        display_info(f"✓ Processed output saved to: {processed_file}", "success")

def find_optimal_audio_speed(clean_wav_path: str) -> float:
    """
    Determines the optimal playback speed for an audio file that maintains
    a minimum transcription accuracy.
    Returns the optimal speed multiplier (e.g., 1.0, 1.25).
    This function is called with a path to a clean (normalized, 16kHz mono) WAV file.
    """
    display_info(f"\nAttempting to find optimal speed for: {os.path.basename(clean_wav_path)}", "info")
    
    try:
        audio = AudioSegment.from_file(clean_wav_path)
    except Exception as e:
        display_info(f"Error loading audio for speed test: {e}. Defaulting to 1.0x speed.", "warning")
        return 1.0

    speeds_to_try = [1.0, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0, 2.25]
    # Configuration for segments: (length_ms, number_of_segments_for_this_length)
    test_segments_config = [(10000, 2), (20000, 1)] # e.g., two 10s segments, one 20s segment
    
    display_info("\nExtracting random segments for speed testing...", "info")
    all_random_segments_data = [] # Will store (AudioSegment_object, start_time_ms)
    for length_ms, num_to_extract in test_segments_config:
         segments_for_this_config = extract_random_segments(audio, [length_ms], num_segments=num_to_extract)
         all_random_segments_data.extend(segments_for_this_config)

    if not all_random_segments_data:
        display_info("No valid audio segments extracted for speed testing. Defaulting to 1.0x speed.", "warning")
        return 1.0
    
    display_info(f"Testing with {len(all_random_segments_data)} random segments.", "info")

    min_similarity_threshold = 0.5  # 50% average similarity needed for a segment to pass
    min_segments_passed_ratio = 0.7 # At least 70% of tested segments must pass the threshold

    speed_test_details = {} # { segment_id: { speed: {avg_similarity, transcript} } }

    # Create a temporary directory for storing segment WAV files for this run
    with tempfile.TemporaryDirectory(prefix="phowhisper_speed_segments_", dir=".") as temp_dir_for_segments:
        baseline_transcripts_map = {} # segment_id: baseline_transcript_text

        # 1. Export segments and get their baseline (1.0x speed) transcripts
        display_info("Generating baseline transcripts for segments...", "info")
        for i, (segment_audio_obj, segment_start_time) in enumerate(all_random_segments_data):
            segment_id_str = f"segment{i}_len{len(segment_audio_obj)/1000:.0f}s_start{segment_start_time/1000:.0f}s"
            original_segment_wav_filename = f"{segment_id_str}_original.wav"
            original_segment_wav_path = os.path.join(temp_dir_for_segments, original_segment_wav_filename)
            
            try:
                segment_audio_obj.export(original_segment_wav_path, format="wav")
                baseline_txt = transcribe_audio(original_segment_wav_path)
                if baseline_txt and not baseline_txt.startswith("Error:") and baseline_txt != "File not found":
                    baseline_transcripts_map[segment_id_str] = baseline_txt
                else:
                    display_info(f"Warning: Failed to get baseline for {segment_id_str}. Reason: '{baseline_txt}'", "warning")
                    baseline_transcripts_map[segment_id_str] = "" # Mark as failed to skip in similarity
            except Exception as e_base:
                display_info(f"Error during baseline processing for {segment_id_str}: {e_base}", "error")
                baseline_transcripts_map[segment_id_str] = ""
        
        num_valid_baselines = sum(1 for t in baseline_transcripts_map.values() if t)
        if num_valid_baselines == 0:
            display_info("No valid baseline transcripts obtained for any segment. Cannot perform speed test. Defaulting to 1.0x.", "warning")
            return 1.0
        display_info(f"Obtained {num_valid_baselines}/{len(all_random_segments_data)} valid baseline transcripts.", "info")

        # 2. Test each speed for each segment that has a valid baseline
        for speed_multiplier in tqdm(speeds_to_try, desc="Testing Speeds", ncols=80, leave=False):
            for i, (segment_audio_obj, segment_start_time) in enumerate(all_random_segments_data):
                segment_id_str = f"segment{i}_len{len(segment_audio_obj)/1000:.0f}s_start{segment_start_time/1000:.0f}s"
                
                if not baseline_transcripts_map.get(segment_id_str): # Skip if baseline failed for this segment
                    continue
                
                if segment_id_str not in speed_test_details: speed_test_details[segment_id_str] = {}

                if speed_multiplier == 1.0:
                    speed_test_details[segment_id_str][speed_multiplier] = {
                        "average_similarity": 1.0, 
                        "transcript": baseline_transcripts_map[segment_id_str]
                    }
                    continue # Already have baseline, similarity is 1.0

                sped_up_wav_filename = f"{segment_id_str}_speed{speed_multiplier:.2f}x.wav"
                sped_up_wav_path = os.path.join(temp_dir_for_segments, sped_up_wav_filename)
                original_segment_wav_path = os.path.join(temp_dir_for_segments, f"{segment_id_str}_original.wav")

                try:
                    ffmpeg_command = ['ffmpeg', '-i', original_segment_wav_path, 
                                      '-filter:a', f'atempo={speed_multiplier}', '-vn', '-y', sped_up_wav_path]
                    subprocess.run(ffmpeg_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    sped_up_txt = transcribe_audio(sped_up_wav_path)
                    if sped_up_txt and not sped_up_txt.startswith("Error:") and sped_up_txt != "File not found":
                        similarity_info = calculate_text_similarity(baseline_transcripts_map[segment_id_str], sped_up_txt)
                        avg_sim = similarity_info["average_similarity_score"]
                        speed_test_details[segment_id_str][speed_multiplier] = {"average_similarity": avg_sim, "transcript": sped_up_txt}
                    else:
                        speed_test_details[segment_id_str][speed_multiplier] = {"average_similarity": 0.0, "transcript": ""}
                except Exception as e_speed_proc:
                    speed_test_details[segment_id_str][speed_multiplier] = {"average_similarity": 0.0, "transcript": ""}

        # 3. Determine optimal speed based on results
        segments_passed_at_each_speed = {s: 0 for s in speeds_to_try}
        for segment_id_str, results_for_segment in speed_test_details.items():
            if not baseline_transcripts_map.get(segment_id_str): continue # Ensure baseline was valid
            for speed_val, result_data in results_for_segment.items():
                if result_data["average_similarity"] >= min_similarity_threshold:
                    segments_passed_at_each_speed[speed_val] += 1
        
        final_optimal_speed = 1.0 # Default
        
        eligible_speeds_meeting_criteria = []
        for speed_val, num_passed_segments in segments_passed_at_each_speed.items():
            if num_valid_baselines > 0 and num_passed_segments >= num_valid_baselines * min_segments_passed_ratio:
                eligible_speeds_meeting_criteria.append(speed_val)
        
        if eligible_speeds_meeting_criteria:
            final_optimal_speed = max(eligible_speeds_meeting_criteria)
        else:
            # Fallback: if no speed meets the 70% segment passage criteria,
            # find the speed that had the highest number of passed segments.
            if any(segments_passed_at_each_speed.values()): # Check if any speed had any success
                max_passed_count = max(segments_passed_at_each_speed.values())
                if max_passed_count > 0:
                    speeds_with_max_passed_count = [s for s, c in segments_passed_at_each_speed.items() if c == max_passed_count]
                    if speeds_with_max_passed_count:
                        final_optimal_speed = max(speeds_with_max_passed_count)
    
    display_info(f"✓ Optimal speed identified for {os.path.basename(clean_wav_path)}: {final_optimal_speed}x", "success")
    return final_optimal_speed

def cleanup_temp_files() -> None:
    """
    Clean up all temporary files before starting the program.
    This includes files in the audio folder, output folder, and any temp files.
    """
    display_info("Cleaning up temporary files...", "info")
    
    # Clean audio folder
    audio_folder = "audio"
    if os.path.exists(audio_folder):
        for file in os.listdir(audio_folder):
            file_path = os.path.join(audio_folder, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    display_info(f"Removed: {file}", "success")
            except Exception as e:
                display_info(f"Error removing {file}: {str(e)}", "error")
    
    # Clean output folder
    output_folder = "output"
    if os.path.exists(output_folder):
        for file in os.listdir(output_folder):
            file_path = os.path.join(output_folder, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    display_info(f"Removed: {file}", "success")
            except Exception as e:
                display_info(f"Error removing {file}: {str(e)}", "error")
    
    # Clean any temp files in current directory
    for file in os.listdir("."):
        if file.startswith("temp_") or file.endswith("_temp.wav"):
            try:
                if os.path.isfile(file):
                    os.remove(file)
                    display_info(f"Removed: {file}", "success")
            except Exception as e:
                display_info(f"Error removing {file}: {str(e)}", "error")
    
    display_info("Temporary files cleanup completed", "success")

def process_files_parallel(files_to_process: list[str], input_folder: str, output_folder: str, max_workers: int = None) -> None:
    """
    Process multiple files in parallel using ThreadPoolExecutor.
    
    Args:
        files_to_process (list[str]): List of files to process
        input_folder (str): Input folder path
        output_folder (str): Output folder path
        max_workers (int): Maximum number of worker threads
    """
    if not max_workers:
        max_workers = min(os.cpu_count() or 4, 4)  # Limit to 4 workers by default
    
    display_info(f"\nProcessing {len(files_to_process)} files in parallel using {max_workers} workers", "info")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for file_name in files_to_process:
            audio_path = os.path.join(input_folder, file_name)
            future = executor.submit(process_single_file, audio_path, file_name, output_folder)
            futures.append(future)
        
        # Wait for all tasks to complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            try:
                future.result()
            except Exception as e:
                display_info(f"Error processing file: {str(e)}", "error")

def optimize_memory_usage():
    """
    Optimize memory usage by clearing caches and unused memory.
    """
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Clear Python garbage collector
    gc.collect()
    
    # Clear system cache if possible
    if os.name == 'nt':  # Windows
        os.system('powershell -Command "Clear-RecycleBin -Force"')
    elif os.name == 'posix':  # Linux/Mac
        os.system('sync; echo 3 > /proc/sys/vm/drop_caches')

def optimize_model_for_inference():
    """
    Optimize the model for faster inference.
    """
    global transcriber
    
    if torch.cuda.is_available():
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Move model to GPU and optimize
        transcriber.model = transcriber.model.cuda()
        transcriber.model = transcriber.model.eval()
        
        # Enable model optimization
        if hasattr(transcriber.model, 'enable_model_cpu_offload'):
            transcriber.model.enable_model_cpu_offload()
        
        # Use mixed precision for faster inference
        with torch.cuda.amp.autocast():
            transcriber.model = torch.compile(transcriber.model)
    else:
        # CPU optimizations
        transcriber.model = transcriber.model.eval()
        transcriber.model = torch.compile(transcriber.model)

def optimize_audio_processing(audio: AudioSegment) -> AudioSegment:
    """
    Optimize audio processing for faster transcription.
    
    Args:
        audio (AudioSegment): Input audio segment
        
    Returns:
        AudioSegment: Optimized audio segment
    """
    try:
        # Convert to mono if stereo
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Set optimal sample rate
        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000)
        
        # Normalize audio
        audio = normalize_audio(audio)
        
        # Apply noise reduction if needed
        if audio.dBFS < -30:
            # Simple noise gate
            audio = audio.apply_gain(10)  # Boost quiet parts
        
        return audio
    except Exception as e:
        display_info(f"Error optimizing audio: {str(e)}", "error")
        return audio

def create_temp_audio_file(audio: AudioSegment, prefix: str = "temp", suffix: str = ".wav") -> str:
    """
    Create a temporary audio file in the audio folder.
    
    Args:
        audio (AudioSegment): Audio segment to save
        prefix (str): Prefix for the temporary file
        suffix (str): File extension
        
    Returns:
        str: Path to the created temporary file
    """
    try:
        # Create a unique filename
        temp_filename = f"{prefix}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}{suffix}"
        temp_path = os.path.join("audio", temp_filename)
        
        # Save the audio file
        audio.export(
            temp_path,
            format="wav",
            parameters=["-ac", "1", "-ar", "16000"]
        )
        
        track_temp_files('create', temp_path)
        return temp_path
    except Exception as e:
        display_info(f"Error creating temporary audio file: {str(e)}", "error")
        return None

def cleanup_audio_files():
    """
    Clean up all audio files in the audio folder.
    """
    display_info("\nCleaning up audio files...", "info")
    
    audio_folder = "audio"
    if not os.path.exists(audio_folder):
        return
        
    try:
        # Get list of all files in audio folder
        files = [f for f in os.listdir(audio_folder) if os.path.isfile(os.path.join(audio_folder, f))]
        
        if not files:
            display_info("No files to clean up in audio folder", "info")
            return
            
        display_info(f"Found {len(files)} files to clean up", "info")
        
        # Remove each file
        for file_name in tqdm(files, desc="Cleaning up files"):
            file_path = os.path.join(audio_folder, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    display_info(f"Removed: {file_name}", "success")
            except Exception as e:
                display_info(f"Error removing {file_name}: {str(e)}", "error")
                
        display_info("Audio folder cleanup completed", "success")
    except Exception as e:
        display_info(f"Error during cleanup: {str(e)}", "error")

def process_audio_chunk(chunk: AudioSegment, chunk_id: int) -> str:
    """
    Process a single audio chunk with optimized settings.
    
    Args:
        chunk (AudioSegment): Audio chunk to process
        chunk_id (int): Chunk identifier
        
    Returns:
        str: Transcribed text
    """
    try:
        # Optimize chunk
        chunk = optimize_audio_processing(chunk)
        
        # Save to temporary file in audio folder
        temp_file = create_temp_audio_file(chunk, f"chunk_{chunk_id}")
        if not temp_file:
            return ""
        
        # Process chunk
        result = process_chunk(temp_file)
        
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)
            track_temp_files('process', temp_file)
        
        return result
    except Exception as e:
        display_info(f"Error processing chunk {chunk_id}: {str(e)}", "error")
        return ""

def split_audio_optimized(audio_path: str, chunk_length_ms: int = 30000) -> List[str]:
    """
    Split audio into optimized chunks with improved performance.
    """
    try:
        audio = AudioSegment.from_file(audio_path)
        audio = optimize_audio_processing(audio)
        temp_files = []
        
        display_info("\nSplitting audio into chunks...", "info")
        
        # Calculate optimal chunk size based on available memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            # Adjust chunk size based on GPU memory
            chunk_length_ms = min(chunk_length_ms, int(gpu_memory / (1024 * 1024 * 10)))  # 10MB per chunk
        
        # Create chunks
        chunks = make_chunks(audio, chunk_length_ms)
        
        # Process chunks in parallel
        with ThreadPoolExecutor() as executor:
            futures = []
            for i, chunk in enumerate(chunks):
                future = executor.submit(process_audio_chunk, chunk, i)
                futures.append(future)
            
            # Collect results
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
                try:
                    result = future.result()
                    if result:
                        temp_files.append(result)
                except Exception as e:
                    display_info(f"Error processing chunk: {str(e)}", "error")
        
        return temp_files
    except Exception as e:
        display_info(f"Error splitting audio: {str(e)}", "error")
        return []

# Main execution
if __name__ == "__main__":
    audio_folder = "audio"
    output_folder = "output"
    
    # Create necessary folders
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(audio_folder, exist_ok=True)
    
    # Optimize system resources
    optimize_memory_usage()
    optimize_model_for_inference()
    
    # Clean up temporary files before starting
    cleanup_temp_files()
    cleanup_audio_files()

    display_info("\n=== Audio Transcription Process Started ===", "info")
    
    try:
        # Check if YouTube URL is provided as command line argument
        if len(sys.argv) > 1 and ("youtube.com" in sys.argv[1] or "youtu.be" in sys.argv[1]):
            youtube_url = convert_youtube_url(sys.argv[1])
            display_info(f"\nProcessing YouTube URL: {youtube_url}", "info")
            audio_path = download_youtube_audio_ytdlp(youtube_url, audio_folder)
            if audio_path:
                audio_name = os.path.basename(audio_path)
                process_single_file(audio_path, audio_name, output_folder)
        else:
            # Process all files in the audio folder
            wav_files = convert_to_wav(audio_folder, output_folder)
    
            if not wav_files:
                display_info("\nNo new files to process.", "warning")
                youtube_url = input("\nPlease enter a YouTube URL (or press Enter to exit): ").strip()
                
                if youtube_url and ("youtube.com" in youtube_url or "youtu.be" in youtube_url):
                    youtube_url = convert_youtube_url(youtube_url)
                    display_info(f"\nProcessing YouTube URL: {youtube_url}", "info")
                    audio_path = download_youtube_audio_ytdlp(youtube_url, audio_folder)
                    if audio_path:
                        audio_name = os.path.basename(audio_path)
                        process_single_file(audio_path, audio_name, output_folder)
                else:
                    display_info("\nNo valid YouTube URL provided. Exiting...", "warning")
                sys.exit(0)
            else:
                display_info(f"\nFound {len(wav_files)} files to process.", "info")
                # Process files in parallel
                process_files_parallel(wav_files, audio_folder, output_folder)

        # Clean up audio files after processing
        cleanup_audio_files()
        
        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        display_info("\n=== Audio Transcription Process Completed ===", "success")
        
    except Exception as e:
        display_info(f"\nError during processing: {str(e)}", "error")
        display_info("Please check the error message above and try again.", "warning")
        
        # Clean up on error
        cleanup_audio_files()
