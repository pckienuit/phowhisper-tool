from transformers import pipeline
import os
from pydub import AudioSegment
from pydub.utils import make_chunks
import torch
import sys
import subprocess
# Removed ThreadPoolExecutor for single-threaded processing
from concurrent.futures import as_completed  # Keep for compatibility
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
import argparse
from urllib.parse import urlparse, parse_qs

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

genai.configure(api_key=GEMINI_API_KEY)

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

def select_device(mode: str = None, cli_device: str = None) -> str:
    """
    Select device for processing: auto (prefer GPU), or manual (ask user or use CLI arg).
    Args:
        mode (str): 'auto' or 'manual'
        cli_device (str): 'cpu' or 'cuda' if provided from CLI
    Returns:
        str: 'cpu' or 'cuda'
    """
    if mode == 'auto':
        if torch.cuda.is_available():
            display_info("\n[Auto] GPU detected. Using GPU (CUDA) for processing.", "info")
            return "cuda"
        else:
            display_info("\n[Auto] GPU not available. Using CPU for processing.", "warning")
            return "cpu"
    elif mode == 'manual':
        if cli_device in ('cpu', 'cuda'):
            display_info(f"\n[Manual] Using {cli_device.upper()} for processing (from CLI)", "info")
            return cli_device
        # If not provided, ask user
        if not torch.cuda.is_available():
            display_info("\nGPU not available. Using CPU for processing.", "warning")
            return "cpu"
        while True:
            choice = input("\nSelect processing device:\n1. CPU\n2. GPU (CUDA)\nEnter your choice (1/2): ").strip()
            if choice == "1":
                display_info("\nUsing CPU for processing.", "info")
                return "cpu"
            elif choice == "2":
                display_info("\nUsing GPU (CUDA) for processing.", "info")
                return "cuda"
            else:
                display_info("\nInvalid choice. Please enter 1 or 2.", "warning")
    else:
        # Default: auto
        if torch.cuda.is_available():
            display_info("\n[Default] GPU detected. Using GPU (CUDA) for processing.", "info")
            return "cuda"
        else:
            display_info("\n[Default] GPU not available. Using CPU for processing.", "warning")
            return "cpu"

# Initialize model with optimized settings
selected_device = select_device()
transcriber = pipeline(
    "automatic-speech-recognition",
    model="vinai/PhoWhisper-large",
    device=selected_device,
    return_timestamps=True,
    framework="pt",
    torch_dtype=torch.float16 if selected_device == "cuda" else torch.float32,  # Use half precision only for GPU
    model_kwargs={"use_cache": True}  # Enable model caching
)

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

# Suppress torchvision image extension warnings
warnings.filterwarnings('ignore', message='Failed to load image Python extension')
warnings.filterwarnings('ignore', module='torchvision.io.image')

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
        
        display_info(f"\nSplitting audio into {chunk_length_ms/1000:.0f}s chunks...", "info")
        
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
        chunk_count = 0
        for i in range(len(split_points) - 1):
            start = split_points[i]
            end = split_points[i + 1]
            chunk = audio[start:end]
            
            # Only save chunks that are long enough and have audible sound
            if len(chunk) >= min_chunk_length and chunk.dBFS > min_silence_threshold:
                temp_file = f"temp_chunk_{chunk_count}.wav"
                chunk.export(
                    temp_file,
                    format="wav",
                    parameters=["-ac", "1", "-ar", "16000"]
                )
                temp_files.append(temp_file)
                chunk_count += 1
        
        if not temp_files:
            display_info("Warning: No valid chunks created. Using original audio file.", "warning")
            temp_file = "temp_chunk_0.wav"
            audio.export(
                temp_file,
                format="wav",
                parameters=["-ac", "1", "-ar", "16000"]
            )
            temp_files.append(temp_file)
        
        display_info(f"Created {len(temp_files)} audio chunks for sequential processing", "info")
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
    """Process a single audio chunk with optimized settings for single-threaded processing."""
    try:
        # Use the global transcriber instance for better efficiency
        result = transcriber(chunk_path)
        return result['text']
    except Exception as e:
        display_info(f"\nError processing chunk {chunk_path}: {str(e)}", "error")
        return ""

def transcribe_audio(audio_path: str, max_workers: int = None) -> str:
    """Transcribe audio using optimized single-threaded processing."""
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
            # Use larger chunks for better efficiency in single-threaded mode
            chunk_length = 60000 if torch.cuda.is_available() else 45000  # 60s for GPU, 45s for CPU
            temp_files = split_audio(audio_path, chunk_length)
            transcriptions = []
            
            display_info("\nTranscribing audio chunks sequentially...", "info")
            
            # Process chunks sequentially with progress bar
            for i, temp_file in enumerate(tqdm(temp_files, desc="Processing chunks", unit="chunk")):
                try:
                    # Process chunk
                    text = process_chunk(temp_file)
                    if text:
                        transcriptions.append(text)
                    
                    # Clear CUDA cache after each chunk to prevent memory buildup
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    display_info(f"\nError processing chunk {i+1}: {str(e)}", "error")
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
            
            # Final cleanup
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

def cleanup_audio_files():
    """
    Clean up all audio files in the audio folder that have been processed.
    Only deletes files that have corresponding processed files in the output folder.
    """
    display_info("\nCleaning up audio files...", "info")
    
    audio_folder = "audio"
    output_folder = "output"
    
    if not os.path.exists(audio_folder):
        return
        
    try:
        # Get list of all files in audio folder
        files = [f for f in os.listdir(audio_folder) if os.path.isfile(os.path.join(audio_folder, f))]
        
        if not files:
            display_info("No files to clean up in audio folder", "info")
            return
            
        display_info(f"Found {len(files)} files to check", "info")
        
        # Check and remove each file
        for file_name in tqdm(files, desc="Checking files"):
            file_path = os.path.join(audio_folder, file_name)
            base_name = os.path.splitext(file_name)[0]
            processed_file = os.path.join(output_folder, f"{base_name}_processed.txt")
            
            try:
                if os.path.isfile(file_path):
                    # Check if file has been processed
                    if os.path.exists(processed_file):
                        display_info(f"File {file_name} has been processed. Skipping.", "info")
                    else:
                        display_info(f"File {file_name} has not been processed. Keeping.", "info")
            except Exception as e:
                display_info(f"Error checking/removing {file_name}: {str(e)}", "error")
                
        display_info("Audio folder cleanup completed", "success")
    except Exception as e:
        display_info(f"Error during cleanup: {str(e)}", "error")

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

    try:
        # Read the system prompt from file
        with open("systemprompt", "r", encoding="utf-8") as f:
            prompt_template = f.read()
        
        # Format the prompt with the transcript
        prompt = f"{prompt_template}\n\nBản ghi âm:\n{transcript_text}"
    except Exception as e:
        display_info(f"\nError reading system prompt: {str(e)}", "error")
        return transcript_text

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
            response = model.generate_content(prompt, stream=False)
            
            full_response = ""
            for chunk in response:
                if chunk.text:
                    print(chunk.text, end="", flush=True)
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
                    print(chunk.text, end="", flush=True)
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

def extract_google_drive_id(url: str) -> str:
    """
    Extract file ID from Google Drive URL.
    
    Args:
        url (str): Google Drive URL
        
    Returns:
        str: File ID or None if not found
    """
    try:
        # Pattern 1: https://drive.google.com/file/d/FILE_ID/view
        pattern1 = r'/file/d/([a-zA-Z0-9_-]+)'
        match = re.search(pattern1, url)
        if match:
            return match.group(1)
        
        # Pattern 2: https://drive.google.com/open?id=FILE_ID
        pattern2 = r'[?&]id=([a-zA-Z0-9_-]+)'
        match = re.search(pattern2, url)
        if match:
            return match.group(1)
        
        # Pattern 3: Direct file ID (if someone just pastes the ID)
        if len(url) > 20 and len(url) < 50 and re.match(r'^[a-zA-Z0-9_-]+$', url):
            return url
            
        return None
    except Exception as e:
        display_info(f"Error extracting Google Drive ID: {str(e)}", "error")
        return None

def is_google_drive_url(url: str) -> bool:
    """
    Check if URL is a Google Drive link.
    
    Args:
        url (str): URL to check
        
    Returns:
        bool: True if it's a Google Drive URL
    """
    return "drive.google.com" in url.lower()

def download_google_drive_file(url: str, output_folder: str = "audio") -> str:
    """
    Download file from Google Drive URL using gdown library with proper filename detection.
    
    Args:
        url (str): Google Drive URL
        output_folder (str): Folder to save downloaded file
        
    Returns:
        str: Path to downloaded file or None if failed
    """
    try:
        import gdown
        
        # Extract file ID
        file_id = extract_google_drive_id(url)
        if not file_id:
            display_info("Could not extract file ID from Google Drive URL", "error")
            return None
        
        display_info(f"\nDownloading from Google Drive (ID: {file_id[:10]}...)", "info")
        
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Use gdown to download with proper filename detection
        download_url = f"https://drive.google.com/uc?id={file_id}"
        
        # Let gdown determine the filename automatically
        display_info("Getting file information...", "info")
        
        try:
            # Use gdown.download with output=None to auto-detect filename
            downloaded_path = gdown.download(download_url, output=None, quiet=False, fuzzy=True)
            
            if downloaded_path and os.path.exists(downloaded_path):
                # Move file to our output folder
                original_filename = os.path.basename(downloaded_path)
                final_path = os.path.join(output_folder, original_filename)
                
                # If file already exists in output folder, remove it first
                if os.path.exists(final_path):
                    os.remove(final_path)
                
                # Move the downloaded file
                os.rename(downloaded_path, final_path)
                
                file_size = os.path.getsize(final_path)
                display_info(f"✓ Downloaded: {original_filename} ({file_size:,} bytes)", "success")
                
                # Check if it's a valid audio file by trying to load it
                try:
                    # Test if file can be loaded as audio
                    test_audio = AudioSegment.from_file(final_path)
                    display_info(f"✓ File verified as valid audio: {len(test_audio)/1000:.1f}s duration", "success")
                    
                    # Convert to WAV if not already
                    if not original_filename.lower().endswith('.wav'):
                        display_info("Converting to WAV format...", "info")
                        audio = normalize_audio(test_audio)
                        wav_path = os.path.splitext(final_path)[0] + '.wav'
                        audio.export(
                            wav_path,
                            format="wav",
                            parameters=["-ac", "1", "-ar", "16000"]
                        )
                        
                        # Remove original file
                        os.remove(final_path)
                        display_info(f"✓ Converted to WAV: {os.path.basename(wav_path)}", "success")
                        return wav_path
                    else:
                        return final_path
                        
                except Exception as e:
                    display_info(f"Error: File is not a valid audio format - {str(e)}", "error")
                    display_info("The downloaded file may be corrupted or not an audio file", "error")
                    os.remove(final_path)
                    return None
            else:
                display_info("gdown failed to download file", "error")
                return None
                
        except Exception as e:
            display_info(f"gdown download failed: {str(e)}", "error")
            
            # Try alternative gdown method with manual filename
            display_info("Trying alternative gdown method...", "info")
            try:
                temp_path = os.path.join(output_folder, f"gdrive_temp_{file_id[:8]}")
                result = gdown.download(download_url, output=temp_path, quiet=False)
                
                if result and os.path.exists(temp_path):
                    # Try to detect file type and set proper extension
                    try:
                        audio = AudioSegment.from_file(temp_path)
                        final_filename = f"gdrive_audio_{file_id[:8]}.wav"
                        final_path = os.path.join(output_folder, final_filename)
                        
                        audio = normalize_audio(audio)
                        audio.export(
                            final_path,
                            format="wav",
                            parameters=["-ac", "1", "-ar", "16000"]
                        )
                        
                        os.remove(temp_path)
                        display_info(f"✓ Downloaded and converted: {final_filename}", "success")
                        return final_path
                        
                    except Exception:
                        # If can't process as audio, keep original
                        final_filename = f"gdrive_file_{file_id[:8]}"
                        final_path = os.path.join(output_folder, final_filename)
                        os.rename(temp_path, final_path)
                        display_info(f"Downloaded but could not verify as audio: {final_filename}", "warning")
                        return None
                else:
                    display_info("Alternative gdown method also failed", "error")
                    return None
                    
            except Exception as e2:
                display_info(f"Alternative gdown method failed: {str(e2)}", "error")
                return None
        
        display_info("All download methods failed. Possible issues:", "error")
        display_info("1. File is not publicly accessible (not set to 'Anyone with the link')", "error")
        display_info("2. File requires Google account authentication", "error")
        display_info("3. File has been moved or deleted", "error")
        display_info("4. Google Drive quota exceeded", "error")
        display_info("\nTo fix this:", "info")
        display_info("1. Right-click file in Google Drive → Share → Change to 'Anyone with the link'", "info")
        display_info("2. Try downloading manually first to verify the link works", "info")
        display_info("3. Consider uploading to a different service (YouTube, Dropbox, etc.)", "info")
        
        return None
            
    except ImportError:
        display_info("gdown library not installed. Installing...", "warning")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            display_info("✓ gdown installed. Please run the command again.", "success")
        except:
            display_info("Failed to install gdown. Please install manually: pip install gdown", "error")
        return None
    except Exception as e:
        display_info(f"\nError downloading from Google Drive: {str(e)}", "error")
        return None

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
    Process a single audio file without any interactive input.
    Args:
        audio_path (str): Path to the audio file
        audio_name (str): Name of the audio file
        output_folder (str): Path to the output folder
    """
    is_converted, is_transcribed = check_file_status(audio_name, output_folder)
    base_name = os.path.splitext(audio_name)[0]
    transcript_file = os.path.join(output_folder, f"{base_name}.txt")
    processed_file = os.path.join(output_folder, f"{base_name}_processed.txt")

    if is_transcribed:
        display_info(f"\nFile {audio_name} has already been processed. Skipping...", "info")
        return
    elif is_converted:
        display_info(f"\nFile {audio_name} has been transcribed but not processed with Gemini. Processing with Gemini...", "info")
        with open(transcript_file, "r", encoding="utf-8") as f:
            output_text = f.read()
        processed_text = process_transcript_with_gemini(output_text)
        with open(processed_file, "w", encoding="utf-8") as f:
            f.write(processed_text)
        display_info(f"✓ Processed output saved to: {processed_file}", "success")
    else:
        display_info(f"\nProcessing new file: {audio_name}", "info")
        # If file is already a _speed file, skip optimal speed detection
        if '_speed' in audio_name:
            display_info("File is a _speed file. Skipping optimal speed detection.", "info")
        else:
            audio = AudioSegment.from_file(audio_path)
            duration_ms = len(audio)
            if duration_ms > 10 * 60 * 1000:  # If longer than 10 minutes
                display_info("File is longer than 10 minutes. Attempting speed optimization...", "info")
                optimal_speed = find_optimal_audio_speed(audio_path)
                if optimal_speed > 1.001:
                    display_info(f"Optimal speed {optimal_speed:.2f}x identified", "success")
                    sped_up_wav_path = os.path.join("audio", f"{os.path.splitext(audio_name)[0]}_speed_{optimal_speed:.2f}x.wav")
                    try:
                        ffmpeg_command = [
                            'ffmpeg', '-i', audio_path,
                            '-filter:a', f'atempo={optimal_speed}',
                            '-vn', '-y', sped_up_wav_path
                        ]
                        subprocess.run(ffmpeg_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        display_info(f"Created sped-up version: {os.path.basename(sped_up_wav_path)}", "success")
                        audio_path = sped_up_wav_path
                    except Exception as e:
                        display_info(f"Error creating sped-up version: {str(e)}", "error")
                        display_info("Falling back to original speed", "warning")
        display_info("Transcribing audio...", "info")
        output_text = transcribe_audio(audio_path)
        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write(output_text)
        display_info(f"✓ Raw transcript saved to: {transcript_file}", "success")
        display_info("Processing with Gemini...", "info")
        processed_text = process_transcript_with_gemini(output_text)
        with open(processed_file, "w", encoding="utf-8") as f:
            f.write(processed_text)
        display_info(f"✓ Processed output saved to: {processed_file}", "success")

def find_optimal_audio_speed(clean_wav_path: str) -> float:
    """
    Determines the optimal playback speed for an audio file that maintains
    a minimum transcription accuracy.
    Returns the optimal speed multiplier (e.g., 1.0, 1.25).
    """
    display_info(f"\nFinding optimal speed for: {os.path.basename(clean_wav_path)}", "info")
    
    try:
        audio = AudioSegment.from_file(clean_wav_path)
    except Exception as e:
        display_info(f"Error loading audio: {e}. Using default speed 1.0x.", "warning")
        return 1.0

    # Test speeds from 1.0x to 2.25x
    speeds_to_try = [1.0, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0, 2.25]
    
    # Extract a 30-second segment from the middle of the audio
    middle_point = len(audio) // 2
    test_segment = audio[middle_point - 15000:middle_point + 15000]
    
    if len(test_segment) < 10000:  # If audio is too short
        display_info("Audio is too short for speed testing. Using default speed 1.0x.", "warning")
        return 1.0
    
    # Save the test segment
    test_segment_path = "temp_speed_test.wav"
    test_segment.export(test_segment_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
    
    try:
        # Get baseline transcript at 1.0x speed
        baseline_text = transcribe_audio(test_segment_path)
        if not baseline_text or baseline_text.startswith("Error:"):
            display_info("Failed to get baseline transcript. Using default speed 1.0x.", "warning")
            return 1.0
            
        best_speed = 1.0
        best_similarity = 1.0
        min_similarity_threshold = 0.5  # 50% average similarity needed for a segment to pass
        min_segments_passed_ratio = 0.7  # At least 70% of tested segments must pass the threshold
        
        # Test each speed
        speed_results = {speed: 0 for speed in speeds_to_try}  # Track successful tests for each speed
        
        for speed in tqdm(speeds_to_try, desc="Testing speeds", ncols=80, leave=False):
            if speed == 1.0:
                continue  # Skip baseline speed
                
            # Create sped-up version
            sped_up_path = f"temp_speed_test_{speed}x.wav"
            try:
                ffmpeg_command = [
                    'ffmpeg', '-i', test_segment_path,
                    '-filter:a', f'atempo={speed}',
                    '-vn', '-y', sped_up_path
                ]
                subprocess.run(ffmpeg_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Get transcript at this speed
                sped_up_text = transcribe_audio(sped_up_path)
                if sped_up_text and not sped_up_text.startswith("Error:"):
                    # Calculate similarity
                    similarity = calculate_text_similarity(baseline_text, sped_up_text)["average_similarity_score"]
                    
                    # Track successful tests
                    if similarity >= min_similarity_threshold:
                        speed_results[speed] += 1
                    
                    # Update best speed if this one is better
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_speed = speed
                        
            except Exception as e:
                display_info(f"Error testing speed {speed}x: {str(e)}", "warning")
                continue
            finally:
                # Clean up sped-up file
                if os.path.exists(sped_up_path):
                    os.remove(sped_up_path)
        
        # Clean up test segment
        if os.path.exists(test_segment_path):
            os.remove(test_segment_path)
            
        # Find speeds that meet the minimum threshold
        eligible_speeds = []
        for speed, success_count in speed_results.items():
            if success_count >= min_segments_passed_ratio:
                eligible_speeds.append(speed)
        
        if eligible_speeds:
            # Choose the highest speed that meets the criteria
            final_speed = max(eligible_speeds)
            display_info(f"✓ Optimal speed found: {final_speed}x (similarity: {best_similarity:.2f})", "success")
            return final_speed
        else:
            # If no speed meets the criteria, use the one with best similarity
            if best_similarity > min_similarity_threshold:
                display_info(f"✓ Best available speed found: {best_speed}x (similarity: {best_similarity:.2f})", "success")
                return best_speed
            else:
                display_info("No suitable speed found. Using default speed 1.0x.", "info")
                return 1.0
            
    except Exception as e:
        display_info(f"Error during speed testing: {str(e)}. Using default speed 1.0x.", "warning")
        return 1.0

def cleanup_temp_files() -> None:
    """
    Clean up only temporary files in the root directory.
    Does not delete files in the audio and output folders.
    """
    display_info("Cleaning up temporary files...", "info")
    
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
    Process multiple files sequentially for better stability and resource management.
    
    Args:
        files_to_process (list[str]): List of files to process
        input_folder (str): Input folder path
        output_folder (str): Output folder path
        max_workers (int): Ignored in single-threaded mode (kept for compatibility)
    """
    display_info(f"\nProcessing {len(files_to_process)} files sequentially", "info")
    
    # Process files one by one with progress tracking
    for i, file_name in enumerate(tqdm(files_to_process, desc="Processing files", unit="file")):
        try:
            audio_path = os.path.join(input_folder, file_name)
            display_info(f"\n[{i+1}/{len(files_to_process)}] Processing: {file_name}", "info")
            
            # Process single file
            process_single_file(audio_path, file_name, output_folder)
            
            # Clear memory after each file
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            display_info(f"✓ Completed: {file_name}", "success")
            
        except Exception as e:
            display_info(f"✗ Error processing {file_name}: {str(e)}", "error")
            continue  # Continue with next file
    
    display_info(f"\n✅ Completed processing all {len(files_to_process)} files", "success")

def optimize_memory_usage():
    """
    Optimize memory usage by clearing caches and unused memory.
    """
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Clear Python garbage collector
    gc.collect()
    
    # Clear system cache if possible (only on Linux/Mac)
    if os.name == 'posix':  # Linux/Mac
        os.system('sync; echo 3 > /proc/sys/vm/drop_caches')

def optimize_model_for_inference():
    """
    Optimize the model for faster inference in single-threaded mode.
    """
    global transcriber
    
    try:
        if transcriber.device == "cuda":
            # Enable CUDA optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Move model to GPU and optimize
            transcriber.model = transcriber.model.cuda()
            transcriber.model = transcriber.model.eval()
            
            # Enable model optimization for single-threaded processing
            if hasattr(transcriber.model, 'enable_model_cpu_offload'):
                transcriber.model.enable_model_cpu_offload()
            
            display_info("GPU model optimized for single-threaded inference", "info")
        else:
            # CPU optimizations
            transcriber.model = transcriber.model.eval()
            # Set optimal thread count for CPU inference
            torch.set_num_threads(min(4, torch.get_num_threads()))
            display_info("CPU model optimized for single-threaded inference", "info")
            
    except Exception as e:
        display_info(f"Warning: Model optimization failed: {str(e)}", "warning")

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
        
        # Process chunks sequentially
        for i, chunk in enumerate(chunks):
            try:
                # Optimize chunk
                chunk = optimize_audio_processing(chunk)
                
                # Save to temporary file in audio folder
                temp_file = create_temp_audio_file(chunk, f"chunk_{i}")
                if not temp_file:
                    continue
                
                # Process chunk
                result = process_audio_chunk(chunk, i)
                if result:
                    temp_files.append(result)
                
                # Clean up
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    track_temp_files('process', temp_file)
                    
            except Exception as e:
                display_info(f"Error processing chunk {i}: {str(e)}", "error")
                continue
        
        return temp_files
    except Exception as e:
        display_info(f"Error splitting audio: {str(e)}", "error")
        return []

def get_urls() -> list[str]:
    """
    Get multiple URLs (YouTube or Google Drive) from user input until they confirm.
    
    Returns:
        list[str]: List of valid URLs
    """
    urls = []
    display_info("\nEnter URLs (YouTube or Google Drive) one per line.", "info")
    display_info("Supported formats:", "info")
    display_info("- YouTube: https://youtube.com/watch?v=... or https://youtu.be/...", "info")
    display_info("- Google Drive: https://drive.google.com/file/d/.../view", "info")
    display_info("Enter 'done' when finished, or 'exit' to cancel.", "info")
    
    while True:
        url = input("\nEnter URL (or 'done'/'exit'): ").strip()
        
        if url.lower() == 'exit':
            return []
        elif url.lower() == 'done':
            if not urls:
                display_info("No URLs provided. Please enter at least one URL.", "warning")
                continue
            break
        elif "youtube.com" in url or "youtu.be" in url:
            urls.append(convert_youtube_url(url))
            display_info(f"Added YouTube URL: {url}", "success")
        elif "drive.google.com" in url:
            urls.append(url)
            display_info(f"Added Google Drive URL: {url}", "success")
        else:
            display_info("Invalid URL. Please enter a valid YouTube or Google Drive URL.", "warning")
            display_info("Examples:", "info")
            display_info("  YouTube: https://youtube.com/watch?v=VIDEO_ID", "info")
            display_info("  Google Drive: https://drive.google.com/file/d/FILE_ID/view", "info")
    
    return urls

def download_from_url(url: str, output_folder: str = "audio") -> str:
    """
    Download file from URL (YouTube or Google Drive).
    
    Args:
        url (str): URL to download from
        output_folder (str): Folder to save downloaded file
        
    Returns:
        str: Path to downloaded file or None if failed
    """
    try:
        if "youtube.com" in url or "youtu.be" in url:
            display_info("Detected YouTube URL", "info")
            return download_youtube_audio_ytdlp(url, output_folder)
        elif "drive.google.com" in url:
            display_info("Detected Google Drive URL", "info")
            # Try the basic method first
            result = download_google_drive_file(url, output_folder)
            if not result:
                # Fallback to gdown method
                display_info("Trying alternative download method...", "info")
                result = download_google_drive_file_gdown(url, output_folder)
            return result
        else:
            display_info(f"Unsupported URL format: {url}", "warning")
            return None
    except Exception as e:
        display_info(f"Error downloading from URL: {str(e)}", "error")
        return None

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PhoWhisper CLI")
    parser.add_argument('--mode', choices=['auto', 'manual'], default='auto', help='Device selection mode: auto/manual (default: auto)')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None, help='Device to use if manual')
    args, unknown = parser.parse_known_args()

    audio_folder = "audio"
    output_folder = "output"
    
    # Create necessary folders
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(audio_folder, exist_ok=True)
    
    # Optimize system resources
    optimize_memory_usage()
    
    # Device selection
    selected_device = select_device(args.mode, args.device)
    transcriber = pipeline(
        "automatic-speech-recognition",
        model="vinai/PhoWhisper-medium",
        device=selected_device,
        return_timestamps=True,
        framework="pt",
        torch_dtype=torch.float16 if selected_device == "cuda" else torch.float32,  # Use half precision only for GPU
        model_kwargs={"use_cache": True}  # Enable model caching
    )
    optimize_model_for_inference()
    
    # Clean up temporary files before starting
    cleanup_temp_files()
    cleanup_audio_files()

    display_info("\n=== Audio Transcription Process Started ===", "info")
    
    try:
        # Check if URL is provided as command line argument
        if len(sys.argv) > 1:
            input_url = sys.argv[1]
            if "youtube.com" in input_url or "youtu.be" in input_url or "drive.google.com" in input_url:
                display_info(f"\nProcessing URL: {input_url}", "info")
                audio_path = download_from_url(input_url, audio_folder)
                if audio_path:
                    audio_name = os.path.basename(audio_path)
                    process_single_file(audio_path, audio_name, output_folder)
                else:
                    display_info("Failed to download from URL", "error")
                    sys.exit(1)
            else:
                display_info(f"Unsupported URL format: {input_url}", "error")
                sys.exit(1)
        else:
            # Process all files in the audio folder
            wav_files = convert_to_wav(audio_folder, output_folder)
    
            if not wav_files:
                display_info("\nNo new files to process.", "warning")
                urls = get_urls()
                
                if urls:
                    display_info(f"\nProcessing {len(urls)} URLs...", "info")
                    for i, url in enumerate(urls):
                        display_info(f"\n[{i+1}/{len(urls)}] Processing URL: {url}", "info")
                        audio_path = download_from_url(url, audio_folder)
                        if audio_path:
                            audio_name = os.path.basename(audio_path)
                            process_single_file(audio_path, audio_name, output_folder)
                        else:
                            display_info(f"Failed to download from URL: {url}", "error")
                else:
                    display_info("\nNo URLs provided. Exiting...", "warning")
                sys.exit(0)
            else:
                display_info(f"\nFound {len(wav_files)} files to process.", "info")
                
                # Auto mode: delete already processed files, process the rest
                if args.mode == 'auto':
                    files_to_process = []
                    # Build a set of base names for speed files
                    speed_bases = set()
                    for file_name in wav_files:
                        if '_speed' in file_name:
                            base = file_name.split('_speed')[0]
                            speed_bases.add(base)
                    # Now process files
                    for file_name in wav_files:
                        # If this is an original file and a _speed version exists, skip it
                        if ('_speed' not in file_name) and (file_name.replace('.wav','') in speed_bases):
                            continue
                        is_converted, is_transcribed = check_file_status(file_name, output_folder)
                        file_path = os.path.join(audio_folder, file_name)
                        if is_transcribed:
                            try:
                                os.remove(file_path)
                                display_info(f"[Auto] Deleted already processed file: {file_name}", "success")
                            except Exception as e:
                                display_info(f"[Auto] Error deleting file {file_name}: {str(e)}", "error")
                        else:
                            files_to_process.append(file_name)
                    if files_to_process:
                        process_files_parallel(files_to_process, audio_folder, output_folder)
                    else:
                        display_info("[Auto] No files left to process.", "info")
                    # Skip menu in auto mode
                else:
                    # Display file management options
                    while True:
                        display_info("\nFile Management Options:", "info")
                        display_info("1. Process all files", "info")
                        display_info("2. Select specific files to process", "info")
                        display_info("3. Skip all files", "info")
                        display_info("4. Delete all files in audio folder", "info")
                        display_info("5. Exit", "info")
                        
                        choice = input("\nEnter your choice (1-5): ").strip()
                        
                        if choice == "1":
                            # Process all files in parallel
                            process_files_parallel(wav_files, audio_folder, output_folder)
                            break
                            
                        elif choice == "2":
                            # List all files with numbers
                            display_info("\nAvailable files:", "info")
                            for i, file in enumerate(wav_files, 1):
                                display_info(f"{i}. {file}", "info")
                            
                            # Get user selection
                            while True:
                                selection = input("\nEnter file numbers to process (comma-separated) or 'all' for all files: ").strip()
                                
                                if selection.lower() == 'all':
                                    selected_files = wav_files
                                    break
                                else:
                                    try:
                                        # Parse selected numbers
                                        numbers = [int(n.strip()) for n in selection.split(',')]
                                        # Validate numbers
                                        if all(1 <= n <= len(wav_files) for n in numbers):
                                            selected_files = [wav_files[n-1] for n in numbers]
                                            break
                                        else:
                                            display_info("Invalid file numbers. Please try again.", "warning")
                                    except ValueError:
                                        display_info("Invalid input. Please enter numbers separated by commas.", "warning")
                            
                            # Process selected files
                            if selected_files:
                                process_files_parallel(selected_files, audio_folder, output_folder)
                            break
                            
                        elif choice == "3":
                            display_info("\nSkipping all files.", "info")
                            break
                            
                        elif choice == "4":
                            confirm = input("\nAre you sure you want to delete all files in the audio folder? (yes/no): ").strip().lower()
                            if confirm == 'yes':
                                try:
                                    for file in os.listdir(audio_folder):
                                        file_path = os.path.join(audio_folder, file)
                                        if os.path.isfile(file_path):
                                            os.remove(file_path)
                                    display_info("\nAll files in audio folder have been deleted.", "success")
                                except Exception as e:
                                    display_info(f"\nError deleting files: {str(e)}", "error")
                            else:
                                display_info("\nOperation cancelled.", "info")
                            break
                            
                        elif choice == "5":
                            display_info("\nExiting program.", "info")
                            sys.exit(0)
                            
                        else:
                            display_info("\nInvalid choice. Please enter a number between 1 and 5.", "warning")

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
