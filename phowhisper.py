import os
import sys

# --- CRITICAL: HuggingFace & Transformers Protection ---
# Must be set before any transformers/huggingface_hub imports
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Low-level monkeypatching to prevent background thread crashes (HTTP 403)
try:
    import huggingface_hub.hf_api
    # Disable discussion fetching globally (this is what triggers 403)
    huggingface_hub.hf_api.get_repo_discussions = lambda *args, **kwargs: []
    
    import transformers.safetensors_conversion
    # Disable the auto-conversion thread entirely
    transformers.safetensors_conversion.auto_conversion = lambda *args, **kwargs: None
except:
    pass
# -------------------------------------------------------

from transformers import pipeline
from pydub import AudioSegment
from pydub.utils import make_chunks
import torch
import subprocess
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
import warnings
from dotenv import load_dotenv
import random
from difflib import SequenceMatcher
import time
import argparse
from scipy import signal



# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# --- Local LLM (Ollama) Configuration ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
LOCAL_MODEL_NAME = "gemma3:12b"


# --- Gemini API Compatibility Layer ---
# Supports both google.genai (new) and google.generativeai (old/deprecated)
_USE_NEW_GENAI = False
_gemini_client = None

try:
    from google import genai as _genai_module
    _USE_NEW_GENAI = True
    _gemini_client = _genai_module.Client(api_key=GEMINI_API_KEY)
except ImportError:
    warnings.filterwarnings('ignore', category=FutureWarning, module='google.generativeai')
    import google.generativeai as _genai_module
    _genai_module.configure(api_key=GEMINI_API_KEY)


class _GeminiModelCompat:
    """Compatibility wrapper providing unified API for both genai packages."""
    
    def __init__(self, model_name, generation_config):
        self.model_name = model_name
        self.generation_config = generation_config
        
        if not _USE_NEW_GENAI:
            # Old API: create model object directly
            self._old_model = _genai_module.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config
            )
    
    def generate_content(self, prompt, stream=False):
        if _USE_NEW_GENAI:
            config = _genai_module.types.GenerateContentConfig(
                temperature=self.generation_config.get('temperature'),
                top_p=self.generation_config.get('top_p'),
                top_k=self.generation_config.get('top_k'),
                max_output_tokens=self.generation_config.get('max_output_tokens'),
            )
            if stream:
                return _gemini_client.models.generate_content_stream(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
            else:
                response = _gemini_client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
                # Wrap in a list-like for compatibility with old iteration pattern
                return _NonStreamResponse(response)
        else:
            return self._old_model.generate_content(prompt, stream=stream)


class _NonStreamResponse:
    """Wraps a non-streaming response to support iteration like the old API."""
    
    def __init__(self, response):
        self._response = response
        self.text = response.text if hasattr(response, 'text') else ""
    
    def __iter__(self):
        yield self
    
    def __bool__(self):
        return bool(self.text)


class _GenaiCompat:
    """Top-level compatibility shim so existing code using genai.GenerativeModel still works."""
    
    @staticmethod
    def GenerativeModel(model_name, generation_config=None):
        return _GeminiModelCompat(model_name, generation_config or {})
    
    @staticmethod
    def configure(api_key):
        pass  # Already configured above


genai = _GenaiCompat()

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
    # Skip debug messages to reduce noise

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

# Initialize model with lazy loading (will be loaded in __main__)
selected_device = None
transcriber = None

# Global variable to store current model info
current_model_info = {
    "name": None,
    "language": None
}

# Global cached language detector (lazy initialization)
_language_detector = None

def get_language_detector():
    """Get or create the cached language detector model."""
    global _language_detector, selected_device
    if _language_detector is None:
        _language_detector = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base",
            device=selected_device if selected_device else "cpu",
            return_timestamps=True
        )
    return _language_detector

def cleanup_language_detector():
    """Clean up the language detector when no longer needed."""
    global _language_detector
    if _language_detector is not None:
        del _language_detector
        _language_detector = None
        gc.collect()
        if selected_device == "cuda":
            torch.cuda.empty_cache()


def detect_audio_language(audio_path: str, sample_duration: int = 30) -> str:
    """
    Detect the language of the audio file by sampling a portion of it.
    
    Args:
        audio_path (str): Path to audio file
        sample_duration (int): Duration in seconds to sample for detection
        
    Returns:
        str: Detected language code ('vi' for Vietnamese, 'en' for English, etc.)
    """
    try:
        display_info("Detecting audio language...", "info")
        
        # Load a sample of the audio
        audio = AudioSegment.from_file(audio_path)
        
        # Take a sample from the middle of the audio (more likely to have speech)
        start_ms = len(audio) // 3
        end_ms = min(start_ms + (sample_duration * 1000), len(audio))
        sample_audio = audio[start_ms:end_ms]
        
        # Export sample to temporary file
        temp_sample_path = os.path.join(os.path.dirname(audio_path), "_temp_language_detect.wav")
        sample_audio.export(temp_sample_path, format="wav")
        
        try:
            # Use cached language detector
            detector = get_language_detector()
            
            # Transcribe with language detection
            result = detector(temp_sample_path, return_timestamps=True, generate_kwargs={"task": "transcribe"})
            
            # Note: We keep the detector cached for reuse, no need to delete
            
            # Analyze the transcription to detect language
            text = result.get("text", "").strip()
            
            # Simple heuristic: check for English vs Vietnamese characteristics
            # English has more ASCII, Vietnamese has more diacritics
            if text:
                ascii_count = sum(1 for c in text if ord(c) < 128)
                total_chars = len(text.replace(" ", ""))
                
                if total_chars > 0:
                    ascii_ratio = ascii_count / total_chars
                    
                    # If more than 90% ASCII and contains common English words, likely English
                    english_words = ['the', 'is', 'are', 'was', 'were', 'and', 'or', 'to', 'of', 'in', 'a', 'an']
                    text_lower = text.lower()
                    english_word_count = sum(1 for word in english_words if word in text_lower.split())
                    
                    if ascii_ratio > 0.9 or english_word_count >= 3:
                        display_info("Detected language: English", "success")
                        return "en"
            
            display_info("Detected language: Vietnamese", "success")
            return "vi"
            
        finally:
            # Clean up temporary sample file
            if os.path.exists(temp_sample_path):
                os.remove(temp_sample_path)
                
    except Exception as e:
        display_info(f"Language detection failed, defaulting to Vietnamese: {str(e)}", "warning")
        return "vi"

def load_transcriber_for_language(language: str, device: str = None, force_model: str = None) -> None:
    """
    Load the appropriate transcription model based on detected language or forced model.
    
    Args:
        language (str): Language code ('vi' or 'en')
        device (str): Device to use ('cpu' or 'cuda')
        force_model (str): Force specific model ('phowhisper', 'whisper', or None for auto)
    """
    global transcriber, current_model_info
    
    if device is None:
        device = selected_device
    
    try:
        # Determine which model to use
        if force_model == "phowhisper":
            model_name = "vinai/PhoWhisper-medium"
            display_info(f"Loading Vietnamese model (forced): {model_name}", "info")
        elif force_model == "whisper":
            model_name = "openai/whisper-medium"
            display_info(f"Loading multilingual model (forced): {model_name}", "info")
        elif language == "en":
            model_name = "openai/whisper-medium"
            display_info(f"Loading English model: {model_name}", "info")
        else:
            model_name = "vinai/PhoWhisper-medium"
            display_info(f"Loading Vietnamese model: {model_name}", "info")
        
        # Check if we need to switch models
        if current_model_info["name"] == model_name:
            display_info(f"Model {model_name} already loaded", "info")
            return
        
        # Clean up existing model
        del transcriber
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        
        # Load new model
        transcriber = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=device,
            return_timestamps=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            model_kwargs={"use_cache": True}
        )
        
        # Update current model info
        current_model_info = {
            "name": model_name,
            "language": language
        }
        
        display_info(f"Successfully loaded {model_name}", "success")
        
    except Exception as e:
        display_info(f"Error loading model for language {language}: {str(e)}", "error")
        raise

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

# Suppress HuggingFace HTTP errors (discussions disabled for some models)
warnings.filterwarnings('ignore', message='.*403 Forbidden.*')
warnings.filterwarnings('ignore', message='.*Discussions are disabled.*')
import logging
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)

def analyze_background_noise(audio: AudioSegment) -> dict:
    """
    Analyze background noise levels in audio to determine optimal chunking strategy.
    
    Args:
        audio (AudioSegment): Audio to analyze
        
    Returns:
        dict: Noise analysis results
    """
    try:
        # Convert to numpy array for analysis
        samples = np.array(audio.get_array_of_samples())
        if audio.channels == 2:
            samples = samples.reshape((-1, 2))
            # Use left channel for analysis
            samples = samples[:, 0]
        
        # Normalize samples
        samples = samples.astype(np.float32) / 32768.0
        
        # Calculate RMS (Root Mean Square) - measure of signal strength
        rms = np.sqrt(np.mean(samples**2))
        
        # Calculate noise floor by analyzing quieter segments
        # Sort samples and take bottom 20% as potential noise floor
        sorted_abs_samples = np.sort(np.abs(samples))
        noise_floor_samples = sorted_abs_samples[:int(len(sorted_abs_samples) * 0.2)]
        noise_floor = np.mean(noise_floor_samples)
        
        # Calculate signal-to-noise ratio
        signal_power = rms**2
        noise_power = noise_floor**2
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 50
        
        # Analyze spectral characteristics
        # Take a sample from middle of audio for analysis
        sample_start = len(samples) // 4
        sample_end = sample_start + min(audio.frame_rate * 5, len(samples) - sample_start)  # 5 seconds max
        sample_segment = samples[sample_start:sample_end]
        
        # Perform FFT analysis
        fft = np.fft.fft(sample_segment)
        freqs = np.fft.fftfreq(len(sample_segment), 1/audio.frame_rate)
        magnitude = np.abs(fft)
        
        # Analyze low frequency content (potential noise)
        low_freq_limit = 100  # Hz
        low_freq_mask = (freqs >= 0) & (freqs <= low_freq_limit)
        low_freq_energy = np.sum(magnitude[low_freq_mask])
        total_energy = np.sum(magnitude[freqs >= 0])
        low_freq_ratio = low_freq_energy / total_energy if total_energy > 0 else 0
        
        # Determine noise characteristics
        noise_level = 'low'
        has_excessive_noise = False
        suggested_chunk_size = 60000  # Default 60 seconds
        
        # Classification based on multiple factors
        if snr < 10:  # Very poor SNR
            noise_level = 'very_high'
            has_excessive_noise = True
            suggested_chunk_size = 30000  # 30 seconds
        elif snr < 15:  # Poor SNR
            noise_level = 'high'
            has_excessive_noise = True
            suggested_chunk_size = 30000  # 30 seconds
        elif snr < 20:  # Moderate SNR
            noise_level = 'moderate'
            has_excessive_noise = False
            suggested_chunk_size = 45000  # 45 seconds
        elif low_freq_ratio > 0.3:  # High low-frequency content (potential AC hum, rumble)
            noise_level = 'moderate'
            has_excessive_noise = True
            suggested_chunk_size = 30000  # 30 seconds
        elif audio.dBFS < -50:  # Very quiet audio (might have relative noise issues)
            noise_level = 'moderate'
            has_excessive_noise = True
            suggested_chunk_size = 30000  # 30 seconds
        
        return {
            'noise_level': noise_level,
            'has_excessive_noise': has_excessive_noise,
            'snr_db': snr,
            'noise_floor': noise_floor,
            'rms_level': rms,
            'low_freq_ratio': low_freq_ratio,
            'suggested_chunk_size': suggested_chunk_size,
            'noise_ratio': 1.0 - min(snr / 30.0, 1.0)  # Normalize SNR to 0-1 scale
        }
        
    except Exception as e:
        display_info(f"Error analyzing background noise: {e}", "warning")
        return {
            'noise_level': 'unknown',
            'has_excessive_noise': False,
            'snr_db': 20.0,
            'noise_floor': 0.01,
            'rms_level': 0.1,
            'low_freq_ratio': 0.1,
            'suggested_chunk_size': 60000,
            'noise_ratio': 0.3
        }

def analyze_audio_characteristics(audio: AudioSegment) -> dict:
    """Analyze audio characteristics to optimize chunking strategy."""
    try:
        stats = {
            'length_ms': len(audio),
            'length_seconds': len(audio) / 1000,
            'dBFS': audio.dBFS,
            'max_dBFS': audio.max_dBFS,
            'channels': audio.channels,
            'frame_rate': audio.frame_rate,
            'is_quiet': audio.dBFS < -40,
            'is_very_quiet': audio.dBFS < -60,
            'dynamic_range': audio.max_dBFS - audio.dBFS if audio.dBFS is not None else 0
        }
        
        # Enhanced noise analysis
        noise_analysis = analyze_background_noise(audio)
        stats.update(noise_analysis)
        
        # Simple silence analysis using dBFS threshold
        try:
            from pydub.silence import detect_silence
            silent_ranges = detect_silence(
                audio, 
                min_silence_len=500,  # 0.5 seconds
                silence_thresh=audio.dBFS-10  # 10dB below average
            )
            stats['silence_count'] = len(silent_ranges)
            stats['has_natural_breaks'] = len(silent_ranges) > 0
        except ImportError:
            # Fallback if silence detection not available
            stats['silence_count'] = 0
            stats['has_natural_breaks'] = False
        
        return stats
    except Exception as e:
        display_info(f"Error analyzing audio: {e}", "warning")
        return {
            'length_ms': len(audio),
            'length_seconds': len(audio) / 1000,
            'dBFS': audio.dBFS if hasattr(audio, 'dBFS') else -30,
            'max_dBFS': audio.max_dBFS if hasattr(audio, 'max_dBFS') else -20,
            'channels': audio.channels,
            'frame_rate': audio.frame_rate,
            'is_quiet': True,
            'is_very_quiet': True,
            'dynamic_range': 0,
            'silence_count': 0,
            'has_natural_breaks': False,
            'noise_level': 'unknown',
            'has_excessive_noise': False,
            'snr_db': 20.0,
            'suggested_chunk_size': 30000,
            'noise_ratio': 0.3
        }


def reduce_noise_spectral_subtraction(audio_segment, noise_factor=2.0):
    """
    Apply spectral subtraction noise reduction
    """
    try:
        # Convert to numpy array
        samples = np.array(audio_segment.get_array_of_samples())
        
        if audio_segment.channels == 2:
            samples = samples.reshape((-1, 2))
            # Process each channel separately
            processed_channels = []
            for channel in range(2):
                channel_data = samples[:, channel]
                processed_channel = _spectral_subtract_channel(channel_data, audio_segment.frame_rate, noise_factor)
                processed_channels.append(processed_channel)
            
            # Combine channels
            processed_samples = np.column_stack(processed_channels).flatten()
        else:
            processed_samples = _spectral_subtract_channel(samples, audio_segment.frame_rate, noise_factor)
        
        # Convert back to audio segment
        processed_samples = processed_samples.astype(np.int16)
        processed_audio = AudioSegment(
            processed_samples.tobytes(),
            frame_rate=audio_segment.frame_rate,
            sample_width=audio_segment.sample_width,
            channels=audio_segment.channels
        )
        
        return processed_audio
        
    except Exception as e:
        display_info(f"Error in spectral subtraction: {e}", "warning")
        return audio_segment


def _spectral_subtract_channel(samples, sample_rate, noise_factor):
    """
    Apply spectral subtraction to a single channel
    """
    try:
        # Estimate noise from the first 0.5 seconds (assuming it's relatively quiet)
        noise_duration = min(int(0.5 * sample_rate), len(samples) // 4)
        noise_sample = samples[:noise_duration]
        
        # Calculate noise spectrum
        noise_fft = np.fft.fft(noise_sample)
        noise_magnitude = np.abs(noise_fft)
        
        # Process audio in overlapping windows
        window_size = 2048
        hop_size = window_size // 2
        processed_samples = np.zeros_like(samples)
        
        for i in range(0, len(samples) - window_size, hop_size):
            window = samples[i:i + window_size]
            
            # Apply window function
            windowed = window * np.hanning(len(window))
            
            # FFT
            window_fft = np.fft.fft(windowed)
            window_magnitude = np.abs(window_fft)
            window_phase = np.angle(window_fft)
            
            # Spectral subtraction
            # Resize noise_magnitude to match window size
            if len(noise_magnitude) != len(window_magnitude):
                noise_interp = np.interp(
                    np.linspace(0, 1, len(window_magnitude)),
                    np.linspace(0, 1, len(noise_magnitude)),
                    noise_magnitude
                )
            else:
                noise_interp = noise_magnitude
            
            # Subtract noise spectrum
            clean_magnitude = window_magnitude - noise_factor * noise_interp
            
            # Ensure magnitude doesn't go negative
            clean_magnitude = np.maximum(clean_magnitude, 0.1 * window_magnitude)
            
            # Reconstruct signal
            clean_fft = clean_magnitude * np.exp(1j * window_phase)
            clean_window = np.real(np.fft.ifft(clean_fft))
            
            # Overlap-add
            processed_samples[i:i + window_size] += clean_window
        
        return processed_samples
        
    except Exception as e:
        display_info(f"Error in channel spectral subtraction: {e}", "warning")
        return samples


def apply_high_pass_filter(audio_segment, cutoff_freq=80):
    """
    Apply high-pass filter to remove low-frequency noise
    """
    try:
        # Convert to numpy array
        samples = np.array(audio_segment.get_array_of_samples())
        sample_rate = audio_segment.frame_rate
        
        if audio_segment.channels == 2:
            samples = samples.reshape((-1, 2))
            # Process each channel
            processed_channels = []
            for channel in range(2):
                channel_data = samples[:, channel]
                filtered = _apply_highpass_channel(channel_data, sample_rate, cutoff_freq)
                processed_channels.append(filtered)
            processed_samples = np.column_stack(processed_channels).flatten()
        else:
            processed_samples = _apply_highpass_channel(samples, sample_rate, cutoff_freq)
        
        # Convert back to audio segment
        processed_samples = processed_samples.astype(np.int16)
        processed_audio = AudioSegment(
            processed_samples.tobytes(),
            frame_rate=audio_segment.frame_rate,
            sample_width=audio_segment.sample_width,
            channels=audio_segment.channels
        )
        
        return processed_audio
        
    except Exception as e:
        display_info(f"Error in high-pass filter: {e}", "warning")
        return audio_segment


def _apply_highpass_channel(samples, sample_rate, cutoff_freq):
    """
    Apply high-pass filter to a single channel
    """
    try:
        # Design Butterworth high-pass filter
        nyquist = sample_rate / 2
        normalized_cutoff = cutoff_freq / nyquist
        
        # Ensure cutoff frequency is valid
        if normalized_cutoff >= 1.0:
            normalized_cutoff = 0.9
        
        b, a = signal.butter(4, normalized_cutoff, btype='high')
        
        # Apply filter
        filtered_samples = signal.filtfilt(b, a, samples.astype(float))
        
        return filtered_samples.astype(np.int16)
        
    except Exception as e:
        display_info(f"Error in channel high-pass filter: {e}", "warning")
        return samples


def reduce_noise_adaptive(audio_segment, reduction_strength=0.5):
    """
    Apply adaptive noise reduction based on audio characteristics
    """
    try:
        # Analyze audio to determine best noise reduction approach
        stats = analyze_audio_characteristics(audio_segment)
        
        processed_audio = audio_segment
        
        # Apply high-pass filter for low-frequency noise
        if stats['dBFS'] < -30:  # Quiet audio, more aggressive filtering
            processed_audio = apply_high_pass_filter(processed_audio, cutoff_freq=100)
        else:
            processed_audio = apply_high_pass_filter(processed_audio, cutoff_freq=80)
        
        # Apply spectral subtraction for general noise
        noise_factor = 1.5 + reduction_strength
        processed_audio = reduce_noise_spectral_subtraction(processed_audio, noise_factor)
        
        # Normalize volume after noise reduction
        processed_audio = normalize_audio(processed_audio)
        
        return processed_audio
        
    except Exception as e:
        display_info(f"Error in adaptive noise reduction: {e}", "warning")
        return audio_segment


def preprocess_audio_with_noise_reduction(audio_path, noise_reduction=True, reduction_strength=0.5):
    """
    Preprocess audio with optional noise reduction
    
    Args:
        audio_path: Path to audio file
        noise_reduction: Whether to apply noise reduction
        reduction_strength: Strength of noise reduction (0.0 to 1.0)
    
    Returns:
        Path to processed audio file
    """
    try:
        if not noise_reduction:
            return audio_path
        
        display_info("Applying noise reduction preprocessing...", "info")
        
        # Load audio
        audio = AudioSegment.from_file(audio_path)
        
        # Apply noise reduction
        processed_audio = reduce_noise_adaptive(audio, reduction_strength)
        
        # Save processed audio to temporary file
        processed_path = audio_path.replace('.', '_denoised.')
        processed_audio.export(processed_path, format="wav")
        
        display_info(f"Noise reduction applied, saved to: {processed_path}", "success")
        return processed_path
        
    except Exception as e:
        display_info(f"Error in noise reduction preprocessing: {e}", "warning")
        return audio_path


def split_audio(audio_input, chunk_length_ms: int = 30000) -> List[str]:
    """
    Split audio into optimized chunks with intelligent analysis.
    Automatically uses 30s chunks for noisy audio.
    Returns paths to temporary files containing the chunks.
    
    Args:
        audio_input: Either a file path (str) or AudioSegment object
        chunk_length_ms: Default chunk length in milliseconds
    """
    try:
        # Handle both AudioSegment objects and file paths
        if isinstance(audio_input, str):
            audio = AudioSegment.from_file(audio_input)
        else:
            audio = audio_input  # Already an AudioSegment

        # Normalize audio volume before splitting
        audio = normalize_audio(audio)
        temp_files = []
        
        # Analyze audio characteristics first
        audio_stats = analyze_audio_characteristics(audio)
        
        # Determine optimal chunk size based on noise analysis
        if audio_stats.get('has_excessive_noise', False):
            optimal_chunk_size = 30000  # Force 30s for noisy audio
            display_info(f"🔊 High noise detected (SNR: {audio_stats.get('snr_db', 0):.1f}dB, Level: {audio_stats.get('noise_level', 'unknown')})", "warning")
            display_info("📏 Using 30-second chunks for better noise handling", "info")
        else:
            optimal_chunk_size = audio_stats.get('suggested_chunk_size', chunk_length_ms)
            display_info(f"🔉 Normal audio quality (SNR: {audio_stats.get('snr_db', 20):.1f}dB, Level: {audio_stats.get('noise_level', 'low')})", "info")
        
        # Override with suggested chunk size
        chunk_length_ms = optimal_chunk_size
        
        display_info(f"\nAnalyzing audio: {audio_stats['length_seconds']:.1f}s, {audio_stats['dBFS']:.1f}dB avg, {audio_stats['max_dBFS']:.1f}dB peak", "info")
        
        if audio_stats['is_very_quiet']:
            display_info("Audio is very quiet, using lenient chunking criteria", "info")
        elif audio_stats['is_quiet']:
            display_info("Audio is quiet, adjusting chunking thresholds", "info")
        
        display_info(f"Splitting audio into {chunk_length_ms/1000:.0f}s chunks...", "info")
        
        # Adjust thresholds based on audio characteristics
        if audio_stats['is_very_quiet']:
            min_silence_threshold = -70  # Very lenient for quiet audio
            min_chunk_length = 3000      # Shorter minimum chunks
        elif audio_stats['is_quiet']:
            min_silence_threshold = -55  # More lenient
            min_chunk_length = 4000      # Slightly shorter
        else:
            min_silence_threshold = -40  # Standard threshold
            min_chunk_length = 5000      # Standard minimum
        
        # For noisy audio, be more aggressive with chunking
        if audio_stats.get('has_excessive_noise', False):
            min_silence_threshold = -35  # Less strict silence requirement
            min_chunk_length = 3000      # Shorter chunks to isolate noise
        
        display_info(f"Using silence threshold: {min_silence_threshold}dB, min chunk: {min_chunk_length/1000:.1f}s", "debug")
        
        # Calculate frame length for analysis (20ms frames for faster processing)
        frame_length = 20
        frames = make_chunks(audio, frame_length)
        
        # Find optimal split points
        split_points = []
        current_chunk_start = 0
        min_silence_duration = 500   # Minimum silence duration for splits
        
        # Process frames in batches for better performance
        batch_size = 100
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            batch_dBFS = [frame.dBFS for frame in batch if frame.dBFS is not None]
            
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
                        next_dBFS = [frame.dBFS for frame in next_frames if frame.dBFS is not None]
                        if next_dBFS and all(dB <= min_silence_threshold for dB in next_dBFS):
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
        
        # Ensure we have at least start and end points
        if len(split_points) < 2:
            split_points = [0, len(audio)]
        
        # Remove duplicates and sort
        split_points = sorted(list(set(split_points)))
        
        display_info(f"Found {len(split_points)-1} potential chunks from split points", "debug")
        
        # Create chunks based on split points with 30s maximum enforcement
        chunk_count = 0
        for i in range(len(split_points) - 1):
            start = split_points[i]
            end = split_points[i + 1]
            chunk = audio[start:end]
            
            chunk_duration = len(chunk) / 1000  # Duration in seconds
            display_info(f"Evaluating chunk {i}: {chunk_duration:.1f}s, dBFS={chunk.dBFS:.1f}, max_dBFS={chunk.max_dBFS:.1f}", "debug")
            
            # Enforce 30-second maximum chunk length
            if chunk_duration > 30.0:
                display_info(f"📏 Chunk {i} too long ({chunk_duration:.1f}s), splitting into 30s segments", "warning")
                # Split into 30-second sub-chunks
                for sub_start in range(0, len(chunk), 30000):  # 30s = 30000ms
                    sub_end = min(sub_start + 30000, len(chunk))
                    sub_chunk = chunk[sub_start:sub_end]
                    sub_duration = len(sub_chunk) / 1000
                    
                    # Apply validation to sub-chunk
                    if len(sub_chunk) >= min_chunk_length:
                        # More lenient audio level check
                        chunk_audio_ok = (
                            sub_chunk.dBFS > -80 or           # Very lenient dBFS threshold
                            sub_chunk.max_dBFS > -60 or       # Check peak volume
                            sub_duration >= 3.0              # Always accept if >= 3 seconds
                        )
                        
                        if chunk_audio_ok:
                            temp_file = f"temp_chunk_{chunk_count}.wav"
                            try:
                                sub_chunk.export(
                                    temp_file,
                                    format="wav",
                                    parameters=["-ac", "1", "-ar", "16000"]
                                )
                                temp_files.append(temp_file)
                                chunk_count += 1
                                display_info(f"✓ Created sub-chunk {chunk_count}: {temp_file} ({sub_duration:.1f}s)", "debug")
                            except Exception as e:
                                display_info(f"Error exporting sub-chunk: {e}", "error")
                continue
            
            # More flexible chunk validation for normal-sized chunks
            chunk_length_ok = len(chunk) >= min_chunk_length
            
            # More lenient audio level check
            chunk_audio_ok = (
                chunk.dBFS > -80 or           # Very lenient dBFS threshold
                chunk.max_dBFS > -60 or       # Check peak volume
                chunk_duration >= 5.0        # Always accept if >= 5 seconds
            )
            
            # Debug info for troubleshooting
            display_info(f"Chunk {i}: length_ok={chunk_length_ok}, audio_ok={chunk_audio_ok}, duration={chunk_duration:.1f}s", "debug")
            
            # Save chunk if it meets criteria
            if chunk_length_ok and chunk_audio_ok:
                temp_file = f"temp_chunk_{chunk_count}.wav"
                try:
                    chunk.export(
                        temp_file,
                        format="wav",
                        parameters=["-ac", "1", "-ar", "16000"]
                    )
                    temp_files.append(temp_file)
                    chunk_count += 1
                    display_info(f"✓ Created chunk {chunk_count}: {temp_file} ({chunk_duration:.1f}s)", "debug")
                except Exception as e:
                    display_info(f"Error exporting chunk {i}: {e}", "error")
        
        # If still no chunks, try even more lenient criteria
        if not temp_files:
            display_info("No chunks with strict criteria. Trying lenient validation...", "warning")
            
            for i in range(len(split_points) - 1):
                start = split_points[i]
                end = split_points[i + 1]
                chunk = audio[start:end]
                
                chunk_duration = len(chunk) / 1000
                
                # Enforce 30s maximum chunk length
                if chunk_duration > 30:
                    display_info(f"Chunk {i} too long ({chunk_duration:.1f}s), splitting to 30s max", "warning")
                    # Split long chunk into 30s segments
                    sub_chunks = []
                    for sub_start in range(0, len(chunk), 30000):  # 30s = 30000ms
                        sub_end = min(sub_start + 30000, len(chunk))
                        sub_chunk = chunk[sub_start:sub_end]
                        if len(sub_chunk) >= 2000:  # At least 2 seconds
                            sub_chunks.append(sub_chunk)
                    
                    # Export sub-chunks
                    for j, sub_chunk in enumerate(sub_chunks):
                        if sub_chunk.dBFS > -100:  # Any audible content
                            temp_file = f"temp_chunk_{chunk_count}.wav"
                            try:
                                sub_chunk.export(
                                    temp_file,
                                    format="wav",
                                    parameters=["-ac", "1", "-ar", "16000"]
                                )
                                temp_files.append(temp_file)
                                chunk_count += 1
                                display_info(f"✓ Created sub-chunk {chunk_count}: {temp_file} ({len(sub_chunk)/1000:.1f}s)", "debug")
                            except Exception as e:
                                display_info(f"Error exporting sub-chunk {j}: {e}", "error")
                else:
                    # Very lenient criteria - any chunk > 2 seconds with any audible content
                    if len(chunk) >= 2000 and chunk.dBFS > -100:  # Almost any audio
                        temp_file = f"temp_chunk_{chunk_count}.wav"
                        try:
                            chunk.export(
                                temp_file,
                                format="wav",
                                parameters=["-ac", "1", "-ar", "16000"]
                            )
                            temp_files.append(temp_file)
                            chunk_count += 1
                            display_info(f"✓ Created lenient chunk {chunk_count}: {temp_file} ({chunk_duration:.1f}s)", "debug")
                        except Exception as e:
                            display_info(f"Error exporting lenient chunk {i}: {e}", "error")
        
        # Final fallback: Force 30-second chunks if algorithm fails
        if not temp_files:
            display_info("⚠️  Warning: No valid chunks created with algorithm. Using 30-second fixed chunks.", "warning")
            display_info(f"Audio stats: length={len(audio)/1000:.1f}s, dBFS={audio.dBFS:.1f}, max_dBFS={audio.max_dBFS:.1f}", "info")
            display_info("📏 Falling back to 30-second chunk strategy for reliability", "info")
            
            # Create fixed 30-second chunks
            chunk_size = 30000  # Fixed 30 seconds
            
            for start in range(0, len(audio), chunk_size):
                end = min(start + chunk_size, len(audio))
                chunk = audio[start:end]
                
                # Accept any chunk > 1 second
                if len(chunk) > 1000:
                    temp_file = f"temp_chunk_{len(temp_files)}.wav"
                    try:
                        chunk.export(
                            temp_file,
                            format="wav",
                            parameters=["-ac", "1", "-ar", "16000"]
                        )
                        temp_files.append(temp_file)
                        display_info(f"✓ Created 30s chunk: {temp_file} ({len(chunk)/1000:.1f}s)", "info")
                    except Exception as e:
                        display_info(f"Error creating 30s chunk: {e}", "error")
            
            # Final fallback if even forced chunks fail
            if not temp_files:
                display_info("All chunking methods failed. Using original file as single chunk.", "error")
                temp_file = "temp_chunk_0.wav"
                try:
                    audio.export(
                        temp_file,
                        format="wav",
                        parameters=["-ac", "1", "-ar", "16000"]
                    )
                    temp_files.append(temp_file)
                except Exception as e:
                    display_info(f"Critical error: Cannot export audio file: {e}", "error")
                    raise Exception(f"Failed to process audio file: {e}")
        
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
        # Use inference mode to disable gradient tracking (5-10% speedup)
        with torch.inference_mode():
            # Use the global transcriber instance for better efficiency
            result = transcriber(chunk_path)
            return result['text']
    except Exception as e:
        display_info(f"\nError processing chunk {chunk_path}: {str(e)}", "error")
        return ""

def transcribe_audio(audio_path: str, noise_reduction: bool = False, reduction_strength: float = 0.5) -> str:
    """Transcribe audio using optimized single-threaded processing with optional noise reduction."""
    if not os.path.exists(audio_path):
        return "File not found"
    
    try:
        # Apply noise reduction preprocessing if requested
        if noise_reduction:
            audio_path = preprocess_audio_with_noise_reduction(audio_path, noise_reduction, reduction_strength)
        
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
            # Pass audio object directly instead of reloading from file
            temp_files = split_audio(audio, chunk_length)
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
    
    # Optimized similar word calculation: use set intersection first
    # This is much faster than nested loop for large texts
    words1_set = set(words1)
    words2_set = set(words2)
    
    # Exact matches (fast O(n+m) operation)
    exact_matches = words1_set & words2_set
    similar_word_pairs = len(exact_matches)
    
    # Only do fuzzy matching if we have small word lists and few exact matches
    # This avoids the O(n*m) nested loop for large texts
    max_fuzzy_pairs = 100  # Limit fuzzy matching to avoid performance issues
    if len(words1) < 100 and len(words2) < 100 and similar_word_pairs < max_fuzzy_pairs:
        # Get words that didn't match exactly
        words1_unmatched = [w for w in words1 if w not in exact_matches]
        words2_unmatched = [w for w in words2 if w not in exact_matches]
        
        # Fuzzy match only the unmatched words (much smaller sets)
        fuzzy_threshold = 0.8
        fuzzy_matches = 0
        
        for w1 in words1_unmatched[:50]:  # Limit to first 50 words
            for w2 in words2_unmatched[:50]:
                if SequenceMatcher(None, w1, w2).ratio() >= fuzzy_threshold:
                    fuzzy_matches += 1
                    break  # Count each w1 only once
        
        similar_word_pairs += fuzzy_matches

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
            process_single_file(audio_path, file_name, output_folder, False, 0.5, False, 'auto')
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
    Uses caching to avoid repeated API calls.
    
    Returns:
        bool: True if Gemini is available and working, False otherwise
    """
    # Cache availability check for 5 minutes
    cache_duration = 300  # 5 minutes in seconds
    
    if not hasattr(check_gemini_availability, '_last_check_time'):
        check_gemini_availability._last_check_time = 0
        check_gemini_availability._cached_result = None
    
    current_time = time.time()
    
    # Return cached result if still valid
    if (current_time - check_gemini_availability._last_check_time) < cache_duration:
        if check_gemini_availability._cached_result is not None:
            return check_gemini_availability._cached_result
    
    # Perform actual check
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
        
        # Cache the successful result
        check_gemini_availability._last_check_time = current_time
        check_gemini_availability._cached_result = True
        return True
    except Exception as e:
        display_info(f"\nError checking Gemini availability: {str(e)}", "error")
        
        # Cache the failed result too (but with shorter duration)
        check_gemini_availability._last_check_time = current_time
        check_gemini_availability._cached_result = False
        return False

# Global cached Gemini model
_gemini_model = None

def get_gemini_model():
    """Get or create the cached Gemini model instance."""
    global _gemini_model
    if _gemini_model is None:
        generation_config = {
            "temperature": 0.5,
            "top_p": 0.7,
            "top_k": 40,
            "max_output_tokens": 128000,
        }
        
        _gemini_model = genai.GenerativeModel(
            model_name='gemini-1.5-flash',
            generation_config=generation_config
        )
    return _gemini_model

def process_transcript_with_ollama(transcript_text: str) -> str:
    """
    Process transcript using local Ollama instance (Gemma 3).
    """
    try:
        # Load prompt template (reusing logic from process_transcript_with_gemini)
        prompt_path = "system_prompt.txt"
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt_template = f.read()
        else:
            # Fallback inline prompt if file missing
            prompt_template = "Bạn là một trợ lý AI giúp sửa lỗi bản ghi âm tiếng Việt. Hãy sửa lỗi chính tả, ngữ pháp và làm văn bản mạch lạc hơn nhưng không thay đổi nội dung gốc."
        
        prompt = f"{prompt_template}\n\nBản ghi âm:\n{transcript_text}"
        
        display_info(f"Processing with Local LLM ({LOCAL_MODEL_NAME})...", "info")
        
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": LOCAL_MODEL_NAME,
                "prompt": prompt,
                "stream": False
            },
            timeout=(5, 120)  # 5s connect, 120s read
        )
        
        if response.status_code == 200:
            result = response.json()
            full_response = result.get('response', '').strip()
            if full_response:
                display_info(f"✓ {LOCAL_MODEL_NAME} processing completed successfully!", "success")
                return full_response
        
        display_info(f"Ollama returned status code {response.status_code}", "warning")
        return None
        
    except Exception as e:
        display_info(f"Ollama processing failed: {str(e)}", "warning")
        return None

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
            # Use cached Gemini model
            model = get_gemini_model()

            
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
            # Use cached Gemini model
            model = get_gemini_model()

            
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

def process_single_file(audio_path: str, audio_name: str, output_folder: str, noise_reduction: bool = False, reduction_strength: float = 0.5, skip_speed_optimization: bool = False, asr_model: str = 'auto') -> None:
    """
    Process a single audio file without any interactive input.
    Args:
        audio_path (str): Path to the audio file
        audio_name (str): Name of the audio file
        output_folder (str): Path to the output folder
        noise_reduction (bool): Whether to apply noise reduction
        reduction_strength (float): Strength of noise reduction (0.0-1.0)
        skip_speed_optimization (bool): Whether to skip speed optimization
        asr_model (str): ASR model selection ('auto', 'phowhisper', 'whisper')
    """
    is_converted, is_transcribed = check_file_status(audio_name, output_folder)
    base_name = os.path.splitext(audio_name)[0]
    transcript_file = os.path.join(output_folder, f"{base_name}.txt")
    processed_file = os.path.join(output_folder, f"{base_name}_processed.txt")

    if is_transcribed:
        display_info(f"\nFile {audio_name} has already been processed. Skipping...", "info")
        return
    elif is_converted:
        display_info(f"\nFile {audio_name} has been transcribed but not refined. Refining transcript...", "info")
        with open(transcript_file, "r", encoding="utf-8") as f:
            output_text = f.read()
        
        # Try Gemma first
        processed_text = process_transcript_with_ollama(output_text)
        
        # Fallback to Gemini if Gemma failed
        if not processed_text:
            display_info("Falling back to Gemini for transcript refinement...", "info")
            processed_text = process_transcript_with_gemini(output_text)

        if processed_text:
            with open(processed_file, "w", encoding="utf-8") as f:
                f.write(processed_text)
            display_info(f"✓ Processed output saved to: {processed_file}", "success")
        else:
            display_info("Failed to refine transcript with any model.", "error")
    else:
        display_info(f"\nProcessing new file: {audio_name}", "info")
        
        # Detect language and load appropriate model
        if asr_model == 'auto':
            detected_language = detect_audio_language(audio_path)
            load_transcriber_for_language(detected_language)
        elif asr_model == 'phowhisper':
            display_info("Using PhoWhisper model (Vietnamese only, forced by user)", "info")
            load_transcriber_for_language('vi', force_model='phowhisper')
        elif asr_model == 'whisper':
            display_info("Using Whisper model (multilingual, forced by user)", "info")
            load_transcriber_for_language('en', force_model='whisper')
        
        # Skip speed optimization if requested or if file is already a _speed file
        if skip_speed_optimization:
            display_info("Speed optimization skipped by user request.", "info")
        elif '_speed' in audio_name:
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
        output_text = transcribe_audio(audio_path, noise_reduction=noise_reduction, reduction_strength=reduction_strength)
        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write(output_text)
        display_info(f"✓ Raw transcript saved to: {transcript_file}", "success")
        
        # Transcript refinement orchestration
        processed_text = process_transcript_with_ollama(output_text)
        
        if not processed_text:
            display_info("Gemma processing failed or unavailable. Falling back to Gemini...", "warning")
            processed_text = process_transcript_with_gemini(output_text)
            
        if processed_text:
            with open(processed_file, "w", encoding="utf-8") as f:
                f.write(processed_text)
            display_info(f"✓ Processed output saved to: {processed_file}", "success")
        else:
            display_info("Failed to refine transcript with any model.", "error")

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

    # Test speeds from 1.0x to 2.25x using binary search
    min_speed = 1.0
    max_speed = 2.25
    min_similarity_threshold = 0.5  # 50% similarity needed
    
    # Extract a 15-second segment from the middle of the audio (reduced from 30s)
    middle_point = len(audio) // 2
    test_segment = audio[middle_point - 7500:middle_point + 7500]  # 15s total
    
    if len(test_segment) < 5000:  # If audio is too short
        display_info("Audio is too short for speed testing. Using default speed 1.0x.", "warning")
        return 1.0
    
    # Save the test segment
    test_segment_path = "temp_speed_test.wav"
    test_segment.export(test_segment_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
    
    try:
        # Get baseline transcript at 1.0x speed
        display_info("Getting baseline transcript at 1.0x...", "info")
        baseline_text = transcribe_audio(test_segment_path)
        if not baseline_text or baseline_text.startswith("Error:"):
            display_info("Failed to get baseline transcript. Using default speed 1.0x.", "warning")
            return 1.0
        
        # Binary search for optimal speed
        def test_speed(speed):
            """Test a specific speed and return similarity score."""
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
                    return similarity
                return 0.0
            except Exception as e:
                display_info(f"Error testing speed {speed}x: {str(e)}", "warning")
                return 0.0
            finally:
                # Clean up sped-up file
                if os.path.exists(sped_up_path):
                    os.remove(sped_up_path)
        
        # Binary search implementation
        best_speed = 1.0
        left, right = min_speed, max_speed
        
        display_info(f"Binary searching for optimal speed between {min_speed}x and {max_speed}x...", "info")
        
        # First test max speed to see if it's acceptable
        max_similarity = test_speed(max_speed)
        if max_similarity >= min_similarity_threshold:
            display_info(f"✓ Max speed {max_speed}x is acceptable (similarity: {max_similarity:.2f})", "success")
            return max_speed
        
        # Binary search for the highest acceptable speed
        iterations = 0
        max_iterations = 4  # Limit to 4 iterations max
        
        while iterations < max_iterations and right - left > 0.1:
            mid = (left + right) / 2
            mid = round(mid * 4) / 4  # Round to nearest 0.25
            
            display_info(f"Testing speed {mid}x... (iteration {iterations + 1})", "info")
            similarity = test_speed(mid)
            
            if similarity >= min_similarity_threshold:
                # This speed is acceptable, try higher
                best_speed = mid
                left = mid
                display_info(f"✓ Speed {mid}x passed (similarity: {similarity:.2f})", "success")
            else:
                # This speed is too fast, try lower
                right = mid
                display_info(f"✗ Speed {mid}x too fast (similarity: {similarity:.2f})", "warning")
            
            iterations += 1
        
        # Clean up test segment
        if os.path.exists(test_segment_path):
            os.remove(test_segment_path)
        
        if best_speed > 1.0:
            display_info(f"✓ Optimal speed found: {best_speed}x", "success")
            return best_speed
        else:
            display_info("No suitable speed increase found. Using default speed 1.0x.", "info")
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

def process_files_sequentially(files_to_process: list[str], input_folder: str, output_folder: str, noise_reduction: bool = False, reduction_strength: float = 0.5, skip_speed_optimization: bool = False, asr_model: str = 'auto') -> None:
    """
    Process multiple files sequentially for better stability and resource management.
    
    Args:
        files_to_process (list[str]): List of files to process
        input_folder (str): Input folder path
        output_folder (str): Output folder path
        noise_reduction (bool): Whether to apply noise reduction
        reduction_strength (float): Strength of noise reduction (0.0-1.0)
        skip_speed_optimization (bool): Whether to skip speed optimization
        asr_model (str): ASR model selection ('auto', 'phowhisper', 'whisper')
    """
    display_info(f"\nProcessing {len(files_to_process)} files sequentially", "info")
    
    # Process files one by one with progress tracking
    for i, file_name in enumerate(tqdm(files_to_process, desc="Processing files", unit="file")):
        try:
            audio_path = os.path.join(input_folder, file_name)
            display_info(f"\n[{i+1}/{len(files_to_process)}] Processing: {file_name}", "info")
            
            # Process single file
            process_single_file(audio_path, file_name, output_folder, noise_reduction, reduction_strength, skip_speed_optimization, asr_model)
            
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
            result = download_google_drive_file(url, output_folder)
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
    parser.add_argument('--asr-model', choices=['auto', 'phowhisper', 'whisper'], default='auto', help='ASR model selection: auto (detect language), phowhisper (Vietnamese only), whisper (multilingual)')
    parser.add_argument('--noise-reduction', action='store_true', help='Apply noise reduction preprocessing')
    parser.add_argument('--reduction-strength', type=float, default=0.5, help='Noise reduction strength (0.0-1.0, default: 0.5)')
    parser.add_argument('--skip-speed', action='store_true', help='Skip speed optimization for faster processing')
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
                    process_single_file(audio_path, audio_name, output_folder, args.noise_reduction, args.reduction_strength, args.skip_speed, args.asr_model)
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
                            process_single_file(audio_path, audio_name, output_folder, args.noise_reduction, args.reduction_strength, args.skip_speed, args.asr_model)
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
                        process_files_sequentially(files_to_process, audio_folder, output_folder, args.noise_reduction, args.reduction_strength, args.skip_speed, args.asr_model)
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
                            process_files_sequentially(wav_files, audio_folder, output_folder, args.noise_reduction, args.reduction_strength, args.skip_speed, args.asr_model)
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
                                process_files_sequentially(selected_files, audio_folder, output_folder, args.noise_reduction, args.reduction_strength, args.skip_speed, args.asr_model)
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
