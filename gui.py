import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import sys
import os
from phowhisper import (
    convert_youtube_url,
    download_youtube_audio_ytdlp,
    transcribe_audio,
    process_transcript_with_gemini,
    cleanup_audio_folder
)

class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Transcription Tool")
        self.root.geometry("800x600")
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # URL Input Section
        url_frame = ttk.LabelFrame(main_frame, text="YouTube URL", padding="5")
        url_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.url_entry = ttk.Entry(url_frame, width=70)
        self.url_entry.grid(row=0, column=0, padx=5, pady=5)
        
        url_button = ttk.Button(url_frame, text="Process URL", command=self.process_url)
        url_button.grid(row=0, column=1, padx=5, pady=5)
        
        # File Input Section
        file_frame = ttk.LabelFrame(main_frame, text="Audio File", padding="5")
        file_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.file_path = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.file_path, width=70)
        file_entry.grid(row=0, column=0, padx=5, pady=5)
        
        browse_button = ttk.Button(file_frame, text="Browse", command=self.browse_file)
        browse_button.grid(row=0, column=1, padx=5, pady=5)
        
        process_file_button = ttk.Button(file_frame, text="Process File", command=self.process_file)
        process_file_button.grid(row=0, column=2, padx=5, pady=5)
        
        # Progress Section
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="5")
        progress_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.progress_var = tk.StringVar(value="Ready")
        progress_label = ttk.Label(progress_frame, textvariable=self.progress_var)
        progress_label.grid(row=0, column=0, padx=5, pady=5)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Output Section
        output_frame = ttk.LabelFrame(main_frame, text="Output", padding="5")
        output_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.output_text = scrolledtext.ScrolledText(output_frame, height=15, width=80)
        self.output_text.grid(row=0, column=0, padx=5, pady=5)
        
        # Configure grid weights
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
    def update_progress(self, message):
        self.progress_var.set(message)
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)
        
    def process_url(self):
        url = self.url_entry.get().strip()
        if not url:
            messagebox.showerror("Error", "Please enter a YouTube URL")
            return
            
        if not ("youtube.com" in url or "youtu.be" in url):
            messagebox.showerror("Error", "Please enter a valid YouTube URL")
            return
            
        # Start processing in a separate thread
        threading.Thread(target=self._process_url_thread, args=(url,), daemon=True).start()
        
    def _process_url_thread(self, url):
        try:
            self.progress_bar.start()
            self.update_progress(f"Processing URL: {url}")
            
            # Convert URL if needed
            youtube_url = convert_youtube_url(url)
            self.update_progress(f"Converted URL: {youtube_url}")
            
            # Download audio
            self.update_progress("Downloading audio...")
            audio_path = download_youtube_audio_ytdlp(youtube_url, "audio")
            if not audio_path:
                raise Exception("Failed to download audio")
                
            # Transcribe audio
            self.update_progress("Transcribing audio...")
            output_text = transcribe_audio(audio_path)
            
            # Save raw transcript
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            transcript_file = os.path.join("output", f"{base_name}.txt")
            os.makedirs("output", exist_ok=True)
            with open(transcript_file, "w", encoding="utf-8") as f:
                f.write(output_text)
            self.update_progress(f"Raw transcript saved to: {transcript_file}")
            
            # Process with Gemini
            self.update_progress("Processing with Gemini AI...")
            processed_text = process_transcript_with_gemini(output_text)
            
            # Save processed output
            processed_file = os.path.join("output", f"{base_name}_processed.txt")
            with open(processed_file, "w", encoding="utf-8") as f:
                f.write(processed_text)
            self.update_progress(f"Processed output saved to: {processed_file}")
            
            # Cleanup
            cleanup_audio_folder("audio")
            self.update_progress("Processing completed successfully!")
            
        except Exception as e:
            self.update_progress(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))
        finally:
            self.progress_bar.stop()
            
    def browse_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.mp4"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.file_path.set(file_path)
            
    def process_file(self):
        file_path = self.file_path.get()
        if not file_path:
            messagebox.showerror("Error", "Please select an audio file")
            return
            
        # Start processing in a separate thread
        threading.Thread(target=self._process_file_thread, args=(file_path,), daemon=True).start()
        
    def _process_file_thread(self, file_path):
        try:
            self.progress_bar.start()
            self.update_progress(f"Processing file: {file_path}")
            
            # Transcribe audio
            self.update_progress("Transcribing audio...")
            output_text = transcribe_audio(file_path)
            
            # Save raw transcript
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            transcript_file = os.path.join("output", f"{base_name}.txt")
            os.makedirs("output", exist_ok=True)
            with open(transcript_file, "w", encoding="utf-8") as f:
                f.write(output_text)
            self.update_progress(f"Raw transcript saved to: {transcript_file}")
            
            # Process with Gemini
            self.update_progress("Processing with Gemini AI...")
            processed_text = process_transcript_with_gemini(output_text)
            
            # Save processed output
            processed_file = os.path.join("output", f"{base_name}_processed.txt")
            with open(processed_file, "w", encoding="utf-8") as f:
                f.write(processed_text)
            self.update_progress(f"Processed output saved to: {processed_file}")
            
            self.update_progress("Processing completed successfully!")
            
        except Exception as e:
            self.update_progress(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))
        finally:
            self.progress_bar.stop()

def main():
    root = tk.Tk()
    app = TranscriptionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 