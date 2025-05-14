import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import sys
import os
from datetime import datetime
from phowhisper import (
    convert_youtube_url,
    download_youtube_audio_ytdlp,
    transcribe_audio,
    process_transcript_with_gemini,
    cleanup_audio_folder
)

class ResultViewer(tk.Toplevel):
    def __init__(self, parent, content, title):
        super().__init__(parent)
        self.title(title)
        self.geometry("900x700")
        
        # Create main frame
        main_frame = ttk.Frame(self, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create text widget with tags for formatting
        self.text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, width=100, height=40)
        self.text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure tags for formatting
        self.text.tag_configure("header1", font=("Arial", 16, "bold"), spacing1=10, spacing3=10)
        self.text.tag_configure("header2", font=("Arial", 14, "bold"), spacing1=8, spacing3=8)
        self.text.tag_configure("bullet", lmargin1=20, lmargin2=40)
        self.text.tag_configure("normal", font=("Arial", 10))
        
        # Configure grid weights
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Insert content with formatting
        self.insert_formatted_content(content)
        
    def insert_formatted_content(self, content):
        lines = content.split('\n')
        for line in lines:
            line = line.strip()  # Strip whitespace from both ends
            if line:  # Only process non-empty lines
                if line.startswith('##'):
                    # Header 1
                    self.text.insert(tk.END, line[2:].strip() + '\n', "header1")
                elif line.startswith('**') and line.endswith('**'):
                    # Header 2
                    self.text.insert(tk.END, line[2:-2].strip() + '\n', "header2")
                elif line.startswith('*'):
                    # Bullet point
                    self.text.insert(tk.END, line + '\n', "bullet")
                else:
                    # Normal text
                    self.text.insert(tk.END, line + '\n', "normal")
            else:
                self.text.insert(tk.END, '\n')  # Keep empty lines for spacing
        
        self.text.config(state=tk.DISABLED)  # Make text read-only

class HistoryViewer(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Processed Files History")
        self.geometry("1000x600")
        
        # Create main frame
        main_frame = ttk.Frame(self, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create listbox for files
        list_frame = ttk.LabelFrame(main_frame, text="Processed Files", padding="5")
        list_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        self.file_listbox = tk.Listbox(list_frame, width=40, height=30)
        self.file_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.file_listbox.bind('<<ListboxSelect>>', self.on_select_file)
        
        # Add scrollbar to listbox
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.file_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.file_listbox.configure(yscrollcommand=scrollbar.set)
        
        # Create preview area
        preview_frame = ttk.LabelFrame(main_frame, text="File Preview", padding="5")
        preview_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.preview_text = scrolledtext.ScrolledText(preview_frame, wrap=tk.WORD, width=80, height=30)
        self.preview_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        
        # Configure text tags for formatting
        self.configure_text_tags()
        
        # Load files
        self.load_files()
        
    def configure_text_tags(self):
        # Headings
        self.preview_text.tag_configure("h1", font=("Arial", 24, "bold"), spacing1=20, spacing3=10)
        self.preview_text.tag_configure("h2", font=("Arial", 20, "bold"), spacing1=15, spacing3=8)
        self.preview_text.tag_configure("h3", font=("Arial", 16, "bold"), spacing1=12, spacing3=6)
        self.preview_text.tag_configure("h4", font=("Arial", 14, "bold"), spacing1=10, spacing3=5)
        self.preview_text.tag_configure("h5", font=("Arial", 12, "bold"), spacing1=8, spacing3=4)
        self.preview_text.tag_configure("h6", font=("Arial", 10, "bold"), spacing1=6, spacing3=3)
        
        # Text formatting
        self.preview_text.tag_configure("bold", font=("Arial", 10, "bold"))
        self.preview_text.tag_configure("italic", font=("Arial", 10, "italic"))
        self.preview_text.tag_configure("bold_italic", font=("Arial", 10, "bold italic"))
        self.preview_text.tag_configure("strikethrough", overstrike=1)
        
        # Lists
        self.preview_text.tag_configure("bullet", lmargin1=20, lmargin2=40)
        self.preview_text.tag_configure("number", lmargin1=20, lmargin2=40)
        
        # Blockquote
        self.preview_text.tag_configure("blockquote", lmargin1=40, lmargin2=60, foreground="gray")
        
        # Code
        self.preview_text.tag_configure("code", font=("Courier", 10), background="#f0f0f0")
        self.preview_text.tag_configure("code_block", font=("Courier", 10), background="#f0f0f0", spacing1=5, spacing3=5)
        
        # Links
        self.preview_text.tag_configure("link", foreground="blue", underline=1)
        
        # Normal text
        self.preview_text.tag_configure("normal", font=("Arial", 10))
        
    def load_files(self):
        self.file_listbox.delete(0, tk.END)
        if not os.path.exists("output"):
            return
            
        files = []
        for file in os.listdir("output"):
            if file.endswith("_processed.txt"):
                file_path = os.path.join("output", file)
                # Get file modification time
                mod_time = os.path.getmtime(file_path)
                mod_time_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
                files.append((mod_time, file, file_path))
        
        # Sort files by modification time (newest first)
        files.sort(reverse=True)
        
        # Add files to listbox
        for _, file, _ in files:
            self.file_listbox.insert(tk.END, file)
            
    def on_select_file(self, event):
        selection = self.file_listbox.curselection()
        if not selection:
            return
            
        file_name = self.file_listbox.get(selection[0])
        file_path = os.path.join("output", file_name)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Clear previous content
            self.preview_text.delete(1.0, tk.END)
            
            # Process content line by line
            lines = content.split('\n')
            in_code_block = False
            code_block_content = []
            
            for line in lines:
                line = line.strip()
                
                # Handle code blocks
                if line.startswith('```'):
                    if in_code_block:
                        # End of code block
                        self.preview_text.insert(tk.END, '\n'.join(code_block_content) + '\n', "code_block")
                        code_block_content = []
                        in_code_block = False
                    else:
                        # Start of code block
                        in_code_block = True
                    continue
                
                if in_code_block:
                    code_block_content.append(line)
                    continue
                
                # Handle headings
                if line.startswith('#'):
                    level = len(line.split()[0])
                    if 1 <= level <= 6:
                        text = line[level:].strip()
                        self.preview_text.insert(tk.END, text + '\n', f"h{level}")
                        continue
                
                # Handle blockquotes
                if line.startswith('>'):
                    text = line[1:].strip()
                    self.preview_text.insert(tk.END, text + '\n', "blockquote")
                    continue
                
                # Handle lists
                if line.startswith('* ') or line.startswith('- ') or line.startswith('+ '):
                    text = line[2:].strip()
                    self.preview_text.insert(tk.END, '• ' + text + '\n', "bullet")
                    continue
                
                # Handle ordered lists - more robust checking
                if line and line[0].isdigit():
                    # Check if there's a period and space after the number
                    dot_pos = line.find('. ')
                    if dot_pos > 0 and dot_pos < 4:  # Number should be 1-3 digits
                        number = line[:dot_pos]
                        text = line[dot_pos + 2:].strip()
                        self.preview_text.insert(tk.END, f"{number}. {text}\n", "number")
                        continue
                
                # Handle inline formatting
                if line:
                    # Process bold and italic
                    line = self.process_inline_formatting(line)
                    self.preview_text.insert(tk.END, line + '\n', "normal")
                else:
                    self.preview_text.insert(tk.END, '\n')
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
            
    def process_inline_formatting(self, line):
        # Process bold and italic
        parts = []
        current_pos = 0
        
        while current_pos < len(line):
            # Find next formatting marker
            next_bold_asterisk = line.find('**', current_pos)
            next_bold_underscore = line.find('__', current_pos)
            next_italic_asterisk = line.find('*', current_pos)
            next_italic_underscore = line.find('_', current_pos)
            next_strike = line.find('~~', current_pos)
            next_code = line.find('`', current_pos)
            
            # Find the next marker
            markers = [
                (next_bold_asterisk, '**', 'bold'),
                (next_bold_underscore, '__', 'bold'),
                (next_italic_asterisk, '*', 'italic'),
                (next_italic_underscore, '_', 'italic'),
                (next_strike, '~~', 'strikethrough'),
                (next_code, '`', 'code')
            ]
            
            # Filter out -1 positions and sort by position
            markers = [(pos, marker, tag) for pos, marker, tag in markers if pos != -1]
            if not markers:
                # No more markers, add remaining text
                parts.append(('normal', line[current_pos:]))
                break
                
            next_marker = min(markers, key=lambda x: x[0])
            pos, marker, tag = next_marker
            
            # Skip if this is a bullet point marker (asterisk followed by space)
            if marker == '*' and pos + 1 < len(line) and line[pos + 1] == ' ':
                parts.append(('normal', line[current_pos:pos + 2]))
                current_pos = pos + 2
                continue
            
            # Add text before marker
            if pos > current_pos:
                parts.append(('normal', line[current_pos:pos]))
            
            # Find end marker
            end_pos = line.find(marker, pos + len(marker))
            if end_pos == -1:
                # No end marker, treat as normal text
                parts.append(('normal', line[pos:]))
                break
            
            # Check for bold-italic (*** or ___)
            if marker in ['*', '_'] and end_pos + 2 < len(line):
                if line[pos:end_pos+3] == marker * 3:
                    # This is a bold-italic marker
                    tag = 'bold_italic'
                    end_pos += 2  # Skip the extra marker
            
            # Add formatted text
            formatted_text = line[pos + len(marker):end_pos]
            parts.append((tag, formatted_text))
            current_pos = end_pos + len(marker)
        
        # Apply formatting
        for tag, text in parts:
            self.preview_text.insert(tk.END, text, tag)
        
        return ""

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
        
        # History Button
        history_button = ttk.Button(main_frame, text="View History", command=self.show_history)
        history_button.grid(row=2, column=0, columnspan=2, pady=5)
        
        # Progress Section
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="5")
        progress_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.progress_var = tk.StringVar(value="Ready")
        progress_label = ttk.Label(progress_frame, textvariable=self.progress_var)
        progress_label.grid(row=0, column=0, padx=5, pady=5)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Output Section
        output_frame = ttk.LabelFrame(main_frame, text="Output", padding="5")
        output_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.output_text = scrolledtext.ScrolledText(output_frame, height=15, width=80)
        self.output_text.grid(row=0, column=0, padx=5, pady=5)
        
        # Configure grid weights
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
    def show_history(self):
        HistoryViewer(self.root)
        
    def update_progress(self, message):
        self.progress_var.set(message)
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)
        
    def show_result(self, content, title):
        ResultViewer(self.root, content, title)
        
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
            
            # Show result in formatted window
            self.root.after(0, lambda: self.show_result(processed_text, f"Processed Result - {base_name}"))
            
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
            
            # Show result in formatted window
            self.root.after(0, lambda: self.show_result(processed_text, f"Processed Result - {base_name}"))
            
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