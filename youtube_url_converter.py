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

# Example usage
if __name__ == "__main__":
    # Test with a shortened URL
    short_url = "https://youtu.be/dQw4w9WgXcQ"
    full_url = convert_youtube_url(short_url)
    print(f"Short URL: {short_url}")
    print(f"Full URL: {full_url}") 