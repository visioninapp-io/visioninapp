"""
Content-Type detection based on file extensions.
"""
from pathlib import Path
from typing import Optional

def detect_content_type(filename: str) -> str:
    """
    Auto-detect Content-Type based on file extension.
    
    Args:
        filename: Filename (e.g., "video.mp4", "image.jpg").
        
    Returns:
        MIME type (e.g., "video/mp4", "image/jpeg").
    """
    file_ext = Path(filename).suffix.lower()
    
    # Image files
    image_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.tiff': 'image/tiff',
        '.tif': 'image/tiff',
        '.webp': 'image/webp'
    }
    
    # Video files
    video_types = {
        '.mp4': 'video/mp4',
        '.avi': 'video/x-msvideo',
        '.mov': 'video/quicktime',
        '.mkv': 'video/x-matroska',
        '.wmv': 'video/x-ms-wmv',
        '.flv': 'video/x-flv',
        '.webm': 'video/webm',
        '.m4v': 'video/x-m4v'
    }
    
    # Audio files (additional support)
    audio_types = {
        '.mp3': 'audio/mpeg',
        '.wav': 'audio/wav',
        '.aac': 'audio/aac',
        '.ogg': 'audio/ogg',
        '.flac': 'audio/flac'
    }
    
    # Document files (additional support)
    document_types = {
        '.pdf': 'application/pdf',
        '.doc': 'application/msword',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.xls': 'application/vnd.ms-excel',
        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        '.ppt': 'application/vnd.ms-powerpoint',
        '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        '.txt': 'text/plain',
        '.csv': 'text/csv'
    }
    
    # Merge all types
    all_types = {**image_types, **video_types, **audio_types, **document_types}
    
    return all_types.get(file_ext, 'application/octet-stream')

def is_image_file(filename: str) -> bool:
    """Return True if the file is an image."""
    content_type = detect_content_type(filename)
    return content_type.startswith('image/')

def is_video_file(filename: str) -> bool:
    """Return True if the file is a video."""
    content_type = detect_content_type(filename)
    return content_type.startswith('video/')

def is_audio_file(filename: str) -> bool:
    """Return True if the file is an audio."""
    content_type = detect_content_type(filename)
    return content_type.startswith('audio/')

def get_file_category(filename: str) -> str:
    """Return file category (image, video, audio, document, other)."""
    content_type = detect_content_type(filename)
    
    if content_type.startswith('image/'):
        return 'image'
    elif content_type.startswith('video/'):
        return 'video'
    elif content_type.startswith('audio/'):
        return 'audio'
    elif content_type.startswith('text/') or content_type.startswith('application/'):
        return 'document'
    else:
        return 'other'
