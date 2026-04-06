"""Utility functions for Pragya."""

import os
import hashlib
import yaml
from pathlib import Path


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_directories(config: dict) -> None:
    """Create necessary directories if they don't exist."""
    dirs = [
        config["vector_store"]["persist_directory"],
        config["paths"]["upload_dir"],
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def generate_paper_id(file_path: str) -> str:
    """Generate a unique ID for a paper based on file content hash."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:12]


def clean_text(text: str) -> str:
    """Clean extracted text: normalize whitespace, remove artifacts."""
    import re
    # Collapse multiple whitespace/newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    # Remove common PDF artifacts
    text = re.sub(r'-\n(\w)', r'\1', text)  # Fix hyphenated line breaks
    text = text.strip()
    return text


def truncate_text(text: str, max_chars: int = 500) -> str:
    """Truncate text to max_chars, ending at a sentence boundary."""
    if len(text) <= max_chars:
        return text
    # Find last sentence boundary before max_chars
    truncated = text[:max_chars]
    last_period = truncated.rfind('.')
    if last_period > max_chars * 0.5:
        return truncated[:last_period + 1]
    return truncated + "..."
