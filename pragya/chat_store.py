"""Chat Store – Persistent per-paper chat history for Pragya.

Stores each paper's conversation as a separate JSON file so chats
are preserved across sessions and isolated per paper.
"""

import json
import os
from datetime import datetime
from typing import Optional

from pragya.utils import load_config


def _get_chat_dir() -> str:
    """Get the directory for storing chat histories."""
    config = load_config()
    chat_dir = os.path.join(
        os.path.dirname(config["paths"]["upload_dir"]),
        "chat_history",
    )
    os.makedirs(chat_dir, exist_ok=True)
    return chat_dir


def _chat_path(paper_id: str) -> str:
    """Get the file path for a paper's chat history."""
    return os.path.join(_get_chat_dir(), f"{paper_id}.json")


def load_chat(paper_id: str) -> list[dict]:
    """Load chat history for a paper.

    Args:
        paper_id: Unique paper identifier

    Returns:
        List of message dicts with 'role', 'content', and optional metadata
    """
    path = _chat_path(paper_id)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("messages", [])
        except (json.JSONDecodeError, IOError):
            return []
    return []


def save_chat(paper_id: str, messages: list[dict], paper_title: str = "Unknown") -> None:
    """Save chat history for a paper.

    Args:
        paper_id: Unique paper identifier
        messages: List of message dicts
        paper_title: Title of the paper (for metadata)
    """
    path = _chat_path(paper_id)

    # Sanitize messages for JSON serialization
    clean_messages = []
    for msg in messages:
        clean_msg = {
            "role": msg.get("role", "user"),
            "content": msg.get("content", ""),
        }
        # Preserve optional metadata
        if msg.get("readability"):
            clean_msg["readability"] = msg["readability"]
        if msg.get("sources"):
            clean_msg["sources"] = msg["sources"]
        clean_messages.append(clean_msg)

    data = {
        "paper_id": paper_id,
        "paper_title": paper_title,
        "updated_at": datetime.now().isoformat(),
        "messages": clean_messages,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def delete_chat(paper_id: str) -> bool:
    """Delete chat history for a paper.

    Args:
        paper_id: Unique paper identifier

    Returns:
        True if deleted, False if file didn't exist
    """
    path = _chat_path(paper_id)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def list_chats() -> list[dict]:
    """List all saved chat histories with metadata.

    Returns:
        List of dicts with paper_id, paper_title, message_count, updated_at
    """
    chat_dir = _get_chat_dir()
    chats = []
    for filename in os.listdir(chat_dir):
        if filename.endswith(".json"):
            path = os.path.join(chat_dir, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    chats.append({
                        "paper_id": data.get("paper_id", filename[:-5]),
                        "paper_title": data.get("paper_title", "Unknown"),
                        "message_count": len(data.get("messages", [])),
                        "updated_at": data.get("updated_at", ""),
                    })
            except (json.JSONDecodeError, IOError):
                continue
    # Sort by most recently updated
    chats.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return chats
