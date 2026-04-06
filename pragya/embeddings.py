"""Embedding Pipeline – Sentence Transformers for local embedding generation."""

from typing import Optional

from sentence_transformers import SentenceTransformer

from pragya.utils import load_config


# Module-level model cache (singleton)
_model_cache: Optional[SentenceTransformer] = None


def _get_model(model_name: str = None) -> SentenceTransformer:
    """Load the embedding model (cached after first call)."""
    global _model_cache
    if _model_cache is None:
        config = load_config()
        if model_name is None:
            model_name = config["embedding"]["model"]
        _model_cache = SentenceTransformer(model_name)
    return _model_cache


def embed_text(text: str, model_name: str = None) -> list[float]:
    """Embed a single text string.
    
    Args:
        text: Text to embed
        model_name: Override model name (uses config default if None)
        
    Returns:
        Embedding vector as list of floats
    """
    model = _get_model(model_name)
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def embed_chunks(chunks: list[dict], model_name: str = None) -> list[dict]:
    """Embed a list of chunks in batch.
    
    Adds 'embedding' key to each chunk dict.
    
    Args:
        chunks: List of chunk dicts with 'text' key
        model_name: Override model name
        
    Returns:
        Same chunks list with 'embedding' added to each
    """
    config = load_config()
    batch_size = config["embedding"]["batch_size"]
    model = _get_model(model_name)
    
    texts = [c["text"] for c in chunks]
    
    # Batch encode
    all_embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 10,
        convert_to_numpy=True,
    )
    
    for chunk, embedding in zip(chunks, all_embeddings):
        chunk["embedding"] = embedding.tolist()
    
    return chunks


def embed_query(query: str, model_name: str = None) -> list[float]:
    """Embed a user query for similarity search.
    
    This is functionally the same as embed_text, but separated
    for clarity and potential future query-specific preprocessing.
    
    Args:
        query: User question text
        
    Returns:
        Embedding vector as list of floats
    """
    return embed_text(query, model_name)
