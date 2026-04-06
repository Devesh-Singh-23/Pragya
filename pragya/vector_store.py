"""Vector Store – ChromaDB wrapper for storing and retrieving paper embeddings."""

from typing import Optional

import chromadb

from pragya.utils import load_config


# Module-level ChromaDB client cache
_client_cache: Optional[chromadb.PersistentClient] = None


def _get_client() -> chromadb.PersistentClient:
    """Get or create the ChromaDB persistent client."""
    global _client_cache
    if _client_cache is None:
        config = load_config()
        persist_dir = config["vector_store"]["persist_directory"]
        _client_cache = chromadb.PersistentClient(path=persist_dir)
    return _client_cache


def get_or_create_collection(paper_id: str) -> chromadb.Collection:
    """Get or create a ChromaDB collection for a paper.
    
    Args:
        paper_id: Unique identifier for the paper
        
    Returns:
        ChromaDB Collection object
    """
    client = _get_client()
    collection = client.get_or_create_collection(
        name=f"paper_{paper_id}",
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def store_chunks(chunks: list[dict], paper_id: str) -> int:
    """Store embedded chunks in ChromaDB.
    
    Args:
        chunks: List of chunk dicts with 'id', 'text', 'embedding', 'metadata'
        paper_id: Paper identifier for collection naming
        
    Returns:
        Number of chunks stored
    """
    collection = get_or_create_collection(paper_id)
    
    ids = [c["id"] for c in chunks]
    embeddings = [c["embedding"] for c in chunks]
    documents = [c["text"] for c in chunks]
    
    # ChromaDB metadata must be flat (no nested dicts/lists)
    metadatas = []
    for c in chunks:
        meta = {}
        for k, v in c["metadata"].items():
            if isinstance(v, list):
                meta[k] = str(v)  # Convert lists to string
            elif isinstance(v, (str, int, float, bool)):
                meta[k] = v
            else:
                meta[k] = str(v)
        metadatas.append(meta)
    
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    
    return len(chunks)


def search(
    query_embedding: list[float],
    paper_id: str,
    top_k: int = None,
    section_filter: str = None,
) -> list[dict]:
    """Search for similar chunks in a paper's collection.
    
    Args:
        query_embedding: Query embedding vector
        paper_id: Paper identifier
        top_k: Number of results to return
        section_filter: Optional section type to filter by
        
    Returns:
        List of result dicts with 'text', 'metadata', 'score'
    """
    config = load_config()
    if top_k is None:
        top_k = config["retrieval"]["top_k"]
    
    collection = get_or_create_collection(paper_id)
    
    # Build where filter for section type
    where_filter = None
    if section_filter:
        where_filter = {"section_type": section_filter}
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )
    
    # Format results
    output = []
    if results["documents"] and results["documents"][0]:
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            output.append({
                "text": doc,
                "metadata": meta,
                "score": 1 - dist,  # Convert distance to similarity score
                "rank": i + 1,
            })
    
    return output


def list_papers() -> list[dict]:
    """List all ingested papers.
    
    Returns:
        List of dicts with paper_id, collection name, and chunk count
    """
    client = _get_client()
    collections = client.list_collections()
    
    papers = []
    for col in collections:
        if col.name.startswith("paper_"):
            paper_id = col.name.replace("paper_", "")
            # Get a sample metadata to extract paper title
            sample = col.peek(limit=1)
            title = "Unknown"
            if sample["metadatas"]:
                title = sample["metadatas"][0].get("paper_title", "Unknown")
            
            papers.append({
                "paper_id": paper_id,
                "title": title,
                "chunk_count": col.count(),
            })
    
    return papers


def delete_paper(paper_id: str) -> bool:
    """Delete a paper's collection from ChromaDB.
    
    Args:
        paper_id: Paper identifier
        
    Returns:
        True if deleted successfully
    """
    client = _get_client()
    try:
        client.delete_collection(f"paper_{paper_id}")
        return True
    except Exception:
        return False
