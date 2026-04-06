"""Smart Chunker – Section-aware recursive text chunking for research papers."""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from pragya.pdf_parser import PaperDocument
from pragya.utils import load_config


def chunk_paper(
    paper: PaperDocument,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> list[dict]:
    """Chunk a parsed paper into smaller pieces, preserving section metadata.
    
    Two-level chunking strategy:
    1. Level 1: Split by detected sections (Abstract, Methodology, etc.)
    2. Level 2: Within each section, use RecursiveCharacterTextSplitter
    
    Args:
        paper: Parsed PaperDocument from pdf_parser
        chunk_size: Max characters per chunk (default from config)
        chunk_overlap: Overlap between chunks (default from config)
        
    Returns:
        List of chunk dicts with keys: text, metadata
    """
    config = load_config()
    if chunk_size is None:
        chunk_size = config["chunking"]["chunk_size"]
    if chunk_overlap is None:
        chunk_overlap = config["chunking"]["chunk_overlap"]
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    
    chunks = []
    chunk_id = 0
    
    for section in paper.sections:
        if not section.content.strip():
            continue
        
        # If section is small enough, keep it as one chunk
        if len(section.content) <= chunk_size:
            chunks.append({
                "id": f"chunk_{chunk_id}",
                "text": section.content,
                "metadata": {
                    "paper_title": paper.title,
                    "authors": paper.authors,
                    "section_title": section.title,
                    "section_type": section.section_type,
                    "page_numbers": section.page_numbers,
                    "chunk_index": chunk_id,
                },
            })
            chunk_id += 1
        else:
            # Split section into smaller chunks
            sub_chunks = splitter.split_text(section.content)
            for i, sub_text in enumerate(sub_chunks):
                chunks.append({
                    "id": f"chunk_{chunk_id}",
                    "text": sub_text,
                    "metadata": {
                        "paper_title": paper.title,
                        "authors": paper.authors,
                        "section_title": section.title,
                        "section_type": section.section_type,
                        "page_numbers": section.page_numbers,
                        "chunk_index": chunk_id,
                        "sub_chunk_index": i,
                        "total_sub_chunks": len(sub_chunks),
                    },
                })
                chunk_id += 1
    
    return chunks


def get_chunks_by_section(chunks: list[dict], section_type: str) -> list[dict]:
    """Filter chunks by section type.
    
    Args:
        chunks: List of chunk dicts
        section_type: e.g. "ABSTRACT", "METHODOLOGY", "RESULTS"
        
    Returns:
        Filtered list of chunks matching the section type
    """
    return [c for c in chunks if c["metadata"].get("section_type") == section_type]
