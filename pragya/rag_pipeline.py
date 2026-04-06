"""RAG Pipeline – End-to-end orchestrator connecting all Pragya components."""

import os
import shutil
from typing import Generator, Optional

from pragya.pdf_parser import parse_pdf, PaperDocument
from pragya.chunker import chunk_paper
from pragya.embeddings import embed_chunks, embed_query
from pragya.vector_store import store_chunks, search, list_papers, delete_paper
from pragya.llm_client import OllamaClient
from pragya.prompt_templates import format_prompt, build_context, get_section_types
from pragya.layman_layer import detect_jargon, calculate_readability
from pragya.utils import load_config, ensure_directories, generate_paper_id


class PragyaPipeline:
    """Main RAG pipeline orchestrator for Pragya."""
    
    def __init__(self):
        self.config = load_config()
        ensure_directories(self.config)
        self.llm = OllamaClient()
        self._paper_cache = {}  # paper_id -> PaperDocument
    
    def ingest_paper(
        self, 
        file_path: str, 
        progress_callback=None,
    ) -> dict:
        """Ingest a PDF research paper into the vector store.
        
        Full pipeline: parse → chunk → embed → store
        
        Args:
            file_path: Path to the PDF file
            progress_callback: Optional callable(step, total, message) for progress updates
            
        Returns:
            Dict with paper_id, title, num_chunks, sections
        """
        def _progress(step, total, msg):
            if progress_callback:
                progress_callback(step, total, msg)
        
        _progress(1, 5, "📄 Parsing PDF and detecting sections...")
        paper = parse_pdf(file_path)
        
        _progress(2, 5, f"📑 Found {len(paper.sections)} sections. Chunking...")
        chunks = chunk_paper(paper)
        
        _progress(3, 5, f"🔢 Embedding {len(chunks)} chunks...")
        chunks = embed_chunks(chunks)
        
        _progress(4, 5, "💾 Storing in vector database...")
        paper_id = generate_paper_id(file_path)
        num_stored = store_chunks(chunks, paper_id)
        
        # Save uploaded file to uploads directory
        upload_dir = self.config["paths"]["upload_dir"]
        dest_path = os.path.join(upload_dir, f"{paper_id}.pdf")
        if not os.path.exists(dest_path):
            shutil.copy2(file_path, dest_path)
        
        # Cache the parsed paper
        self._paper_cache[paper_id] = paper
        
        _progress(5, 5, "✅ Paper ingested successfully!")
        
        return {
            "paper_id": paper_id,
            "title": paper.title,
            "authors": paper.authors,
            "num_chunks": num_stored,
            "num_sections": len(paper.sections),
            "sections": paper.section_types,
            "page_count": paper.metadata.get("page_count", 0),
        }
    
    def query(
        self,
        question: str,
        paper_id: str,
        mode: str = "layman",
        section_filter: str = None,
        stream: bool = True,
    ) -> dict:
        """Query a paper with a question in the specified mode.
        
        Full RAG: embed query → retrieve → prompt → LLM → response
        
        Args:
            question: User's question
            paper_id: ID of the paper to query
            mode: "qa", "layman", or "summary"
            section_filter: Optional section type to restrict search
            stream: If True, returns a generator for streaming
            
        Returns:
            Dict with:
              - response: str or Generator (if stream=True)
              - sources: list of source chunks
              - readability: dict (for layman mode)
              - jargon: list of detected terms
        """
        # Step 1: Embed the query
        query_emb = embed_query(question)
        
        # Step 2: Retrieve relevant chunks
        retrieved = search(
            query_embedding=query_emb,
            paper_id=paper_id,
            section_filter=section_filter,
        )
        
        if not retrieved:
            return {
                "response": "I couldn't find any relevant content in this paper for your question.",
                "sources": [],
                "readability": None,
                "jargon": [],
            }
        
        # Step 3: Detect jargon in retrieved context
        context_text = " ".join([r["text"] for r in retrieved])
        jargon_terms = detect_jargon(context_text)
        
        # Step 4: Get paper title
        paper_title = retrieved[0].get("metadata", {}).get("paper_title", "Unknown")
        
        # Step 5: Build prompts
        system_prompt, user_prompt = format_prompt(
            mode=mode,
            question=question,
            retrieved_chunks=retrieved,
            paper_title=paper_title,
            jargon_terms=jargon_terms,
        )
        
        # Step 6: Generate response
        if stream:
            response_gen = self.llm.generate_stream(
                prompt=user_prompt,
                system_prompt=system_prompt,
            )
        else:
            response_text = self.llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
            )
            response_gen = response_text
        
        # Step 7: Build source info
        sources = []
        for r in retrieved:
            sources.append({
                "section": r["metadata"].get("section_title", "Unknown"),
                "section_type": r["metadata"].get("section_type", "BODY"),
                "text_preview": r["text"][:150] + "..." if len(r["text"]) > 150 else r["text"],
                "score": round(r.get("score", 0), 3),
                "pages": r["metadata"].get("page_numbers", ""),
            })
        
        return {
            "response": response_gen,
            "sources": sources,
            "jargon": jargon_terms,
            "mode": mode,
        }
    
    def get_readability(self, text: str) -> dict:
        """Calculate readability metrics for a response."""
        return calculate_readability(text)
    
    def get_papers(self) -> list[dict]:
        """List all ingested papers."""
        return list_papers()
    
    def remove_paper(self, paper_id: str) -> bool:
        """Delete a paper from the vector store."""
        if paper_id in self._paper_cache:
            del self._paper_cache[paper_id]
        
        # Remove uploaded PDF
        upload_path = os.path.join(
            self.config["paths"]["upload_dir"], f"{paper_id}.pdf"
        )
        if os.path.exists(upload_path):
            os.remove(upload_path)
        
        return delete_paper(paper_id)
    
    def check_ollama(self) -> dict:
        """Check Ollama status and available models."""
        available = self.llm.is_available()
        models = self.llm.list_models() if available else []
        return {
            "running": available,
            "models": models,
            "selected_model": self.llm.model,
            "model_available": available,
        }
