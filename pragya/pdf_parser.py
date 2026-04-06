"""PDF Parser – Section-aware research paper extraction using PyMuPDF."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from pragya.utils import clean_text


# Common section heading patterns in research papers
SECTION_PATTERNS = [
    (r'^\s*abstract\s*$', 'ABSTRACT'),
    (r'^\s*introduction\s*$', 'INTRODUCTION'),
    (r'^\s*(?:\d+\.?\s*)?related\s+work', 'RELATED_WORK'),
    (r'^\s*(?:\d+\.?\s*)?literature\s+review', 'RELATED_WORK'),
    (r'^\s*(?:\d+\.?\s*)?background', 'BACKGROUND'),
    (r'^\s*(?:\d+\.?\s*)?method(?:ology|s)?', 'METHODOLOGY'),
    (r'^\s*(?:\d+\.?\s*)?(?:proposed\s+)?(?:approach|framework|model|system)', 'METHODOLOGY'),
    (r'^\s*(?:\d+\.?\s*)?experiment(?:al)?(?:\s+(?:setup|results|design))?', 'EXPERIMENTS'),
    (r'^\s*(?:\d+\.?\s*)?results?\s*(?:and\s+(?:discussion|analysis))?', 'RESULTS'),
    (r'^\s*(?:\d+\.?\s*)?discussion', 'DISCUSSION'),
    (r'^\s*(?:\d+\.?\s*)?evaluation', 'EVALUATION'),
    (r'^\s*(?:\d+\.?\s*)?(?:conclusion|concluding\s+remarks)s?', 'CONCLUSION'),
    (r'^\s*(?:\d+\.?\s*)?future\s+work', 'FUTURE_WORK'),
    (r'^\s*(?:\d+\.?\s*)?acknowledg(?:e)?ments?', 'ACKNOWLEDGMENTS'),
    (r'^\s*(?:\d+\.?\s*)?references?\s*$', 'REFERENCES'),
    (r'^\s*(?:\d+\.?\s*)?(?:appendix|appendices)', 'APPENDIX'),
    (r'^\s*(?:\d+\.?\s*)?supplementary', 'APPENDIX'),
]


@dataclass
class PaperSection:
    """A detected section of a research paper."""
    title: str
    content: str
    page_numbers: list = field(default_factory=list)
    section_type: str = "BODY"


@dataclass
class PaperDocument:
    """Parsed representation of a research paper."""
    title: str
    authors: str
    sections: list = field(default_factory=list)
    full_text: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def section_types(self) -> list:
        """Return unique section types in order."""
        seen = set()
        result = []
        for s in self.sections:
            if s.section_type not in seen:
                seen.add(s.section_type)
                result.append(s.section_type)
        return result


def _classify_section(heading: str) -> str:
    """Match a heading string to a known section type."""
    heading_lower = heading.lower().strip()
    for pattern, section_type in SECTION_PATTERNS:
        if re.match(pattern, heading_lower, re.IGNORECASE):
            return section_type
    return "BODY"


def _is_heading(block: dict, median_font_size: float) -> bool:
    """Determine if a text block is likely a section heading.
    
    Heuristics:
    - Font size larger than median
    - Bold font
    - Short text (< 100 chars)
    - Matches known section patterns
    """
    text = block.get("text", "").strip()
    if not text or len(text) > 120:
        return False

    font_size = block.get("size", 0)
    is_bold = block.get("flags", 0) & 2 ** 4  # Bold flag in PyMuPDF

    # Check if it matches known section patterns
    for pattern, _ in SECTION_PATTERNS:
        if re.match(pattern, text.lower(), re.IGNORECASE):
            return True

    # Larger font + short text = likely heading
    if font_size > median_font_size * 1.15 and len(text) < 80:
        return True

    # Bold + short text with numbered pattern (e.g., "3. Results")
    if is_bold and re.match(r'^\d+\.?\s+\w', text) and len(text) < 80:
        return True

    return False


def _extract_spans_info(page) -> list:
    """Extract text spans with font information from a page."""
    blocks = []
    text_dict = page.get_text("dict", sort=True)
    
    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:  # Skip non-text blocks
            continue
        for line in block.get("lines", []):
            line_text = ""
            line_size = 0
            line_flags = 0
            span_count = 0
            for span in line.get("spans", []):
                line_text += span.get("text", "")
                line_size += span.get("size", 0)
                line_flags |= span.get("flags", 0)
                span_count += 1
            
            if span_count > 0:
                avg_size = line_size / span_count
            else:
                avg_size = 0
            
            if line_text.strip():
                blocks.append({
                    "text": line_text.strip(),
                    "size": avg_size,
                    "flags": line_flags,
                    "bbox": line.get("bbox", (0, 0, 0, 0)),
                })
    return blocks


def _extract_metadata(doc: fitz.Document) -> tuple:
    """Extract title and authors from the first page of the paper."""
    if doc.page_count == 0:
        return "Untitled", "Unknown"
    
    first_page = doc[0]
    blocks = _extract_spans_info(first_page)
    
    if not blocks:
        return "Untitled", "Unknown"
    
    # Title is typically the largest font on page 1
    sizes = [b["size"] for b in blocks if b["size"] > 0]
    if not sizes:
        return "Untitled", "Unknown"
    
    max_size = max(sizes)
    
    # Collect all blocks with the largest font size as the title
    title_parts = []
    for b in blocks:
        if abs(b["size"] - max_size) < 1.0 and len(b["text"]) > 3:
            title_parts.append(b["text"])
        elif title_parts:
            break  # Stop after title blocks end
    
    title = " ".join(title_parts) if title_parts else "Untitled"
    
    # Authors: typically second-largest font, right after title
    if len(sizes) > 1:
        sorted_sizes = sorted(set(sizes), reverse=True)
        if len(sorted_sizes) >= 2:
            author_size = sorted_sizes[1]
            author_parts = []
            found_title = False
            for b in blocks:
                if abs(b["size"] - max_size) < 1.0:
                    found_title = True
                    continue
                if found_title and abs(b["size"] - author_size) < 1.0:
                    text = b["text"].strip()
                    # Authors shouldn't be too long or look like body text
                    if len(text) < 200 and not any(
                        text.lower().startswith(w) for w in 
                        ['abstract', 'introduction', 'keywords', 'the ', 'this ', 'we ', 'in ']
                    ):
                        author_parts.append(text)
                    else:
                        break
                elif found_title and author_parts:
                    break
            authors = ", ".join(author_parts) if author_parts else "Unknown"
        else:
            authors = "Unknown"
    else:
        authors = "Unknown"
    
    return title, authors


def parse_pdf(file_path: str) -> PaperDocument:
    """Parse a PDF research paper into a structured PaperDocument.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        PaperDocument with detected sections, metadata, and full text
    """
    doc = fitz.open(file_path)
    
    # Extract metadata
    title, authors = _extract_metadata(doc)
    pdf_metadata = doc.metadata or {}
    
    # Extract all text blocks with font info across all pages
    all_blocks = []
    page_texts = []
    
    for page_num in range(doc.page_count):
        page = doc[page_num]
        page_text = page.get_text("text")
        page_texts.append(page_text)
        
        blocks = _extract_spans_info(page)
        for b in blocks:
            b["page"] = page_num + 1
        all_blocks.extend(blocks)
    
    full_text = clean_text("\n".join(page_texts))
    
    # Calculate median font size for heading detection
    sizes = [b["size"] for b in all_blocks if b["size"] > 0]
    if sizes:
        sorted_sizes = sorted(sizes)
        median_size = sorted_sizes[len(sorted_sizes) // 2]
    else:
        median_size = 10.0
    
    # Detect sections by finding headings
    sections = []
    current_heading = None
    current_content = []
    current_pages = set()
    current_type = "BODY"
    
    for block in all_blocks:
        if _is_heading(block, median_size):
            # Save previous section
            if current_heading is not None or current_content:
                section_title = current_heading or "Preamble"
                section_text = clean_text("\n".join(current_content))
                if section_text.strip():
                    sections.append(PaperSection(
                        title=section_title,
                        content=section_text,
                        page_numbers=sorted(current_pages),
                        section_type=current_type,
                    ))
            
            # Start new section
            current_heading = block["text"]
            current_type = _classify_section(current_heading)
            current_content = []
            current_pages = {block["page"]}
        else:
            current_content.append(block["text"])
            current_pages.add(block["page"])
    
    # Save last section
    if current_heading is not None or current_content:
        section_title = current_heading or "Preamble"
        section_text = clean_text("\n".join(current_content))
        if section_text.strip():
            sections.append(PaperSection(
                title=section_title,
                content=section_text,
                page_numbers=sorted(current_pages),
                section_type=current_type,
            ))
    
    # If no sections detected, put everything in one section
    if not sections:
        sections.append(PaperSection(
            title="Full Document",
            content=full_text,
            page_numbers=list(range(1, doc.page_count + 1)),
            section_type="BODY",
        ))
    
    total_pages = doc.page_count
    doc.close()
    
    return PaperDocument(
        title=title,
        authors=authors,
        sections=sections,
        full_text=full_text,
        metadata={
            "file_path": str(file_path),
            "page_count": total_pages,
            "pdf_title": pdf_metadata.get("title", ""),
            "pdf_author": pdf_metadata.get("author", ""),
            "pdf_subject": pdf_metadata.get("subject", ""),
        },
    )
