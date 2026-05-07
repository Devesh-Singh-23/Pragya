"""
Pragya – A Local RAG System for Simplified Understanding of Research Papers

Streamlit UI entry point.
"""

import os
import re
import tempfile
import time
import urllib.parse

import streamlit as st

from pragya.rag_pipeline import PragyaPipeline
from pragya.layman_layer import calculate_readability, detect_jargon
from pragya.chat_store import load_chat, save_chat, delete_chat


# =============================================================================
# Page Config
# =============================================================================
st.set_page_config(
    page_title="Pragya – Research Paper Simplified",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Custom CSS
# =============================================================================
st.markdown("""
<style>
    /* Global */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    /* Header */
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #a0aec0;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    /* Chat messages */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Cards */
    .status-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
    }
    
    /* Source chips */
    .source-chip {
        display: inline-block;
        background: rgba(102, 126, 234, 0.2);
        border: 1px solid rgba(102, 126, 234, 0.4);
        border-radius: 20px;
        padding: 0.25rem 0.75rem;
        margin: 0.2rem;
        font-size: 0.8rem;
        color: #a0b4f0;
    }
    
    /* Readability badge */
    .readability-badge {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-easy { background: rgba(72, 187, 120, 0.2); color: #68d391; border: 1px solid rgba(72, 187, 120, 0.4); }
    .badge-moderate { background: rgba(236, 201, 75, 0.2); color: #ecc94b; border: 1px solid rgba(236, 201, 75, 0.4); }
    .badge-complex { background: rgba(245, 101, 101, 0.2); color: #fc8181; border: 1px solid rgba(245, 101, 101, 0.4); }
    
    /* Jargon highlight */
    .jargon-term {
        background: rgba(236, 153, 75, 0.15);
        border: 1px solid rgba(236, 153, 75, 0.45);
        border-radius: 4px;
        padding: 0.05em 0.4em;
        color: #f6ad55;
        font-weight: 600;
        cursor: pointer;
        text-decoration: none;
        transition: all 0.2s ease;
        white-space: nowrap;
    }
    .jargon-term:hover {
        background: rgba(236, 153, 75, 0.35);
        box-shadow: 0 0 8px rgba(236, 153, 75, 0.3);
        color: #fbd38d;
        text-decoration: none;
    }
    
    /* Mode selector */
    .mode-active {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 8px;
        padding: 0.5rem;
        text-align: center;
        color: white;
        font-weight: 600;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(102, 126, 234, 0.4) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Initialize Session State
# =============================================================================
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "pipeline": None,
        "messages": [],
        "current_paper_id": None,
        "current_paper_title": None,
        "_previous_paper_id": None,   # Track paper switches
        "mode": "layman",
        "papers": [],
        "ollama_status": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


def highlight_jargon(text: str, jargon_terms: list[str]) -> str:
    """Replace jargon terms in text with clickable highlighted HTML links.
    
    Each term becomes a pill that opens a Google Scholar search in a new tab.
    """
    if not jargon_terms or not text:
        return text
    
    # Sort by length descending so longer terms get matched first
    # (e.g. "neural network" before "network")
    sorted_terms = sorted(jargon_terms, key=len, reverse=True)
    
    for term in sorted_terms:
        # Match the term only when it's NOT already inside an <a> tag
        # Uses a negative lookbehind for '>' and negative lookahead for '</a>'
        pattern = re.compile(
            r'(?<!["\w>])(' + re.escape(term) + r')(?![^<]*</a>)',
            re.IGNORECASE,
        )
        
        def _replace_keep_case(match, _term=term):
            original = match.group(1)
            url = (
                "https://www.google.com/search?q="
                + urllib.parse.quote_plus(original)
            )
            return (
                f'<a class="jargon-term" href="{url}" '
                f'target="_blank" title="Search: {original}">'
                f'{original}</a>'
            )
        
        text = pattern.sub(_replace_keep_case, text, count=1)
    
    return text


def switch_paper_chat(new_paper_id: str, new_paper_title: str):
    """Save current chat and load chat for the new paper."""
    old_id = st.session_state._previous_paper_id

    # Save the outgoing paper's chat (if any messages exist)
    if old_id and old_id != new_paper_id and st.session_state.messages:
        save_chat(
            paper_id=old_id,
            messages=st.session_state.messages,
            paper_title=st.session_state.current_paper_title or "Unknown",
        )

    # Load the incoming paper's chat
    if new_paper_id != old_id:
        st.session_state.messages = load_chat(new_paper_id)

    st.session_state.current_paper_id = new_paper_id
    st.session_state.current_paper_title = new_paper_title
    st.session_state._previous_paper_id = new_paper_id


# =============================================================================
# Pipeline Initialization
# =============================================================================
@st.cache_resource
def get_pipeline():
    """Create and cache the Pragya pipeline."""
    return PragyaPipeline()


# Load pipeline
pipeline = get_pipeline()
st.session_state.pipeline = pipeline


# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.markdown('<div class="main-header" style="font-size: 1.8rem;">🧠 Pragya</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header" style="font-size: 0.85rem;">Research Papers, Simplified</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # --- Ollama Status ---
    st.markdown("### ⚡ System Status")
    status = pipeline.check_ollama()
    
    if status["running"]:
        st.success(f"✅ Ollama connected")
        st.caption(f"Model: `{status['selected_model']}`")
    else:
        st.error("❌ Ollama not running")
        st.caption("Run `ollama serve` and pull a model:")
        st.code("ollama pull llama3:8b-instruct-q4_K_M", language="bash")
    
    st.divider()
    
    # --- PDF Upload ---
    st.markdown("### 📄 Upload Paper")
    uploaded_file = st.file_uploader(
        "Drop a research paper (PDF)",
        type=["pdf"],
        help="Upload a PDF research paper to analyze",
    )
    
    if uploaded_file is not None:
        if st.button("📥 Ingest Paper", use_container_width=True):
            # Save uploaded file to temp location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            
            # Ingest with progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(step, total, msg):
                progress_bar.progress(step / total)
                status_text.caption(msg)
            
            try:
                result = pipeline.ingest_paper(tmp_path, progress_callback=update_progress)
                st.session_state.papers = pipeline.get_papers()
                switch_paper_chat(result["paper_id"], result["title"])
                
                st.success(f"✅ **{result['title']}**")
                st.caption(
                    f"{result['num_chunks']} chunks · "
                    f"{result['num_sections']} sections · "
                    f"{result['page_count']} pages"
                )
            except Exception as e:
                import traceback
                st.error(f"Error: {str(e)}")
                st.code(traceback.format_exc(), language="python")
            finally:
                os.unlink(tmp_path)
                progress_bar.empty()
    
    st.divider()
    
    # --- Paper Selector ---
    st.markdown("### 📚 Your Papers")
    papers = pipeline.get_papers()
    st.session_state.papers = papers
    
    if papers:
        paper_options = {p["title"]: p["paper_id"] for p in papers}
        
        # Set default selection
        default_idx = 0
        if st.session_state.current_paper_id:
            ids = list(paper_options.values())
            if st.session_state.current_paper_id in ids:
                default_idx = ids.index(st.session_state.current_paper_id)
        
        selected_title = st.selectbox(
            "Select paper",
            options=list(paper_options.keys()),
            index=default_idx,
        )
        selected_paper_id = paper_options[selected_title]
        switch_paper_chat(selected_paper_id, selected_title)
        
        for p in papers:
            with st.container():
                st.caption(f"📄 {p['title'][:40]}... — {p['chunk_count']} chunks")
        
        # --- Chat Controls ---
        st.divider()
        st.markdown("### 💬 Chat")
        msg_count = len(st.session_state.messages)
        if msg_count > 0:
            st.caption(f"{msg_count} message{'s' if msg_count != 1 else ''} in this conversation")
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                delete_chat(selected_paper_id)
                st.rerun()
        else:
            st.caption("No messages yet — start chatting!")
    else:
        st.info("No papers uploaded yet. Upload a PDF to get started!")
    
    st.divider()
    
    # --- Mode Selector ---
    st.markdown("### 🎯 Response Mode")
    
    mode_descriptions = {
        "layman": ("🌟 Layman", "Simple explanations with analogies"),
        "qa": ("🔬 Research QA", "Technical, paper-level answers"),
        "summary": ("📋 Summary", "Structured paper overview"),
    }
    
    selected_mode = st.radio(
        "Choose how responses are generated:",
        options=list(mode_descriptions.keys()),
        format_func=lambda x: f"{mode_descriptions[x][0]}",
        index=list(mode_descriptions.keys()).index(st.session_state.mode),
        help="\n".join([f"**{v[0]}**: {v[1]}" for v in mode_descriptions.values()]),
    )
    st.session_state.mode = selected_mode
    st.caption(mode_descriptions[selected_mode][1])
    
    st.divider()
    
    # --- Settings ---
    with st.expander("⚙️ Advanced Settings"):
        section_filter = st.selectbox(
            "Filter by section",
            options=["All sections", "ABSTRACT", "INTRODUCTION", "METHODOLOGY", 
                     "RESULTS", "DISCUSSION", "CONCLUSION"],
            index=0,
        )
        st.session_state.section_filter = None if section_filter == "All sections" else section_filter


# =============================================================================
# Main Chat Area
# =============================================================================

# Header
st.markdown('<div class="main-header">🧠 Pragya</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask questions about research papers in plain English</div>', unsafe_allow_html=True)

# Current paper info
if st.session_state.current_paper_title:
    st.markdown(
        f'<div class="status-card">📄 Currently analyzing: '
        f'<strong>{st.session_state.current_paper_title}</strong> '
        f'&nbsp;|&nbsp; Mode: <strong>{mode_descriptions[st.session_state.mode][0]}</strong></div>',
        unsafe_allow_html=True,
    )
else:
    st.info("👈 Upload a research paper from the sidebar to begin!")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        content = msg["content"]
        
        # Highlight jargon in assistant messages
        if msg["role"] == "assistant" and msg.get("jargon"):
            content = highlight_jargon(content, msg["jargon"])
            st.markdown(content, unsafe_allow_html=True)
        else:
            st.markdown(content)
        
        # Show readability badge for assistant layman responses
        if msg["role"] == "assistant" and msg.get("readability"):
            r = msg["readability"]
            badge_class = f"badge-{r['rating'].lower()}"
            st.markdown(
                f'<span class="readability-badge {badge_class}">'
                f'{r["emoji"]} Readability: {r["rating"]} '
                f'(Flesch: {r["flesch_reading_ease"]})</span>',
                unsafe_allow_html=True,
            )
        
        # Show sources
        if msg.get("sources"):
            with st.expander(f"📚 Sources ({len(msg['sources'])} chunks)", expanded=False):
                for src in msg["sources"]:
                    st.markdown(
                        f'<span class="source-chip">'
                        f'{src["section"]} (score: {src["score"]})'
                        f'</span>',
                        unsafe_allow_html=True,
                    )
                    st.caption(src["text_preview"])

# Chat input
if prompt := st.chat_input(
    "Ask about the paper..." if st.session_state.current_paper_id else "Upload a paper first...",
    disabled=not st.session_state.current_paper_id,
):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        if not status["running"]:
            st.error("❌ Ollama is not running. Please start it with `ollama serve`.")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "❌ Ollama is not running. Please start it with `ollama serve`.",
            })
        else:
            # Get response from pipeline
            result = pipeline.query(
                question=prompt,
                paper_id=st.session_state.current_paper_id,
                mode=st.session_state.mode,
                section_filter=st.session_state.get("section_filter"),
                stream=True,
            )
            
            # Stream the response
            response_text = st.write_stream(result["response"])
            
            # Detect jargon in the response itself
            jargon_in_response = detect_jargon(response_text) if response_text else []
            # Merge with jargon from retrieved context
            all_jargon = sorted(set(jargon_in_response + (result.get("jargon") or [])))
            
            # Re-render with highlighted jargon (replaces the plain streamed text)
            if all_jargon and response_text:
                highlighted = highlight_jargon(response_text, all_jargon)
                st.markdown(highlighted, unsafe_allow_html=True)
            
            # Calculate readability for layman mode
            readability = None
            if st.session_state.mode == "layman" and response_text:
                readability = calculate_readability(response_text)
                r = readability
                badge_class = f"badge-{r['rating'].lower()}"
                st.markdown(
                    f'<span class="readability-badge {badge_class}">'
                    f'{r["emoji"]} Readability: {r["rating"]} '
                    f'(Flesch: {r["flesch_reading_ease"]})</span>',
                    unsafe_allow_html=True,
                )
            
            # Show sources
            if result["sources"]:
                with st.expander(f"📚 Sources ({len(result['sources'])} chunks)", expanded=False):
                    for src in result["sources"]:
                        st.markdown(
                            f'<span class="source-chip">'
                            f'{src["section"]} (score: {src["score"]})'
                            f'</span>',
                            unsafe_allow_html=True,
                        )
                        st.caption(src["text_preview"])
            
            # Save to history (with jargon for re-highlighting on replay)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "sources": result["sources"],
                "readability": readability,
                "jargon": all_jargon,
            })
            
            # Persist chat to disk
            save_chat(
                paper_id=st.session_state.current_paper_id,
                messages=st.session_state.messages,
                paper_title=st.session_state.current_paper_title or "Unknown",
            )
