"""Layman Layer – Jargon detection and readability scoring for simplified explanations."""

import re
from typing import Optional


# Curated list of academic/technical jargon commonly found in research papers
JARGON_TERMS = {
    # Machine Learning / AI
    "neural network", "deep learning", "machine learning", "convolutional",
    "recurrent", "transformer", "attention mechanism", "backpropagation",
    "gradient descent", "loss function", "optimization", "hyperparameter",
    "overfitting", "underfitting", "regularization", "dropout", "batch normalization",
    "embedding", "latent space", "generative", "discriminative", "encoder",
    "decoder", "autoencoder", "GAN", "reinforcement learning", "fine-tuning",
    "transfer learning", "pre-training", "tokenization", "BERT", "GPT",
    "feature extraction", "classification", "regression", "clustering",
    
    # Statistics
    "p-value", "statistical significance", "confidence interval", "regression",
    "correlation", "variance", "standard deviation", "hypothesis testing",
    "null hypothesis", "t-test", "ANOVA", "chi-square", "bayesian",
    "posterior", "prior", "likelihood", "gaussian", "distribution",
    "stochastic", "deterministic", "Monte Carlo", "sampling",
    
    # General Science/Engineering
    "algorithm", "heuristic", "paradigm", "framework", "methodology",
    "empirical", "quantitative", "qualitative", "ablation study",
    "baseline", "benchmark", "state-of-the-art", "SOTA", "novel",
    "robust", "scalable", "inference", "convergence", "iteration",
    "epoch", "parameter", "architecture", "pipeline", "preprocessing",
    "postprocessing", "normalization", "dimensionality reduction",
    
    # Biology/Medical
    "phenotype", "genotype", "genome", "proteomics", "metabolomics",
    "in vitro", "in vivo", "pathogenesis", "biomarker", "assay",
    "polymerase chain reaction", "PCR", "antibody", "antigen",
    
    # Physics/Math
    "eigenvalue", "eigenvector", "tensor", "convex", "non-convex",
    "gradient", "Jacobian", "Hessian", "Lagrangian", "entropy",
    "differential equation", "partial derivative", "Fourier transform",
}


def detect_jargon(text: str) -> list[str]:
    """Scan text for technical jargon terms.
    
    Detects terms from:
      1. The curated JARGON_TERMS dictionary
      2. Long academic-sounding words (suffix patterns)
      3. Bold terms in LLM responses (e.g. **Term:** or **Term**)
    
    Args:
        text: Text to scan (typically retrieved context or LLM response)
        
    Returns:
        List of detected jargon terms found in the text
    """
    text_lower = text.lower()
    found = []
    
    for term in JARGON_TERMS:
        if term.lower() in text_lower:
            found.append(term)
    
    # Also detect potential jargon by pattern (words ending in -tion, -ment, etc.
    # that are unusually long and likely academic)
    words = re.findall(r'\b[a-zA-Z]{10,}\b', text)
    for word in words:
        word_lower = word.lower()
        if word_lower not in [t.lower() for t in found]:
            # Check for common academic word patterns
            if any(word_lower.endswith(suffix) for suffix in 
                   ['ization', 'isation', 'ological', 'ometric', 'odynamic']):
                found.append(word)
    
    # Extract bold terms that look like definitions (e.g. **Deep Neural Network:** or **MediaPipe:**)
    # Only match bold text followed by a colon — this filters out conversational bold like **How does it work**
    bold_terms = re.findall(r'\*\*([^*]+?)\*\*\s*:', text)
    for bt in bold_terms:
        clean = bt.strip().rstrip(':').strip()
        # Skip very short or very long matches, and pure numbers
        if 2 <= len(clean) <= 50 and not clean.isdigit():
            if clean.lower() not in [t.lower() for t in found]:
                found.append(clean)
    
    return sorted(set(found))


def calculate_readability(text: str) -> dict:
    """Calculate readability metrics for a response.
    
    Uses the textstat library for standard readability scores.
    Falls back to simple heuristics if textstat is not available.
    
    Args:
        text: Response text to evaluate
        
    Returns:
        Dict with readability metrics
    """
    try:
        import textstat
        
        flesch = textstat.flesch_reading_ease(text)
        grade = textstat.flesch_kincaid_grade(text)
        
        # Determine rating
        if flesch >= 60:
            rating = "Easy"
            emoji = "🟢"
        elif flesch >= 30:
            rating = "Moderate"
            emoji = "🟡"
        else:
            rating = "Complex"
            emoji = "🔴"
        
        return {
            "flesch_reading_ease": round(flesch, 1),
            "grade_level": round(grade, 1),
            "avg_sentence_length": round(textstat.avg_sentence_length(text), 1),
            "difficult_words": textstat.difficult_words(text),
            "rating": rating,
            "emoji": emoji,
        }
        
    except ImportError:
        # Fallback: simple heuristic-based readability
        return _simple_readability(text)


def _simple_readability(text: str) -> dict:
    """Simple readability estimation without textstat."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return {"rating": "Unknown", "emoji": "⚪"}
    
    words = text.split()
    avg_sentence_len = len(words) / len(sentences)
    
    # Count syllables (rough estimate)
    def count_syllables(word):
        word = word.lower()
        count = 0
        vowels = "aeiou"
        if word[0] in vowels:
            count += 1
        for i in range(1, len(word)):
            if word[i] in vowels and word[i - 1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        return max(count, 1)
    
    total_syllables = sum(count_syllables(w) for w in words if w.isalpha())
    avg_syllables = total_syllables / max(len(words), 1)
    
    # Simplified Flesch formula
    flesch = 206.835 - (1.015 * avg_sentence_len) - (84.6 * avg_syllables)
    
    if flesch >= 60:
        rating = "Easy"
        emoji = "🟢"
    elif flesch >= 30:
        rating = "Moderate"
        emoji = "🟡"
    else:
        rating = "Complex"
        emoji = "🔴"
    
    return {
        "flesch_reading_ease": round(flesch, 1),
        "grade_level": round((0.39 * avg_sentence_len) + (11.8 * avg_syllables) - 15.59, 1),
        "avg_sentence_length": round(avg_sentence_len, 1),
        "difficult_words": len([w for w in words if len(w) > 8]),
        "rating": rating,
        "emoji": emoji,
    }


def is_response_too_complex(text: str, min_flesch: float = None) -> bool:
    """Check if a response is too complex for layman mode.
    
    Args:
        text: Response to check
        min_flesch: Minimum acceptable Flesch score (default from config)
        
    Returns:
        True if the response is too complex
    """
    if min_flesch is None:
        from pragya.utils import load_config
        config = load_config()
        min_flesch = config["readability"]["min_flesch_score"]
    
    metrics = calculate_readability(text)
    return metrics.get("flesch_reading_ease", 0) < min_flesch
