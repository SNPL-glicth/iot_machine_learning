"""Semantic name generator for analysis results.

Extracts meaningful keywords from conclusion to create human-readable names.
Format: "Word1 Word2 Word3 Word4 Word5 — MMM-DD"

Example: "Incidente Crítico Infraestructura TMP-004 Enfriamiento — Mar-20"
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import List, Set

# Spanish stopwords to exclude from semantic names
SPANISH_STOPWORDS: Set[str] = {
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "de", "del", "al", "a", "en", "con", "por", "para", "sin",
    "sobre", "entre", "desde", "hasta", "hacia", "que", "y", "o",
    "pero", "si", "no", "es", "son", "está", "están", "fue", "fueron",
    "ser", "estar", "tener", "hacer", "puede", "pueden", "debe", "deben",
    "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
    "su", "sus", "mi", "mis", "tu", "tus", "nuestro", "nuestra",
    "más", "menos", "muy", "mucho", "poco", "todo", "toda", "todos",
    "como", "cuando", "donde", "porque", "cual", "cuales", "quien", "quienes",
    "ha", "han", "he", "hemos", "había", "habían", "hay",
}

# English stopwords
ENGLISH_STOPWORDS: Set[str] = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "through", "during",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "should", "could", "may", "might",
    "this", "that", "these", "those", "it", "its", "he", "she", "his", "her",
    "you", "your", "we", "our", "they", "their", "all", "some", "any", "each",
    "more", "most", "very", "can", "as", "so", "if", "when", "where", "what",
}

ALL_STOPWORDS = SPANISH_STOPWORDS | ENGLISH_STOPWORDS

# Minimum word length to consider
MIN_WORD_LENGTH = 3

# Maximum words in semantic name
MAX_WORDS = 5


def generate_semantic_name(
    conclusion: str,
    domain: str,
    analyzed_at: datetime,
    monte_carlo_confidence: Optional[float] = None,
) -> str:
    """Generate semantic name from analysis conclusion.
    
    Extracts 5 most meaningful words from conclusion, formats with date.
    Optionally appends Monte Carlo confidence score.
    
    Args:
        conclusion: Analysis conclusion text
        domain: Domain (infrastructure, security, etc.)
        analyzed_at: Analysis timestamp
        monte_carlo_confidence: Optional confidence from Monte Carlo simulation
        
    Returns:
        Semantic name in format "Word1 Word2 Word3 Word4 Word5 — MMM-DD (XX%conf)"
        
    Examples:
        >>> generate_semantic_name(
        ...     "Incidente crítico en infraestructura TMP-004 enfriamiento",
        ...     "infrastructure",
        ...     datetime(2024, 3, 20),
        ...     0.73
        ... )
        'Incidente Crítico Infraestructura TMP-004 Enfriamiento — Mar-20 (73%conf)'
    """
    if not conclusion or not isinstance(conclusion, str):
        return _default_name(domain, analyzed_at)
    
    # Extract meaningful words
    meaningful_words = _extract_meaningful_words(conclusion)
    
    if not meaningful_words:
        return _default_name(domain, analyzed_at)
    
    # Take up to MAX_WORDS
    selected_words = meaningful_words[:MAX_WORDS]
    
    # Capitalize first letter of each word
    capitalized = [w.capitalize() for w in selected_words]
    
    # Format date as "MMM-DD"
    date_str = analyzed_at.strftime("%b-%d")
    name = " ".join(capitalized) + f" — {date_str}"
    # Append Monte Carlo confidence if available
    if monte_carlo_confidence is not None and 0.0 <= monte_carlo_confidence <= 1.0:
        conf_pct = int(monte_carlo_confidence * 100)
        name += f" ({conf_pct}%conf)"
    
    # Limit total length to 200 chars (DB column constraint)
    if len(name) > 200:
        # Truncate words to fit
        words_part = " ".join(capitalized)
        max_words_len = 200 - len(f" — {date_str}") - 3  # -3 for "..."
        if len(words_part) > max_words_len:
            words_part = words_part[:max_words_len] + "..."
        name = words_part + f" — {date_str}"
    
    return name


def _extract_meaningful_words(text: str) -> List[str]:
    """Extract meaningful words from text.
    
    Args:
        text: Input text
        
    Returns:
        List of meaningful words, ordered by importance
    """
    # Normalize: lowercase
    text_lower = text.lower()
    
    # Extract words (alphanumeric + hyphens, preserving IDs like "TMP-004")
    words = re.findall(r'\b[\w\-]+\b', text_lower)
    
    # Score words by importance
    scored_words = []
    for word in words:
        # Skip stopwords
        if word in ALL_STOPWORDS:
            continue
        
        # Skip very short words (unless they look like IDs)
        if len(word) < MIN_WORD_LENGTH and not _looks_like_id(word):
            continue
        
        # Skip pure numbers
        if word.isdigit():
            continue
        
        # Compute importance score
        score = _compute_word_importance(word, text_lower)
        scored_words.append((word, score))
    
    # Sort by score (descending)
    scored_words.sort(key=lambda x: x[1], reverse=True)
    
    # Return words only (no scores)
    return [w for w, _ in scored_words]


def _compute_word_importance(word: str, full_text: str) -> float:
    """Compute importance score for a word.
    
    Higher score = more important.
    
    Args:
        word: Word to score
        full_text: Full text context
        
    Returns:
        Importance score
    """
    score = 0.0
    
    # Length bonus (longer words often more meaningful)
    score += min(len(word) / 10.0, 1.0)
    
    # ID pattern bonus (e.g., "TMP-004", "SRV-01")
    if _looks_like_id(word):
        score += 2.0
    
    # Capitalization in original (likely important)
    # Check if word appears capitalized in original text
    if re.search(r'\b' + re.escape(word.capitalize()) + r'\b', full_text):
        score += 1.0
    
    # Domain-specific keyword boost
    if word in ("crítico", "critical", "urgent", "urgente", "alert", "alerta",
                "incidente", "incident", "fallo", "failure", "error"):
        score += 1.5
    
    # Frequency penalty (very common words less important)
    frequency = full_text.count(word)
    if frequency > 3:
        score -= 0.5 * (frequency - 3)
    
    return max(0.0, score)


def _looks_like_id(word: str) -> bool:
    """Check if word looks like an identifier (e.g., TMP-004, SRV01).
    
    Args:
        word: Word to check
        
    Returns:
        True if looks like ID pattern
    """
    # Pattern: letters + hyphen/underscore + numbers
    if re.match(r'^[a-z]{2,5}[-_]\d{1,4}$', word, re.IGNORECASE):
        return True
    
    # Pattern: letters + numbers (no separator)
    if re.match(r'^[a-z]{2,5}\d{1,4}$', word, re.IGNORECASE):
        return True
    
    return False


def _default_name(domain: str, analyzed_at: datetime) -> str:
    """Generate default name when conclusion extraction fails.
    
    Args:
        domain: Domain
        analyzed_at: Analysis timestamp
        
    Returns:
        Default semantic name
    """
    date_str = analyzed_at.strftime("%b-%d")
    domain_capitalized = domain.capitalize() if domain else "General"
    return f"Análisis {domain_capitalized} — {date_str}"


def truncate_semantic_name(name: str, max_length: int = 200) -> str:
    """Truncate semantic name to maximum length.
    
    Args:
        name: Semantic name
        max_length: Maximum allowed length
        
    Returns:
        Truncated name if necessary
    """
    if len(name) <= max_length:
        return name
    
    # Try to truncate at word boundary
    truncated = name[:max_length - 3]
    last_space = truncated.rfind(' ')
    
    if last_space > 0:
        return truncated[:last_space] + "..."
    else:
        return truncated + "..."
