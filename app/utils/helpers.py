"""
Utility helpers for the AI Fact-Checker.

Collection of reusable utility functions used across services.
"""

import re
from urllib.parse import urlparse


def clean_text(text: str) -> str:
    """
    Clean and normalize text content.

    Removes extra whitespace, special characters, and normalizes
    the text for consistent processing.

    Args:
        text: Raw text string to clean.

    Returns:
        Cleaned and normalized text string.
    """
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove zero-width characters
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    return text


def is_valid_url(url: str) -> bool:
    """
    Validate whether a string is a well-formed URL.

    Args:
        url: The URL string to validate.

    Returns:
        True if the URL is valid, False otherwise.
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def truncate_text(text: str, max_length: int = 500) -> str:
    """
    Truncate text to a maximum length, adding ellipsis if cut.

    Args:
        text: The text to truncate.
        max_length: Maximum allowed character count.

    Returns:
        Truncated text with '...' appended if it was shortened.
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def format_confidence(score: float) -> str:
    """
    Format a confidence score as a human-readable percentage.

    Args:
        score: Confidence score between 0.0 and 1.0.

    Returns:
        Formatted string like "92.5%".
    """
    return f"{score * 100:.1f}%"
