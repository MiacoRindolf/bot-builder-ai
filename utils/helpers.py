"""
Utility helper functions for the Bot Builder AI system.
Provides common utilities for UUID generation, input validation, and other helper functions.
"""

import uuid
import re
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import json
import hashlib
import base64

logger = logging.getLogger(__name__)

def generate_uuid() -> str:
    """Generate a unique identifier."""
    return str(uuid.uuid4())

def generate_employee_id(role: str) -> str:
    """Generate a unique employee ID with role prefix."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = str(uuid.uuid4())[:8]
    return f"{role}_{timestamp}_{random_suffix}"

def validate_input(text: str) -> bool:
    """
    Validate user input text.
    
    Args:
        text: Input text to validate
        
    Returns:
        True if input is valid, False otherwise
    """
    if not text or not isinstance(text, str):
        return False
    
    # Check for minimum length
    if len(text.strip()) < 3:
        return False
    
    # Check for maximum length
    if len(text) > 10000:
        return False
    
    # Check for potentially harmful content
    harmful_patterns = [
        r'<script.*?>.*?</script>',
        r'javascript:',
        r'data:text/html',
        r'vbscript:',
        r'onload=',
        r'onerror=',
        r'onclick='
    ]
    
    for pattern in harmful_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False
    
    return True

def sanitize_input(text: str) -> str:
    """
    Sanitize user input to prevent injection attacks.
    
    Args:
        text: Input text to sanitize
        
    Returns:
        Sanitized text
    """
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove potentially harmful characters
    text = re.sub(r'[<>"\']', '', text)
    
    # Limit length
    if len(text) > 10000:
        text = text[:10000]
    
    return text.strip()

def validate_email(email: str) -> bool:
    """
    Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if email is valid, False otherwise
    """
    if not email:
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_api_key(api_key: str) -> bool:
    """
    Validate API key format.
    
    Args:
        api_key: API key to validate
        
    Returns:
        True if API key format is valid, False otherwise
    """
    if not api_key:
        return False
    
    # Check for OpenAI API key format (starts with sk-)
    if api_key.startswith('sk-'):
        return len(api_key) >= 20
    
    # Check for other API key formats
    if len(api_key) >= 10:
        return True
    
    return False

def format_timestamp(timestamp: Union[datetime, str]) -> str:
    """
    Format timestamp for display.
    
    Args:
        timestamp: Timestamp to format
        
    Returns:
        Formatted timestamp string
    """
    if isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            return timestamp
    
    if isinstance(timestamp, datetime):
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    
    return str(timestamp)

def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f}h"
    else:
        days = seconds / 86400
        return f"{days:.1f}d"

def calculate_percentage(value: float, total: float) -> float:
    """
    Calculate percentage.
    
    Args:
        value: Current value
        total: Total value
        
    Returns:
        Percentage as float
    """
    if total == 0:
        return 0.0
    return (value / total) * 100

def round_to_decimal(value: float, decimal_places: int = 2) -> float:
    """
    Round value to specified decimal places.
    
    Args:
        value: Value to round
        decimal_places: Number of decimal places
        
    Returns:
        Rounded value
    """
    return round(value, decimal_places)

def format_currency(amount: float, currency: str = "USD") -> str:
    """
    Format amount as currency.
    
    Args:
        amount: Amount to format
        currency: Currency code
        
    Returns:
        Formatted currency string
    """
    currency_symbols = {
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "JPY": "¥"
    }
    
    symbol = currency_symbols.get(currency, currency)
    return f"{symbol}{amount:,.2f}"

def format_percentage(value: float, decimal_places: int = 2) -> str:
    """
    Format value as percentage.
    
    Args:
        value: Value to format
        decimal_places: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value:.{decimal_places}f}%"

def safe_json_loads(data: str) -> Optional[Any]:
    """
    Safely load JSON data.
    
    Args:
        data: JSON string to parse
        
    Returns:
        Parsed JSON data or None if error
    """
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return None

def safe_json_dumps(data: Any) -> str:
    """
    Safely dump data to JSON string.
    
    Args:
        data: Data to serialize
        
    Returns:
        JSON string
    """
    try:
        return json.dumps(data, default=str)
    except (TypeError, ValueError):
        return "{}"

def hash_string(text: str) -> str:
    """
    Create SHA-256 hash of string.
    
    Args:
        text: Text to hash
        
    Returns:
        Hash string
    """
    return hashlib.sha256(text.encode()).hexdigest()

def encode_base64(data: str) -> str:
    """
    Encode string to base64.
    
    Args:
        data: String to encode
        
    Returns:
        Base64 encoded string
    """
    return base64.b64encode(data.encode()).decode()

def decode_base64(data: str) -> str:
    """
    Decode base64 string.
    
    Args:
        data: Base64 encoded string
        
    Returns:
        Decoded string
    """
    try:
        return base64.b64decode(data.encode()).decode()
    except Exception:
        return ""

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text.
    
    Args:
        text: Text to extract keywords from
        max_keywords: Maximum number of keywords to extract
        
    Returns:
        List of keywords
    """
    if not text:
        return []
    
    # Remove special characters and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    
    # Split into words
    words = text.split()
    
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
    }
    
    # Filter out stop words and short words
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Count frequency
    word_count = {}
    for word in keywords:
        word_count[word] = word_count.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    sorted_keywords = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    return [word for word, count in sorted_keywords[:max_keywords]]

def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts using Jaccard similarity.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0
    
    # Extract keywords
    keywords1 = set(extract_keywords(text1))
    keywords2 = set(extract_keywords(text2))
    
    if not keywords1 and not keywords2:
        return 1.0
    
    # Calculate Jaccard similarity
    intersection = len(keywords1.intersection(keywords2))
    union = len(keywords1.union(keywords2))
    
    return intersection / union if union > 0 else 0.0

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        
        if start >= len(text):
            break
    
    return chunks

def normalize_text(text: str) -> str:
    """
    Normalize text by removing extra whitespace and normalizing line breaks.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize line breaks
    text = re.sub(r'\n+', '\n', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def extract_numbers(text: str) -> List[float]:
    """
    Extract numbers from text.
    
    Args:
        text: Text to extract numbers from
        
    Returns:
        List of numbers found in text
    """
    if not text:
        return []
    
    # Find all numbers (including decimals)
    numbers = re.findall(r'-?\d+\.?\d*', text)
    
    # Convert to float
    result = []
    for num in numbers:
        try:
            result.append(float(num))
        except ValueError:
            continue
    
    return result

def extract_dates(text: str) -> List[str]:
    """
    Extract dates from text.
    
    Args:
        text: Text to extract dates from
        
    Returns:
        List of date strings found in text
    """
    if not text:
        return []
    
    # Common date patterns
    patterns = [
        r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
        r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
        r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
        r'\d{1,2}/\d{1,2}/\d{2,4}',  # M/D/YY or M/D/YYYY
    ]
    
    dates = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        dates.extend(matches)
    
    return list(set(dates))  # Remove duplicates

def is_valid_url(url: str) -> bool:
    """
    Check if string is a valid URL.
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid URL, False otherwise
    """
    if not url:
        return False
    
    pattern = r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?$'
    return bool(re.match(pattern, url))

def generate_random_string(length: int = 8) -> str:
    """
    Generate a random string of specified length.
    
    Args:
        length: Length of string to generate
        
    Returns:
        Random string
    """
    import random
    import string
    
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

def retry_function(func, max_retries: int = 3, delay: float = 1.0):
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        delay: Initial delay between retries
        
    Returns:
        Function result
        
    Raises:
        Exception: If all retries fail
    """
    import time
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries:
                raise e
            
            # Exponential backoff
            sleep_time = delay * (2 ** attempt)
            time.sleep(sleep_time)
            logger.warning(f"Retry attempt {attempt + 1} for function {func.__name__}")

def memoize(func):
    """
    Simple memoization decorator.
    
    Args:
        func: Function to memoize
        
    Returns:
        Memoized function
    """
    cache = {}
    
    def wrapper(*args, **kwargs):
        key = str(args) + str(sorted(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    
    return wrapper

def time_function(func):
    """
    Decorator to time function execution.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function that logs execution time
    """
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Function {func.__name__} took {duration:.2f} seconds")
        
        return result
    
    return wrapper

def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of validation errors
    """
    errors = []
    
    required_fields = ['openai_api_key', 'database_url']
    
    for field in required_fields:
        if field not in config or not config[field]:
            errors.append(f"Missing required field: {field}")
    
    if 'max_ai_employees' in config:
        try:
            max_employees = int(config['max_ai_employees'])
            if max_employees <= 0:
                errors.append("max_ai_employees must be positive")
        except (ValueError, TypeError):
            errors.append("max_ai_employees must be an integer")
    
    return errors

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override with
        
    Returns:
        Merged configuration
    """
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result 