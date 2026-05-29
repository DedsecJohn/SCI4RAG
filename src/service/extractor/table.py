import re


def identify_table(chunk: str, next_chunk: str = None) -> bool:
    """
    Check if a markdown chunk is a table using rule-based patterns.

    Args:
        chunk (str): Markdown chunk text.
        next_chunk (str): Next chunk content for positional context.

    Returns:
        bool: True if the chunk is identified as a table.
    """
    # HTML table
    if '<table>' in chunk.lower():
        return True

    # Table caption: starts with "Table N" and followed by an actual <table> chunk
    if next_chunk and '<table>' in next_chunk.lower():
        if re.match(r'(?i)^\s*table\s+\d+[\s\.:]', chunk):
            return True

    return False
