from src.llm.chat.response import llm_response


_ABSTRACT_SYSTEM_PROMPT = (
    "You are an expert in academic paper structure analysis.\n"
    "Classify the given text into ONE of the following categories ONLY:\n\n"
    "abstract or other\n\n"
    "Category definitions:\n"
    "abstract: The abstract of the paper, a brief standalone summary of the main findings, "
    "methods, and conclusions. Usually a single paragraph near the beginning of the paper, "
    "typically 50-300 words, without citation references like [1] or (Author, year).\n"
    "other: Anything else — including author names, affiliations, "
    "acknowledgments, references, figure captions, tables, or any non-abstract content.\n\n"
    "Rules:\n"
    "1. Return ONLY the category name in lowercase.\n"
    "2. Do NOT add explanations or extra text.\n"
    "3. If uncertain, return 'other'.\n"
)


def identify_abstract(chunk: str) -> bool:
    """
    Check if a markdown chunk is the abstract using LLM classification.

    Args:
        chunk (str): Markdown chunk text.

    Returns:
        bool: True if the chunk is identified as the abstract.
    """
    if len(chunk) < 20:
        return False
    result = llm_response(query=chunk, system_prompt=_ABSTRACT_SYSTEM_PROMPT, temperature=0.1)["content"].strip().lower()
    return result == "abstract"

