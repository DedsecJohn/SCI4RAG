from typing import Any, Dict, Optional
from src.llm.chat.api.chat_model import get_chat_model
from langchain_core.messages import HumanMessage, SystemMessage


def llm_response(query: str,
                 system_prompt: str = "",
                 temperature: float = 1,
                 reasoning_effort: Optional[str] = None,
    ) -> Dict[str, Any]:
    """
    Call the LLM with a system prompt and user query.

    Args:
        query : str
            The user text
        system_prompt : str
            System instructions for the model.
        temperature : float
            Sampling temperature (0 = deterministic).
        reasoning_effort : Optional[str]
            When set (e.g. "high", "max"), enables thinking/reasoning mode
            for providers that support it (DeepSeek).

    Returns:
        Dict[str, Any]
            {"content": str, "reasoning_content": str | None}
    """

    llm = get_chat_model(temperature=temperature, reasoning_effort=reasoning_effort)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]
    result = llm.invoke(messages)
    content = result.content
    reasoning_content = None
    if result.additional_kwargs:
        reasoning_content = result.additional_kwargs.get("reasoning_content")
    return {"content": content, "reasoning_content": reasoning_content}