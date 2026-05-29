"""
Chat model API - Loads configuration from config.py
"""
from typing import Optional
from warnings import warn
from langchain_openai import ChatOpenAI
from src.core.config import config


def aliyun_chat_llm(modelType, apiKey, temperature=1.0, max_tokens=4096, reasoning_effort=None):
    return ChatOpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=apiKey,
        model=modelType,
        temperature=temperature,
        max_tokens=max_tokens,
        request_timeout=120,
        max_retries=3
    )


def deepseek_chat_llm(modelType, apiKey, temperature=0.7, max_tokens=4096, reasoning_effort=None):
    model_kwargs = {}
    if reasoning_effort is not None:
        if reasoning_effort not in ("high", "max"):
            warn(
                f"reasoning_effort must be None, 'high', or 'max', "
                f"got '{reasoning_effort}'. Falling back to disabled thinking mode."
            )
        else:
            model_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            model_kwargs["reasoning_effort"] = reasoning_effort

    kwargs = dict(
        base_url="https://api.deepseek.com/v1",
        api_key=apiKey,
        model=modelType,
        temperature=temperature,
        max_tokens=max_tokens,
        request_timeout=120,
        max_retries=3,
    )
    if model_kwargs:
        kwargs["model_kwargs"] = model_kwargs
    return ChatOpenAI(**kwargs)


def openai_chat_llm(modelType, apiKey, temperature=0.7, max_tokens=4096, reasoning_effort=None):
    return ChatOpenAI(
        api_key=apiKey,
        model=modelType,
        temperature=temperature,
        max_tokens=max_tokens,
        request_timeout=120,
        max_retries=3
    )


def qwen_chat_llm(modelType, apiKey, temperature=0.7, max_tokens=4096, reasoning_effort=None):
    """Qwen uses Aliyun's API"""
    return aliyun_chat_llm(modelType, apiKey, temperature, max_tokens, reasoning_effort)


# Dispatch map for providers
chat_provider_map = {
    "aliyun": aliyun_chat_llm,
    "deepseek": deepseek_chat_llm,
    "openai": openai_chat_llm,
    "qwen": qwen_chat_llm,
}


def get_chat_model(temperature=0.7, max_tokens=4096, reasoning_effort=None):
    """
    Get chat model from config.py
    
    Args:
        reasoning_effort: Optional[str] — None (disabled), "high", or "max".
                           Other values trigger a warning and fall back to disabled.
    
    Raises:
        ValueError: If provider is unknown or API key is missing
    """
    cfg = config.get_chat_config()
    provider = cfg['provider']
    model = cfg['model']
    api_key = cfg['api_key']
    
    if provider not in chat_provider_map:
        raise ValueError(
            f"Unknown chat provider: '{provider}'. "
            f"Available providers: {list(chat_provider_map.keys())}"
        )
    
    return chat_provider_map[provider](
        modelType=model,
        apiKey=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort
    )


if __name__ == "__main__":
    print(f"Chat provider: {config.get_chat_config()['provider']}")
    print(f"Chat model: {config.get_chat_config()['model']}")
    print(get_chat_model())