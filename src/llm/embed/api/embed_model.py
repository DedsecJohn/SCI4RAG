"""
Embedding model API - Loads configuration from config.py
"""
from langchain_community.embeddings import (
    OpenAIEmbeddings,
    DashScopeEmbeddings,
)
from src.core.config import config


def aliyun_embed_llm(modelType, apiKey):
    return DashScopeEmbeddings(
        model=modelType,
        dashscope_api_key=apiKey
    )


def openai_embed_llm(modelType, apiKey):
    return OpenAIEmbeddings(
        model=modelType,
        openai_api_key=apiKey
    )


# Dispatch map for providers
embedding_provider_map = {
    "aliyun": aliyun_embed_llm,
    "openai": openai_embed_llm,
}


def get_embed_model():
    """
    Get embedding model from config.py
    
    Raises:
        ValueError: If provider is unknown or API key is missing
    """
    cfg = config.get_embed_config()
    provider = cfg['provider']
    model = cfg['model']
    api_key = cfg['api_key']
    
    if provider not in embedding_provider_map:
        raise ValueError(
            f"Unknown embedding provider: '{provider}'. "
            f"Available providers: {list(embedding_provider_map.keys())}"
        )
    
    return embedding_provider_map[provider](
        modelType=model,
        apiKey=api_key
    )


if __name__ == "__main__":
    print(f"Embedding provider: {config.get_embed_config()['provider']}")
    print(f"Embedding model: {config.get_embed_config()['model']}")
    emb_model = get_embed_model()
    vec = emb_model.embed_query("What is electromagnetism?")
    docs = [
        "A proton has positive charge",
        "An electron has negative charge"
    ]
    vecs = emb_model.embed_documents(docs)
    print(f"Embedded {len(vecs)} documents")
