"""
Configuration loader for SCI4RAG
Loads model configurations from config.json and API keys from .env
"""
import os
import json
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv


class Config:
    """Configuration manager that combines config.json and .env"""
    
    def __init__(self, config_path: str = "config.json", env_path: str = ".env"):
        # Load environment variables
        load_dotenv(env_path)
        
        # Load configuration file
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
    
    def get_chat_config(self) -> Dict[str, Any]:
        """Get chat model configuration with API key from .env"""
        cfg = self.config['chatmodel'].copy()
        
        # Map provider to environment variable
        # Note: Aliyun and Qwen use the same API key (same provider)
        key_map = {
            'deepseek': 'DEEPSEEK_API_KEY',
            'openai': 'OPENAI_API_KEY',
            'qwen': 'QWEN_API_KEY',
            'aliyun': 'QWEN_API_KEY',  # Aliyun uses Qwen API key
        }
        
        if cfg['method'] == 'api':
            provider = cfg['provider']
            env_key = key_map.get(provider)
            if not env_key:
                raise ValueError(f"Unknown chat provider: {provider}")
            
            api_key = os.getenv(env_key)
            if not api_key:
                raise ValueError(
                    f"API key not found: {env_key} is not set in .env file. "
                    f"Please add {env_key}=your_api_key to your .env file."
                )
            
            cfg['api_key'] = api_key
        
        return cfg
    
    def get_embed_config(self) -> Dict[str, Any]:
        """Get embedding model configuration with API key from .env"""
        cfg = self.config['embedmodel'].copy()
        
        # Note: Aliyun and Qwen use the same API key (same provider)
        key_map = {
            'aliyun': 'QWEN_API_KEY',  # Aliyun uses Qwen API key
            'qwen': 'QWEN_API_KEY',
            'openai': 'OPENAI_EMBED_API_KEY',
        }
        
        if cfg['method'] == 'api':
            provider = cfg['provider']
            env_key = key_map.get(provider)
            if not env_key:
                raise ValueError(f"Unknown embed provider: {provider}")
            
            api_key = os.getenv(env_key)
            if not api_key:
                raise ValueError(
                    f"API key not found: {env_key} is not set in .env file. "
                    f"Please add {env_key}=your_api_key to your .env file."
                )
            
            cfg['api_key'] = api_key
        
        return cfg
    
    def get_parser_config(self) -> Dict[str, Any]:
        """Get parser configuration with API key from .env"""
        cfg = self.config['parser'].copy()
        
        if cfg['method'] == 'api':
            api_key = os.getenv('MINERU_API_KEY')
            if not api_key:
                raise ValueError(
                    "API key not found: MINERU_API_KEY is not set in .env file. "
                    "Please add MINERU_API_KEY=your_api_key to your .env file."
                )
            cfg['api_key'] = api_key
        
        return cfg
    
    def get_web_search_config(self) -> Dict[str, Any]:
        """
        Get web search configuration with API keys from .env
        Supports multiple providers in fallback order (e.g., ddgs -> google)
        """
        cfg = self.config['web_search'].copy()
        
        # Map provider names to environment variable keys
        key_map = {
            'google': 'GOOGLE_SEARCH_API_KEY',
            'bing': 'BING_SEARCH_API_KEY',
            # ddgs doesn't need API key
        }
        
        # Load API keys for providers that need them
        providers = cfg.get('providers', [])
        api_keys = {}
        
        for provider in providers:
            env_key = key_map.get(provider)
            if env_key:  # Only load if provider needs API key
                api_key = os.getenv(env_key)
                if api_key:
                    api_keys[provider] = api_key
        
        cfg['api_keys'] = api_keys
        return cfg


# Global configuration instance
config = Config()


if __name__ == "__main__":
    print("=== SCI4RAG Configuration ===")
    print(f"\nChat Model: {config.get_chat_config()['provider']} - {config.get_chat_config()['model']}")
    print(f"Embed Model: {config.get_embed_config()['provider']} - {config.get_embed_config()['model']}")
    print(f"Parser: {config.get_parser_config()['provider']}")
    print(f"Web Search Providers: {config.get_web_search_config()['providers']}")
    print("\nConfiguration loaded successfully!")
