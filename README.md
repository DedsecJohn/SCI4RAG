<h1 align="center">SchwarzRAG</h1>

<!-- <p align="center">
  <a href="./README.md">English</a> |
  <a href="./README_zh.md">简体中文</a>
</p> -->

<details open>
<summary><b>📕 Table of Contents</b></summary>

- 💡 [Project Overview](#-project-overview)
- 🔥 [Latest Updates](#-latest-updates)
- 🌟 [Quick Start](#-quick-start)
  - [Installation](#installation)
  - [Configuration](#configuration)
- 🎉 [Key Features](#-key-features)
- 📜 [System Architecture](#-system-architecture)
- 🔎 [Technical Roadmap](#-technical-roadmap)
- ⚙️ [Configuration Guide](#-configuration-guide)
  - [Switching Models](#switching-models)
  - [Adding New Providers](#adding-new-providers)
  - [Usage in Code](#usage-in-code)
  - [Logging](#logging)
  - [Error Handling](#error-handling)
  - [Troubleshooting](#troubleshooting)
- 🔒 [Security](#-security)
- 🙌 [Contributing](#-contributing)

</details>

## 💡 Project Overview

SchwarzRAG is an intelligent retrieval system built on deep document understanding, specializing in research evolution tracking, knowledge retrieval, and professional assistance for the Schwarz Crystal domain. As a vertical domain LLM application, it provides conversational interaction, intelligent agency, and information query capabilities.

## 🔥 Latest Updates

- 2026-03-07 Complete text vector database construction and data clean; finish the fundamental functionality of the system like knowledge retrieval, web searching, and information query.
- 2026-05-20 Refactored configuration system with hybrid `.env` + `config.json` approach for better security and flexibility.

## 🌟 Quick Start

### Installation

Need to install Python 3.10+, recommended to use Anaconda or Miniconda

```bash
# Create environment
conda create -n scirag python=3.10.19
conda activate scirag

# Install dependencies
pip install -r requirements.txt
```

### Configuration

SCI4RAG uses a hybrid configuration system:
- **`.env`** - Stores sensitive API keys (not committed to git)
- **`config.json`** - Stores model configurations (can be committed to git)

**No need to run `config.py` manually** - configurations are loaded automatically when you import models.

#### 1. Setup API Keys

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```bash
DEEPSEEK_API_KEY=your_actual_key_here
QWEN_API_KEY=your_actual_key_here
MINERU_API_KEY=your_actual_key_here
GOOGLE_SEARCH_API_KEY=your_actual_key_here  # Optional
```

**Note:** `QWEN_API_KEY` is used for both Qwen/Aliyun chat and embedding models (same provider).

#### 2. Configure Models

Edit `config.json` to select which models to use:
```json
{
  "chatmodel": {
    "method": "api",
    "provider": "deepseek",
    "model": "deepseek-chat"
  },
  "embedmodel": {
    "method": "api",
    "provider": "aliyun",
    "model": "text-embedding-v1"
  },
  "parser": {
    "method": "api",
    "provider": "mineru"
  },
  "web_search": {
    "providers": ["ddgs", "google"]
  }
}
```

#### 3. Run Your Code

That's it! Just run your code directly:
```bash
python src/pipeline/rag_pipeline.py
```

Configuration is loaded automatically when you import models.

## 🎉 Key Features

### 🍭 Data Clean and Extract

- After parsing, the files inputted by the user will be turned into markdown format, which still exist many noise. This step will help to clean the data and extract the essential information.

### 🌱 Scientific Evolution Knowledge Graph

- Construct Citation-Aware Scientific Knowledge Graph
- Based citation traceability with scientific letter, combine the knowledge graph and the citation graph.

### 🍔 Multi-Retrieve Strategy

- Support multiple retrieval strategies with hybrid fallback
- Analyze the relevance of the retrieved information
- Web search: DuckDuckGo (free) → Google (API) → Bing (API)

## 📜 System Architecture

![alt text](Figure/Architecture.png)

## 🔎 Technical Roadmap

### 📝 Data Parse and Clean (Initial Step)

- Parse model: MinerU
- DOI Agent
- Identify Agent for scientific articles

![alt text](Figure/data_clean.png)

### 🚀 Localized High-Code Solution (Target Phase)

- Deployment: FastAPI + vLLM inference optimization
- Framework: LangChain agent orchestration
- Web App: Gradio/Streamlit interfaces
- Data Processing: RAGFlow Parser engine

---

## ⚙️ Configuration Guide

### Switching Models

#### Chat Model

Edit `config.json`:
```json
{
  "chatmodel": {
    "method": "api",
    "provider": "openai",     // Change provider
    "model": "gpt-4"          // Change model
  }
}
```

**Supported providers:**
- `deepseek` - DeepSeek Chat (uses `DEEPSEEK_API_KEY`)
- `openai` - OpenAI GPT (uses `OPENAI_API_KEY`)
- `qwen` - Alibaba Qwen (uses `QWEN_API_KEY`)
- `aliyun` - Alibaba Cloud (uses `QWEN_API_KEY`)

#### Embedding Model

Edit `config.json`:
```json
{
  "embedmodel": {
    "method": "api",
    "provider": "openai",     // Change provider
    "model": "text-embedding-3-small"
  }
}
```

**Supported providers:**
- `aliyun` - Alibaba Cloud Embeddings (uses `QWEN_API_KEY`)
- `qwen` - Qwen Embeddings (uses `QWEN_API_KEY`)
- `openai` - OpenAI Embeddings (uses `OPENAI_EMBED_API_KEY`)

#### Web Search

Web search uses a **hybrid fallback strategy** - tries providers in order until one succeeds.

Edit `config.json`:
```json
{
  "web_search": {
    "providers": ["ddgs", "google", "bing"]  // Try in this order
  }
}
```

**Supported providers:**
- `ddgs` - DuckDuckGo Search (free, no API key required)
- `google` - Google Search (requires `GOOGLE_SEARCH_API_KEY`)
- `bing` - Bing Search (requires `BING_SEARCH_API_KEY`)

**How it works:**
1. First tries `ddgs` (free, no API key)
2. If `ddgs` fails, tries `google` (if API key is configured)
3. If `google` fails, tries `bing` (if API key is configured)
4. Returns results from the first successful provider

### Adding New Providers

#### Example: Add Bing Search

**1. Add API key to `.env`:**
```bash
BING_SEARCH_API_KEY=your_bing_api_key
```

**2. Add to `config.json`:**
```json
{
  "web_search": {
    "providers": ["ddgs", "google", "bing"]
  }
}
```

**3. Implement search function in `src/service/web/retriever.py`:**
```python
def get_web_Bing(query, results=5):
    # Implement Bing search logic
    pass

# Add to dispatch map
web_search_map = {
    "ddgs": get_web_DDGS,
    "google": get_web_Google,
    "bing": get_web_Bing,  # New provider
}
```

**4. Update `src/core/config.py` key mapping:**
```python
key_map = {
    'google': 'GOOGLE_SEARCH_API_KEY',
    'bing': 'BING_SEARCH_API_KEY',  # Add this line
}
```

### Usage in Code

#### Import and Use Models

```python
from src.llm.chat.api.chat_model import get_chat_model
from src.llm.embed.api.embed_model import get_embed_model
from src.service.web.retriever import get_web_message

# Get chat model (automatically loads config from .env + config.json)
chat_model = get_chat_model(temperature=0.7)
response = chat_model.invoke("What is RAG?")

# Get embedding model
embed_model = get_embed_model()
vector = embed_model.embed_query("What is RAG?")

# Get web search results (uses fallback strategy from config.json)
results = get_web_message("What is RAG?")
```

#### Check Current Configuration

```bash
python -c "import sys; sys.path.insert(0, 'src'); from core.config import config; print('Chat:', config.get_chat_config()['provider']); print('Embed:', config.get_embed_config()['provider'])"
```

Output:
```
Chat: deepseek
Embed: aliyun
```

#### Advanced Configuration

**Custom Temperature and Max Tokens:**
```python
from src.llm.chat.api.chat_model import get_chat_model

# Override default parameters
chat_model = get_chat_model(
    temperature=0.3,  # More deterministic
    max_tokens=8192   # Longer responses
)
```

**Specify Web Search Provider:**
```python
from src.service.web.retriever import get_web_message

# Use specific provider (skip fallback)
results = get_web_message("query", method="google")

# Use fallback strategy (default)
results = get_web_message("query")  # Tries ddgs -> google -> bing
```

### Logging

SCI4RAG uses a three-tier logging system powered by [loguru](https://github.com/Delgan/loguru).

#### Quick Usage

```python
from src.core.logger import get_logger, get_user_logger

# Global logger (system-wide)
logger = get_logger()
logger.info("System started")

# User logger (logs to users/{username}/logs/)
logger = get_user_logger("admin")
logger.info("User logged in")

# Dataset logger (logs to users/{username}/{dataset}/logs/)
logger = get_user_logger("admin", "schwarz")
logger.info("Processing document")
logger.success("Document parsed")
logger.warning("Missing DOI")
logger.error("Failed to extract references")
```

#### Log Levels

| Level | Use Case |
|-------|----------|
| `debug()` | Detailed debugging info, variable values |
| `info()` | General progress, step indicators |
| `success()` | Completed operations |
| `warning()` | Non-critical issues, missing optional data |
| `error()` | Operation failures |
| `critical()` | Critical system errors |
| `exception()` | Errors with full traceback |

#### Log File Locations

| Scope | Directory |
|-------|-----------|
| Global | `logs/sci4rag.log`, `logs/error.log` |
| User | `users/{username}/logs/activity.log` |
| Dataset | `users/{username}/{dataset}/logs/activity.log` |

#### Best Practice Pattern

```python
from src.core.logger import get_user_logger
from src.core.paths import parse_path_info

def process_document(file_data: dict):
    # Parse context from file path
    username, dataset_name = parse_path_info(file_data["file_path"])
    logger = get_user_logger(username, dataset_name)
    
    logger.info("Processing: {name}", name=file_data["file_name"])
    
    try:
        result = do_work(file_data)
        logger.success("Processing complete")
        return result
    except Exception:
        logger.exception("Processing failed")
        raise
```

### Error Handling

#### Missing API Key

If an API key is not configured, you'll get a clear error message:

```
ValueError: API key not found: DEEPSEEK_API_KEY is not set in .env file.
Please add DEEPSEEK_API_KEY=your_api_key to your .env file.
```

**Solution:** Add the missing API key to your `.env` file.

#### Unknown Provider

If you specify an unknown provider:

```
ValueError: Unknown chat provider: 'unknown_provider'.
Available providers: ['aliyun', 'deepseek', 'openai', 'qwen']
```

**Solution:** Use one of the available providers listed in the error message.

### Troubleshooting

#### Configuration not loading

**Problem:** Changes to `config.json` not taking effect

**Solution:** Restart your Python process. Configuration is loaded once when you first import a model.

#### Import errors

**Problem:** `ModuleNotFoundError: No module named 'config'`

**Solution:** Make sure you're running from the project root directory and `src/` is in your Python path.

#### Web search not working

**Problem:** All web search providers failing

**Solution:** 
1. Check if `ddgs` is in your providers list (it's free and doesn't need API key)
2. If using `google`, verify `GOOGLE_SEARCH_API_KEY` is set in `.env`
3. Check your internet connection

### File Structure

```
SCI4RAG/
├── .env                    # API keys (DO NOT COMMIT)
├── .env.example            # Template (commit this)
├── config.json             # Model configurations (commit this)
└── src/
    ├── core/
    │   └── config.py       # Configuration loader
    ├── llm/
    │   ├── chat/api/chat_model.py      # Loads from config.py
    │   └── embed/api/embed_model.py    # Loads from config.py
    └── service/
        ├── parse/mineru/api/parser.py  # Loads from config.py
        └── web/retriever.py            # Loads from config.py
```

**Old `api_key` files are no longer needed** - they have been removed.

---

## 🔒 Security

### What's Protected

`.gitignore` is configured to ignore:
- `.env` - Your API keys
- `**/api_key` - Old API key files (no longer used)
- `users/` - User data
- `__pycache__/` - Python cache files

### What's Safe to Commit

- `.env.example` - Template without real keys
- `config.json` - Model configurations (no sensitive data)
- All source code

### If You Accidentally Commit Secrets

1. **Immediately revoke all exposed API keys**
2. Generate new API keys from your providers
3. Update your `.env` file with new keys
4. Use `git filter-branch` or `BFG Repo-Cleaner` to remove secrets from git history

---

## 🙌 Contributing

SchwarzRAG thrives through open-source collaboration. We welcome:
- Code contributions
- Domain knowledge expansion
- System testing & issue reporting
- Documentation translation & refinement

### Development Workflow

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Configure your `.env` file (never commit this)
4. Make your changes
5. Test your changes
6. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
7. Push to the branch (`git push origin feature/AmazingFeature`)
8. Open a Pull Request

---

## 📝 License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

## 📧 Contact

For questions or support, please open an issue on GitHub.
