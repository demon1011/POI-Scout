"""
配置文件模板 - 存放API密钥和配置信息

使用方法：
1. 复制此文件并重命名为 config.py
2. 填入你的 API 密钥
3. 确保 config.py 不被提交到 git（已在 .gitignore 中配置）

命令: cp config.example.py config.py
"""

# ====================
# SiliconFlow API 配置
# ====================

SILICONFLOW_API_KEY = "your-siliconflow-api-key-here"
SILICONFLOW_API_BASE = "https://api.siliconflow.cn/v1"

# ====================
# 搜索 API 配置
# ====================

# 博查搜索 API
BOCHA_API_KEY = "your-bocha-api-key-here"
BOCHA_API_URL = "https://api.bochaai.com/v1/web-search"

# ====================
# Embedding API 配置
# ====================

EMBEDDING_API_URL = "https://api.siliconflow.cn/v1/embeddings"
EMBEDDING_API_KEY = SILICONFLOW_API_KEY
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"

# ====================
# 模型配置
# ====================

DEFAULT_LARGE_MODEL = "deepseek-r1"
DEFAULT_SMALL_MODEL = "Qwen/Qwen3-8B"
DEFAULT_V3_MODEL = "Pro/deepseek-ai/DeepSeek-V3.2"
DEFAULT_R1_MODEL = "Pro/deepseek-ai/DeepSeek-R1"

# ====================
# 系统配置
# ====================

DECISION_TREE_SAVE_DIR = "./data/decision_trees"
DEFAULT_URL_COUNT = 5
MAX_RETRY_TIMES = 3
