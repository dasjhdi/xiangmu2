# app/config/rag_config.py
"""
RAG (Retrieval-Augmented Generation) configuration models.
These models define the structure for the config.yml file.

Design principles:
1. Database connection configs are defined in DatabaseConfig and passed in
2. RAG-specific configs only define their unique fields
3. LLM config uses the global LLMConfig (layered: fast/normal)
"""

from pydantic import BaseModel
from typing import List, Literal, Optional, Dict


# =============================================================================
# RAG-Specific Configurations
# =============================================================================

class PathsConfig(BaseModel):
    """Data paths configuration."""
    base_data_path: str = "data/HowToCook"


class VectorStoreConfig(BaseModel):
    """
    Vector store configuration.
    Note: Connection settings (host, port, credentials) are in DatabaseConfig.milvus
    """
    type: Literal["milvus"] = "milvus"
    collection_names: Dict[str, str] = {
        "recipes": "cook_hero_recipes",
        "personal": "cook_hero_personal_docs",
    }


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""
    model_name: str = "BAAI/bge-small-zh-v1.5"


class RetrievalConfig(BaseModel):
    """Retrieval pipeline configuration."""
    top_k: int = 9
    score_threshold: float = 0.2
    ranker_type: Literal["rrf", "weighted"] = "weighted"
    ranker_weights: List[float] = [0.8, 0.2]  # [dense, sparse]
    # LLM-driven query decomposition (extra latency/cost; off by default)
    multi_query_enabled: bool = False
    max_queries: int = 3
    # 查询改写是最大的延迟抖动源：超过预算直接用原问题检索。
    rewrite_timeout_seconds: float = 2.0
    # 首轮或问题本身已包含明确菜名/做法时，跳过改写，省一次 fast LLM 往返。
    skip_rewrite_when_explicit: bool = True
    # 元数据过滤是第二次 fast LLM 调用，同样需要封顶。
    metadata_timeout_seconds: float = 2.0


class RerankerConfig(BaseModel):
    """
    Reranker configuration.
    Note: api_key defaults to global LLM API key if not set.
    """
    enabled: bool = True
    type: Literal["siliconflow"] = "siliconflow"
    model_name: str = "Qwen/Qwen3-Reranker-8B"
    base_url: Optional[str] = "https://api.siliconflow.cn/v1/rerank"
    api_key: Optional[str] = None  # Falls back to global LLM API key
    temperature: float = 0.0
    max_tokens: int = 8192
    score_threshold: float = 0.1
    # 单次 HTTP 请求超时：30s 会让一次故障拖垮整轮对话，3s 足够正常请求。
    request_timeout: float = 3.0
    # 整个重排阶段的硬上限（含本地 cross-encoder 降级），超时保留原顺序。
    timeout_seconds: float = 6.0
    # 送入重排的候选数与重排后保留数：少而精，降低延迟与上下文噪声。
    max_candidates: int = 6
    top_n: int = 5
    # 明确菜名/做法的查询，混合检索排序已足够好；重排反而会把正确菜名往后挪。
    skip_rerank_when_explicit: bool = True
    # Optional local cross-encoder fallback (off by default; no model download at startup)
    local_cross_encoder_enabled: bool = False
    local_cross_encoder_model: Optional[str] = None


class CacheConfig(BaseModel):
    """
    Cache configuration for RAG retrieval results.
    
    Cache strategy:
    - L1: Exact match (Redis) - fast lookup for identical queries
    - L2: Semantic match (Milvus) - handles similar queries
    
    Note: 
    - Connection settings are in DatabaseConfig (redis/milvus)
    - Only caches Query -> Retrieved Documents, NOT LLM responses.
    """
    enabled: bool = True
    # TTL for both L1 and L2
    ttl: int = 3600  # 1 hour
    # L2 semantic cache
    l2_enabled: bool = True
    similarity_threshold: float = 0.92
    vector_collection: str = "cookhero_retrieval_cache"
    # Bump this when the indexed corpus changes so cached results are not reused
    corpus_version: Optional[str] = None


class HowToCookConfig(BaseModel):
    """HowToCook recipe data source configuration (includes tips)."""
    path_suffix: str = "dishes"
    tips_path_suffix: str = "tips"  # Tips are now loaded together
    headers_to_split_on: List[List[str]] = [["#", "header_1"], ["##", "header_2"]]


class DataSourceConfig(BaseModel):
    """Data sources configuration."""
    howtocook: HowToCookConfig = HowToCookConfig()


# =============================================================================
# Main RAG Configuration
# =============================================================================

class RAGConfig(BaseModel):
    """
    Main RAG configuration model.
    
    Note: 
    - Database connections are in DatabaseConfig and passed separately
    - LLM configuration uses global LLMConfig (layered: fast/normal)
    """
    # Module configurations
    paths: PathsConfig = PathsConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    reranker: RerankerConfig = RerankerConfig()
    cache: CacheConfig = CacheConfig()
    data_source: DataSourceConfig = DataSourceConfig()


