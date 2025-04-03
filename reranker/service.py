"""
Service layer for reranking functionality
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from flashrank import Ranker, RerankRequest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PassageModel(BaseModel):
    """Single passage model"""
    passage_id: Optional[Any] = None
    doc_id: Optional[str] = None
    text: str
    score: Optional[float] = None
    position: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        json_encoders = {
            str: lambda v: v.encode('utf-8').decode('utf-8')
        }


class SearchResultModel(BaseModel):
    """Search result containing multiple passages"""
    query: str
    results: List[PassageModel]
    total: Optional[int] = None
    reranked: Optional[bool] = False

    class Config:
        json_encoders = {
            str: lambda v: v.encode('utf-8').decode('utf-8')
        }


class RerankerResponseModel(BaseModel):
    """Response model for reranker API"""
    query: str
    results: List[PassageModel]
    total: int
    reranked: bool = True

    class Config:
        json_encoders = {
            str: lambda v: v.encode('utf-8').decode('utf-8')
        }

    def json(self, **kwargs):
        return json.dumps(self.dict(), ensure_ascii=False, **kwargs)


class RerankerService:
    """Service for reranking passages"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize the reranker service
        
        Args:
            config_path: Path to config file, if None, use default settings
        """
        self.config = self._load_config(config_path)
        self.model_name = os.getenv("FLASHRANK_MODEL", self.config.get("model_name", "ms-marco-TinyBERT-L-2-v2"))
        self.cache_dir = os.getenv("FLASHRANK_CACHE_DIR", self.config.get("cache_dir", "/reranker/models"))
        self.max_length = int(os.getenv("FLASHRANK_MAX_LENGTH", self.config.get("max_length", 512)))
        
        logger.info(f"Initializing FlashRank reranker with model: {self.model_name}")
        self.ranker = Ranker(
            model_name=self.model_name,
            cache_dir=self.cache_dir,
            max_length=self.max_length
        )
    
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            "model_name": os.getenv("FLASHRANK_MODEL", "ms-marco-TinyBERT-L-2-v2"),
            "cache_dir": os.getenv("FLASHRANK_CACHE_DIR", "/reranker/models"),
            "max_length": int(os.getenv("FLASHRANK_MAX_LENGTH", "512")),
            "batch_size": int(os.getenv("FLASHRANK_BATCH_SIZE", "32"))
        }
        
        if not config_path:
            return default_config
            
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                return {**default_config, **config}
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            logger.info("Using default configuration")
            return default_config
    
    def process_search_results(self, query: str, search_result: Dict[str, Any], top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Process search results using reranker
        
        Args:
            query: Search query
            search_result: Search results to rerank
            top_k: Number of results to return
            
        Returns:
            Reranked search results
        """
        try:
            # Convert passages to FlashRank format
            passages = []
            for result in search_result["results"]:
                passage = {
                    "id": result.get("passage_id"),
                    "text": result["text"],
                    "meta": {
                        "doc_id": result.get("doc_id"),
                        "original_score": result.get("score")
                    }
                }
                passages.append(passage)
            
            # Create rerank request
            rerank_request = RerankRequest(query=query, passages=passages)
            
            # Rerank passages
            logger.info(f"Reranking {len(passages)} passages for query: '{query}'")
            reranked_results = self.ranker.rerank(rerank_request)
            
            # Convert back to original format
            processed_results = []
            for result in reranked_results:
                processed_result = {
                    "passage_id": result["id"],
                    "doc_id": result["meta"]["doc_id"],
                    "text": result["text"],
                    "score": float(result["score"]),
                    "metadata": result["meta"]
                }
                processed_results.append(processed_result)
            
            # Apply top_k if specified
            if top_k is not None:
                processed_results = processed_results[:top_k]
            
            return {
                "query": query,
                "results": processed_results,
                "total": len(processed_results),
                "reranked": True
            }
            
        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            raise 