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
logging.basicConfig(level=logging.DEBUG)
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
        try:
            logger.debug("Loading configuration...")
            self.config = self._load_config(config_path)
            self.model_name = os.getenv("FLASHRANK_MODEL", self.config.get("model_name", "ms-marco-TinyBERT-L-2-v2"))
            self.cache_dir = os.getenv("FLASHRANK_CACHE_DIR", self.config.get("cache_dir", "/reranker/models"))
            self.max_length = int(os.getenv("FLASHRANK_MAX_LENGTH", self.config.get("max_length", 512)))
            
            logger.info(f"Initializing FlashRank reranker with model: {self.model_name}")
            logger.debug(f"Cache directory: {self.cache_dir}")
            logger.debug(f"Max length: {self.max_length}")
            
            # 모델 초기화를 try-except로 감싸서 실패해도 서비스는 계속 실행되도록 함
            try:
                logger.info("Starting model initialization...")
                logger.debug(f"Model path: {os.path.join(self.cache_dir, self.model_name)}")
                
                # 모델 디렉토리 존재 여부 확인
                if not os.path.exists(self.cache_dir):
                    logger.info(f"Creating cache directory: {self.cache_dir}")
                    os.makedirs(self.cache_dir, exist_ok=True)
                
                self.ranker = Ranker(
                    model_name=self.model_name,
                    cache_dir=self.cache_dir,
                    max_length=self.max_length
                )
                logger.info("FlashRank reranker initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize FlashRank reranker: {str(e)}")
                logger.error(f"Error type: {type(e)}")
                logger.error(f"Error details: {str(e)}")
                logger.info("Using dummy reranker for testing")
                self.ranker = None
                
        except Exception as e:
            logger.error(f"Failed to initialize RerankerService: {str(e)}")
            raise
    
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
    
    def process_search_results(self, query: str, search_result: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
        """
        Process search results with reranking
        
        Args:
            query: Search query
            search_result: Search results to rerank
            top_k: Number of top results to return
            
        Returns:
            Reranked search results
        """
        try:
            # 모델이 초기화되지 않은 경우 원본 결과를 그대로 반환
            if self.ranker is None:
                logger.warning("Reranker not initialized, returning original results")
                return search_result
                
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