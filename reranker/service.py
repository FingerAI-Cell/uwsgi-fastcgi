"""
Service layer for reranking functionality
"""

import os
import logging
import torch
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from flashrank import Ranker, RerankRequest

# 더 빠른 JSON 처리를 위해 ujson 사용
try:
    import ujson as json
    print("Using ujson for faster JSON processing")
except ImportError:
    import json
    print("ujson not available, using default json")

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 메모리 캐시 - 자주 사용되는 재랭킹 요청 결과 캐싱
_RERANK_CACHE = {}
_CACHE_SIZE_LIMIT = 1000  # 최대 캐시 항목 수

def get_cache_key(query: str, passages_hash: str) -> str:
    """캐시 키 생성"""
    return f"{query}:{passages_hash}"

def hash_passages(passages: List[Dict]) -> str:
    """패시지 리스트의 해시 생성"""
    try:
        passage_texts = [p.get('text', '')[:100] for p in passages]  # 각 패시지의 앞부분만 사용
        return str(hash(tuple(passage_texts)))
    except Exception as e:
        logger.warning(f"Failed to hash passages: {e}")
        return str(hash(str(passages)))  # 폴백 해싱

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
        
    # 메모리 효율성을 위한 __slots__ 추가
    __slots__ = ('passage_id', 'doc_id', 'text', 'score', 'position', 'metadata')


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
        
    # 메모리 효율성을 위한 __slots__ 추가
    __slots__ = ('query', 'results', 'total', 'reranked')


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
        
    # 메모리 효율성을 위한 __slots__ 추가
    __slots__ = ('query', 'results', 'total', 'reranked')

    def json(self, **kwargs):
        # ujson 사용 가능하면 ujson으로 직렬화
        if 'ujson' in globals():
            return json.dumps(self.dict(), ensure_ascii=False, **kwargs)
        return json.dumps(self.dict(), ensure_ascii=False, **kwargs)


class RerankerService:
    """Service for reranking passages"""
    
    _instance = None  # 싱글톤 인스턴스
    
    @classmethod
    def get_instance(cls, config_path=None):
        """싱글톤 패턴으로 인스턴스 반환"""
        if cls._instance is None:
            cls._instance = cls(config_path)
        return cls._instance
    
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
            
            # 배치 크기 설정
            self.batch_size = self._get_batch_size()
            
            logger.info(f"Initializing FlashRank reranker with model: {self.model_name}")
            logger.debug(f"Cache directory: {self.cache_dir}")
            logger.debug(f"Max length: {self.max_length}")
            logger.debug(f"Batch size: {self.batch_size}")
            
            # 모델 초기화를 try-except로 감싸서 실패해도 서비스는 계속 실행되도록 함
            try:
                logger.info("Starting model initialization...")
                logger.debug(f"Model path: {os.path.join(self.cache_dir, self.model_name)}")
                
                # 모델 디렉토리 존재 여부 확인
                if not os.path.exists(self.cache_dir):
                    logger.info(f"Creating cache directory: {self.cache_dir}")
                    os.makedirs(self.cache_dir, exist_ok=True)
                
                # GPU 사용 가능 여부 확인
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Using device: {self.device}")
                
                # FlashRank 0.2.10 버전에 맞게 Ranker 초기화
                try:
                    # FlashRank 0.2.10 버전에서는 batch_size, max_length 등의 매개변수를 지원하지 않음
                    # 기본 매개변수만 사용
                    self.ranker = Ranker(
                        model_name=self.model_name
                    )
                    logger.info("FlashRank reranker initialized with basic parameters")
                except Exception as e:
                    logger.warning(f"Failed with basic parameters: {e}, trying without parameters")
                    # 매개변수 없이 초기화 시도
                    self.ranker = Ranker()
                    logger.info("FlashRank reranker initialized without parameters")
                
                # 모델 미리 로드 (첫 요청 지연 방지)
                logger.info("Pre-warming model...")
                try:
                    # FlashRank 0.2.10 API에 맞게 예열 요청 구성
                    dummy_request = RerankRequest(
                        query="warm up query",
                        passages=[{"id": "0", "text": "warm up passage", "meta": {}}]
                    )
                    self.ranker.rerank(dummy_request)
                    logger.info("Model pre-warming complete")
                except Exception as e:
                    logger.warning(f"Model pre-warming failed: {e}, this is not critical")
                
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
    
    def _get_batch_size(self) -> int:
        """환경에 맞는 배치 크기 결정"""
        # 기본 배치 크기
        default_batch_size = {
            "cpu": 32,
            "gpu": 256
        }
        
        # GPU 여부에 따라 배치 크기 선택
        mode = "gpu" if torch.cuda.is_available() else "cpu"
        
        # 설정된 배치 크기 가져오기
        if isinstance(self.config.get("batch_size"), dict):
            return self.config["batch_size"].get(mode, default_batch_size[mode])
        elif isinstance(self.config.get("batch_size"), (int, str)):
            return int(self.config["batch_size"])
        else:
            return default_batch_size[mode]
    
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dictionary
        """
        default_batch_size = {
            "cpu": 32,
            "gpu": 256
        }
        
        default_config = {
            "model_name": os.getenv("FLASHRANK_MODEL", "ms-marco-TinyBERT-L-2-v2"),
            "cache_dir": os.getenv("FLASHRANK_CACHE_DIR", "/reranker/models"),
            "max_length": int(os.getenv("FLASHRANK_MAX_LENGTH", "512")),
            "batch_size": default_batch_size
        }
        
        if not config_path:
            return default_config
            
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
                # GPU 여부에 따라 배치 사이즈 선택
                if isinstance(config.get("batch_size"), dict):
                    mode = "gpu" if torch.cuda.is_available() else "cpu"
                    config["batch_size"] = config["batch_size"].get(mode, default_batch_size[mode])
                elif isinstance(config.get("batch_size"), (int, str)):
                    # 이전 형식의 설정을 위한 하위 호환성 유지
                    config["batch_size"] = int(config["batch_size"])
                    
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
            # 성능 측정 시작
            import time
            start_time = time.time()
            
            # 모델이 초기화되지 않은 경우 원본 결과를 그대로 반환
            if self.ranker is None:
                logger.warning("Reranker not initialized, returning original results")
                return search_result
                
            # 결과가 없으면 빈 결과 반환
            if not search_result.get("results"):
                logger.warning("No results to rerank")
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
            
            # 캐시 키 생성 및 캐시 조회
            passages_hash = hash_passages(passages)
            cache_key = get_cache_key(query, passages_hash)
            
            if cache_key in _RERANK_CACHE:
                logger.info(f"Cache hit for query: '{query}'")
                cached_result = _RERANK_CACHE[cache_key]
                
                # top_k가 다른 경우 조정
                if top_k is not None:
                    cached_result["results"] = cached_result["results"][:top_k]
                    cached_result["total"] = len(cached_result["results"])
                
                # 캐시된 결과 반환, 처리 시간은 현재 시간으로 업데이트
                elapsed_time = time.time() - start_time
                cached_result["processing_time"] = elapsed_time
                cached_result["cached"] = True
                return cached_result
            
            # 대량 패시지 처리 최적화
            total_passages = len(passages)
            logger.info(f"Reranking {total_passages} passages for query: '{query}'")
            
            # 배치 처리를 위한 최적 크기 계산
            batch_size = min(self.batch_size, total_passages)
            
            # 배치 처리
            reranked_results = []
            try:
                if total_passages > batch_size:
                    for i in range(0, total_passages, batch_size):
                        batch_passages = passages[i:i + batch_size]
                        
                        # Create rerank request
                        rerank_request = RerankRequest(query=query, passages=batch_passages)
                        
                        # Rerank passages
                        batch_results = self.ranker.rerank(rerank_request)
                        reranked_results.extend(batch_results)
                        
                        # GPU 메모리 정리
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                else:
                    # 소량 데이터는 한 번에 처리
                    rerank_request = RerankRequest(query=query, passages=passages)
                    reranked_results = self.ranker.rerank(rerank_request)
            except Exception as e:
                logger.error(f"Error during reranking: {e}")
                logger.warning("Falling back to original results")
                # 오류 발생 시 원본 결과 반환
                return search_result
            
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
            
            # 성능 측정 종료
            elapsed_time = time.time() - start_time
            logger.info(f"Reranking completed in {elapsed_time:.3f} seconds for {total_passages} passages")
            
            # 결과 생성
            final_result = {
                "query": query,
                "results": processed_results,
                "total": len(processed_results),
                "reranked": True,
                "processing_time": elapsed_time,
                "cached": False
            }
            
            # 캐시에 결과 저장 (top_k 적용 전의 전체 결과)
            if len(_RERANK_CACHE) >= _CACHE_SIZE_LIMIT:
                # 캐시가 가득 차면 무작위로 하나 제거
                import random
                key_to_remove = random.choice(list(_RERANK_CACHE.keys()))
                _RERANK_CACHE.pop(key_to_remove, None)
                
            _RERANK_CACHE[cache_key] = final_result
            
            return final_result
            
        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            raise 