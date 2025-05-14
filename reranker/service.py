"""
Service layer for reranking functionality
"""

import os
import logging
import torch
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from flashrank import Ranker, RerankRequest

# PyTorch 프로파일러 임포트
try:
    from torch.profiler import profile, record_function, ProfilerActivity
    PROFILER_AVAILABLE = True
except ImportError:
    PROFILER_AVAILABLE = False
    
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

# 프로파일링 활성화 여부
ENABLE_PROFILING = os.getenv("ENABLE_PROFILING", "0") == "1"
PROFILE_OUTPUT_DIR = os.getenv("PROFILE_OUTPUT_DIR", "/reranker/profiles")

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

# GPU 메모리 상태 로깅 함수 추가
def log_gpu_memory(tag: str = ""):
    """GPU 메모리 사용량 로깅"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)    # MB
        max_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
        logger.info(f"GPU Memory [{tag}] - Allocated: {allocated:.2f}MB, Reserved: {reserved:.2f}MB, Max: {max_allocated:.2f}MB")
    else:
        logger.info(f"GPU Memory [{tag}] - CUDA not available")

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
            logger.info(f"Creating new RerankerService instance with config: {config_path}")
            cls._instance = cls(config_path)
        else:
            logger.info("Returning existing RerankerService instance")
        return cls._instance
    
    def __init__(self, config_path: str = None):
        """
        Initialize the reranker service
        
        Args:
            config_path: Path to config file, if None, use default settings
        """
        try:
            init_start_time = time.time()  # 초기화 시작 시간
            logger.debug("Loading configuration...")
            self.config = self._load_config(config_path)
            logger.info(f"Configuration loaded in {(time.time() - init_start_time)*1000:.2f}ms")
            
            self.model_name = os.getenv("FLASHRANK_MODEL", self.config.get("model_name", "ms-marco-TinyBERT-L-2-v2"))
            self.cache_dir = os.getenv("FLASHRANK_CACHE_DIR", self.config.get("cache_dir", "/reranker/models"))
            self.max_length = int(os.getenv("FLASHRANK_MAX_LENGTH", self.config.get("max_length", 512)))
            
            # 배치 크기 설정
            self.batch_size = self._get_batch_size()
            
            # 프로파일링 설정
            self.enable_profiling = ENABLE_PROFILING and PROFILER_AVAILABLE
            self.profile_dir = PROFILE_OUTPUT_DIR
            if self.enable_profiling:
                logger.info(f"PyTorch profiling enabled. Profiles will be saved to {self.profile_dir}")
                os.makedirs(self.profile_dir, exist_ok=True)
            
            logger.info(f"Initializing FlashRank reranker with model: {self.model_name}")
            logger.debug(f"Cache directory: {self.cache_dir}")
            logger.debug(f"Max length: {self.max_length}")
            logger.debug(f"Batch size: {self.batch_size}")
            
            # 시스템 정보 로깅
            self._log_system_info()
            
            # 모델 초기화를 try-except로 감싸서 실패해도 서비스는 계속 실행되도록 함
            try:
                model_init_start = time.time()  # 모델 초기화 시작 시간
                logger.info("Starting model initialization...")
                logger.debug(f"Model path: {os.path.join(self.cache_dir, self.model_name)}")
                
                # 모델 디렉토리 존재 여부 확인
                if not os.path.exists(self.cache_dir):
                    logger.info(f"Creating cache directory: {self.cache_dir}")
                    os.makedirs(self.cache_dir, exist_ok=True)
                
                # GPU 사용 가능 여부 확인
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Using device: {self.device}")
                
                # GPU 메모리 초기 상태 로깅
                log_gpu_memory("초기화 전")
                
                # FlashRank 0.2.10 버전에 맞게 Ranker 초기화
                try:
                    ranker_init_start = time.time()  # Ranker 초기화 시작 시간
                    # FlashRank 0.2.10 버전에서는 batch_size, max_length 등의 매개변수를 지원하지 않음
                    # 기본 매개변수만 사용
                    self.ranker = Ranker(
                        model_name=self.model_name
                    )
                    logger.info(f"FlashRank ranker initialized in {(time.time() - ranker_init_start)*1000:.2f}ms")
                    
                    # 모델 아키텍처 정보 로깅
                    if hasattr(self.ranker, 'model'):
                        logger.info(f"Model type: {type(self.ranker.model).__name__}")
                        
                    logger.info("FlashRank reranker initialized with basic parameters")
                except Exception as e:
                    logger.warning(f"Failed with basic parameters: {e}, trying without parameters")
                    ranker_init_start = time.time()  # 두 번째 시도 시작 시간
                    # 매개변수 없이 초기화 시도
                    self.ranker = Ranker()
                    logger.info(f"FlashRank ranker initialized without parameters in {(time.time() - ranker_init_start)*1000:.2f}ms")
                
                # GPU 메모리 모델 로드 후 상태 로깅
                log_gpu_memory("모델 로드 후")
                
                # 모델 미리 로드 (첫 요청 지연 방지)
                logger.info("Pre-warming model...")
                warm_start = time.time()  # 예열 시작 시간
                try:
                    # CUDA 스트림 동기화
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        
                    # FlashRank 0.2.10 API에 맞게 예열 요청 구성
                    dummy_request = RerankRequest(
                        query="warm up query",
                        passages=[{"id": "0", "text": "warm up passage", "meta": {}}]
                    )
                    
                    # 예열 수행
                    warm_results = self.ranker.rerank(dummy_request)
                    
                    # CUDA 스트림 동기화
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        
                    logger.info(f"Model pre-warming completed in {(time.time() - warm_start)*1000:.2f}ms")
                    logger.debug(f"Warm-up result: {warm_results}")
                except Exception as e:
                    logger.warning(f"Model pre-warming failed: {e}, this is not critical")
                
                # GPU 메모리 예열 후 상태 로깅
                log_gpu_memory("예열 후")
                
                logger.info(f"FlashRank reranker initialization completed in {(time.time() - model_init_start)*1000:.2f}ms")
            except Exception as e:
                logger.error(f"Failed to initialize FlashRank reranker: {str(e)}")
                logger.error(f"Error type: {type(e)}")
                logger.error(f"Error details: {str(e)}")
                logger.info("Using dummy reranker for testing")
                self.ranker = None
                
            logger.info(f"Total initialization time: {(time.time() - init_start_time)*1000:.2f}ms")
                
        except Exception as e:
            logger.error(f"Failed to initialize RerankerService: {str(e)}")
            raise
    
    def _log_system_info(self):
        """시스템 정보 로깅"""
        logger.info("=== System Information ===")
        
        # Python 버전
        import sys
        logger.info(f"Python version: {sys.version}")
        
        # PyTorch 버전
        logger.info(f"PyTorch version: {torch.__version__}")
        
        # CUDA 정보
        if torch.cuda.is_available():
            logger.info(f"CUDA available: True")
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            logger.info("CUDA available: False")
            
        logger.info("========================")
    
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
            start_time = time.time()
            
            # 상세 시간 측정을 위한 타임스탬프 딕셔너리
            timestamps = {
                "start": start_time,
                "steps": []
            }
            
            def log_step(name):
                now = time.time()
                step_time = now - timestamps.get("last_time", start_time)
                elapsed = now - start_time
                timestamps["steps"].append({"name": name, "time": step_time, "elapsed": elapsed})
                timestamps["last_time"] = now
                logger.debug(f"Step '{name}' took {step_time*1000:.2f}ms (elapsed: {elapsed*1000:.2f}ms)")
            
            # 모델이 초기화되지 않은 경우 원본 결과를 그대로 반환
            if self.ranker is None:
                logger.warning("Reranker not initialized, returning original results")
                return search_result
                
            # 결과가 없으면 빈 결과 반환
            if not search_result.get("results"):
                logger.warning("No results to rerank")
                return search_result
            
            # GPU 메모리 초기 상태 로깅
            log_gpu_memory("재랭킹 시작")
                
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
            
            log_step("데이터 포맷 변환")
            
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
                
                log_step("캐시 히트 처리")
                return cached_result
            
            log_step("캐시 확인")
            
            # 대량 패시지 처리 최적화
            total_passages = len(passages)
            logger.info(f"Reranking {total_passages} passages for query: '{query}'")
            
            # 배치 처리를 위한 최적 크기 계산
            batch_size = min(self.batch_size, total_passages)
            logger.debug(f"Using batch size: {batch_size} for {total_passages} passages")
            
            # 동기화 시간 측정을 위한 변수 초기화
            sync_time = 0
            
            # CUDA 스트림 동기화 함수
            def sync_cuda():
                if torch.cuda.is_available():
                    sync_start = time.time()
                    torch.cuda.synchronize()
                    nonlocal sync_time
                    sync_time += time.time() - sync_start
            
            # 배치 처리 전 GPU 상태 확인
            log_gpu_memory("배치 처리 전")
            
            # CUDA 초기 동기화
            sync_cuda()
            
            # 프로파일링 결과를 저장할 변수
            profiler_output = None
            
            # 배치 처리
            reranked_results = []
            try:
                # 프로파일링 활성화 여부 확인
                if self.enable_profiling and PROFILER_AVAILABLE and torch.cuda.is_available():
                    logger.info("Starting PyTorch profiling for this reranking request")
                    profile_path = os.path.join(self.profile_dir, f"rerank_profile_{int(time.time())}")
                    
                    # 프로파일링 시작
                    with profile(
                        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        record_shapes=True,
                        profile_memory=True,
                        with_stack=True,
                        with_flops=True,
                        on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_path)
                    ) as prof:
                        if total_passages > batch_size:
                            logger.info(f"Processing in {(total_passages + batch_size - 1) // batch_size} batches with profiling")
                            
                            for i in range(0, total_passages, batch_size):
                                with record_function("batch_processing"):
                                    batch_start = time.time()
                                    batch_end = min(i + batch_size, total_passages)
                                    batch_passages = passages[i:batch_end]
                                    
                                    logger.debug(f"Processing batch {i//batch_size + 1}/{(total_passages + batch_size - 1) // batch_size} with {len(batch_passages)} passages")
                                    
                                    # Create rerank request
                                    rerank_request = RerankRequest(query=query, passages=batch_passages)
                                    
                                    # Rerank passages
                                    with record_function("rerank_call"):
                                        batch_results = self.ranker.rerank(rerank_request)
                                    
                                    # CUDA 동기화
                                    with record_function("cuda_sync"):
                                        sync_cuda()
                                    
                                    reranked_results.extend(batch_results)
                                    
                                    # GPU 메모리 정리
                                    if torch.cuda.is_available():
                                        with record_function("cuda_empty_cache"):
                                            torch.cuda.empty_cache()
                                    
                                    batch_time = time.time() - batch_start
                                    logger.debug(f"Batch {i//batch_size + 1} completed in {batch_time*1000:.2f}ms ({len(batch_passages)/batch_time:.1f} passages/sec)")
                                
                                # 프로파일러 스텝 진행
                                prof.step()
                        else:
                            # 소량 데이터는 한 번에 처리
                            with record_function("single_batch_processing"):
                                logger.debug(f"Processing all {total_passages} passages in a single batch")
                                
                                rerank_request = RerankRequest(query=query, passages=passages)
                                
                                with record_function("rerank_call"):
                                    reranked_results = self.ranker.rerank(rerank_request)
                                
                                # CUDA 동기화
                                with record_function("cuda_sync"):
                                    sync_cuda()
                                
                                # GPU 메모리 정리
                                if torch.cuda.is_available():
                                    with record_function("cuda_empty_cache"):
                                        torch.cuda.empty_cache()
                                
                            # 프로파일러 스텝 진행
                            prof.step()
                    
                    # 프로파일링 결과 저장
                    profiler_output = prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)
                    logger.info(f"Profiling complete. Results saved to {profile_path}")
                    logger.debug(f"Top 20 operations by CUDA time:\n{profiler_output}")
                    
                else:
                    # 프로파일링 없이 일반 실행
                    if total_passages > batch_size:
                        logger.info(f"Processing in {(total_passages + batch_size - 1) // batch_size} batches")
                        
                        for i in range(0, total_passages, batch_size):
                            batch_start = time.time()
                            batch_end = min(i + batch_size, total_passages)
                            batch_passages = passages[i:batch_end]
                            
                            logger.debug(f"Processing batch {i//batch_size + 1}/{(total_passages + batch_size - 1) // batch_size} with {len(batch_passages)} passages")
                            
                            # Create rerank request
                            rerank_request = RerankRequest(query=query, passages=batch_passages)
                            
                            # Rerank passages
                            batch_results = self.ranker.rerank(rerank_request)
                            
                            # CUDA 동기화
                            sync_cuda()
                            
                            reranked_results.extend(batch_results)
                            
                            # GPU 메모리 정리
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                
                            batch_time = time.time() - batch_start
                            logger.debug(f"Batch {i//batch_size + 1} completed in {batch_time*1000:.2f}ms ({len(batch_passages)/batch_time:.1f} passages/sec)")
                            
                            # 배치 처리 후 GPU 상태 확인
                            log_gpu_memory(f"배치 {i//batch_size + 1} 후")
                    else:
                        # 소량 데이터는 한 번에 처리
                        logger.debug(f"Processing all {total_passages} passages in a single batch")
                        
                        rerank_request = RerankRequest(query=query, passages=passages)
                        reranked_results = self.ranker.rerank(rerank_request)
                        
                        # CUDA 동기화
                        sync_cuda()
                        
                        # GPU 메모리 정리
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Error during reranking: {e}")
                logger.warning("Falling back to original results")
                # 오류 발생 시 원본 결과 반환
                
                # 오류 시 GPU 메모리 확인
                log_gpu_memory("랭킹 오류 후")
                
                # 오류 발생 시점까지의 진행 상황 로깅
                logger.debug(f"Progress before error: {json.dumps(timestamps['steps'])}")
                
                return search_result
            
            log_step("재랭킹 처리")
            
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
            
            log_step("결과 포맷 변환")
            
            # Apply top_k if specified
            if top_k is not None:
                processed_results = processed_results[:top_k]
            
            # 결과 준비
            result = {
                "query": query,
                "results": processed_results,
                "total": len(processed_results),
                "reranked": True
            }
            
            # 최종 성능 측정
            elapsed_time = time.time() - start_time
            result["processing_time"] = elapsed_time
            result["cached"] = False
            result["sync_time"] = sync_time
            result["performance"] = {
                "total_time": elapsed_time,
                "passages_per_second": total_passages / (elapsed_time - sync_time) if elapsed_time > sync_time else 0,
                "steps": timestamps["steps"]
            }
            
            # 프로파일링 결과가 있으면 추가
            if profiler_output:
                result["profiler_summary"] = str(profiler_output)
            
            # CUDA 최종 동기화
            sync_cuda()
            
            # 성능 로그 기록
            logger.info(f"Reranking completed in {elapsed_time:.3f} seconds for {total_passages} passages")
            logger.info(f"CUDA synchronization overhead: {sync_time:.3f} seconds")
            logger.info(f"Effective throughput: {total_passages / (elapsed_time - sync_time):.1f} passages/second")
            
            # 최종 GPU 메모리 상태 로깅
            log_gpu_memory("재랭킹 완료")
            
            # 최종 단계 로깅
            log_step("최종 결과 준비")
            
            # 결과 캐싱
            if len(_RERANK_CACHE) >= _CACHE_SIZE_LIMIT:
                # 캐시가 가득 찬 경우 오래된 항목 제거
                _RERANK_CACHE.pop(next(iter(_RERANK_CACHE)))
            _RERANK_CACHE[cache_key] = result
            
            log_step("캐시 업데이트")
            
            # 최종 결과 및 성능 측정 로그
            detailed_steps = "\n".join([
                f"  - {step['name']}: {step['time']*1000:.2f}ms ({step['elapsed']*1000:.2f}ms elapsed)"
                for step in timestamps["steps"]
            ])
            logger.debug(f"Detailed timing:\n{detailed_steps}")
            
            return result
        except Exception as e:
            logger.error(f"Error in process_search_results: {str(e)}", exc_info=True)
            return search_result 