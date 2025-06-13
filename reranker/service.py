"""
Service layer for reranking functionality
"""

import os
import sys
import logging
import torch
import time
import threading
from typing import List, Dict, Any, Optional, Union, Tuple
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

# 파일 로그 추가 (볼륨에 저장)
try:
    file_handler = logging.FileHandler('/var/log/reranker/reranker_detail.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.info("상세 로그 파일 설정 완료: /var/log/reranker/reranker_detail.log")
except Exception as e:
    logger.warning(f"로그 파일 설정 실패: {str(e)}")

# MRC 모듈 임포트
try:
    # 여러 경로 시도
    try:
        from src.mrc import MRCReranker
        MRC_AVAILABLE = True
        logger.info("MRC 모듈 가져오기 성공 (from src.mrc)")
    except ImportError as e:
        logger.error(f"MRC 모듈 가져오기 실패 (from src.mrc): {str(e)}")
        try:
            import sys
            sys.path.append('/reranker')
            logger.debug(f"Python 경로에 '/reranker' 추가: {sys.path}")
            from src.mrc import MRCReranker
            MRC_AVAILABLE = True
            logger.info("MRC 모듈 가져오기 성공 (from /reranker/src.mrc)")
        except ImportError as e:
            logger.error(f"MRC 모듈 가져오기 실패 (from /reranker/src.mrc): {str(e)}")
            try:
                from reranker.src.mrc import MRCReranker
                MRC_AVAILABLE = True
                logger.info("MRC 모듈 가져오기 성공 (from reranker.src.mrc)")
            except ImportError as e:
                logger.error(f"MRC 모듈 가져오기 실패 (from reranker.src.mrc): {str(e)}")
                try:
                    from .src.mrc import MRCReranker
                    MRC_AVAILABLE = True
                    logger.info("MRC 모듈 가져오기 성공 (from .src.mrc)")
                except ImportError as e:
                    logger.error(f"MRC 모듈 가져오기 실패 (from .src.mrc): {str(e)}")
                    MRC_AVAILABLE = False
                    logger.warning("MRC 모듈을 가져올 수 없습니다")
except Exception as e:
    MRC_AVAILABLE = False
    logger.error(f"MRC 모듈 임포트 중 오류 발생: {str(e)}", exc_info=True)

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
            
            # 로그 레벨 설정
            log_level = self.config.get("log_level", "INFO")
            log_level_int = getattr(logging, log_level.upper(), logging.INFO)
            logger.setLevel(log_level_int)
            logger.info(f"Log level set to {log_level}")
            
            self.model_name = os.getenv("FLASHRANK_MODEL", self.config.get("model_name", "ms-marco-TinyBERT-L-2-v2"))
            self.cache_dir = os.getenv("FLASHRANK_CACHE_DIR", self.config.get("cache_dir", "/reranker/models"))
            self.max_length = int(os.getenv("FLASHRANK_MAX_LENGTH", self.config.get("max_length", 512)))
            
            # 배치 크기 설정
            self.batch_size = self._get_batch_size()
            
            # GPU 동시 접근 제한 (환경변수로 설정 가능)
            self.max_gpu_workers = int(os.getenv('MAX_GPU_WORKERS', '7'))
            self._gpu_semaphore = threading.Semaphore(self.max_gpu_workers)
            logger.info(f"GPU 동시 작업 제한: {self.max_gpu_workers}개")
            
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
                
            # MRC 재랭커 초기화 (설정에서 활성화된 경우)
            self.mrc_enabled = self.config.get("mrc", {}).get("enabled", False)
            self.mrc_reranker = None
            self.hybrid_weight_mrc = self.config.get("mrc", {}).get("hybrid_weight_mrc", 0.7)
            
            logger.debug(f"MRC 초기화 시작: enabled={self.mrc_enabled}, MRC_AVAILABLE={MRC_AVAILABLE}")
            logger.debug(f"MRC 설정: {self.config.get('mrc', {})}")
            
            if self.mrc_enabled and MRC_AVAILABLE:
                try:
                    logger.info("MRC 재랭커 초기화 중...")
                    mrc_config_path = self.config.get("mrc", {}).get("model_config_path")
                    mrc_model_path = self.config.get("mrc", {}).get("model_ckpt_path")
                    
                    # 절대 경로 변환 시도
                    if not os.path.isabs(mrc_config_path) and not os.path.exists(mrc_config_path):
                        abs_config_path = os.path.abspath(mrc_config_path)
                        logger.debug(f"절대 경로로 변환: {mrc_config_path} -> {abs_config_path}")
                        mrc_config_path = abs_config_path
                        
                    if not os.path.isabs(mrc_model_path) and not os.path.exists(mrc_model_path):
                        abs_model_path = os.path.abspath(mrc_model_path)
                        logger.debug(f"절대 경로로 변환: {mrc_model_path} -> {abs_model_path}")
                        mrc_model_path = abs_model_path
                    
                    logger.debug(f"MRC 설정 파일 경로: {mrc_config_path}, 존재 여부: {os.path.exists(mrc_config_path)}")
                    logger.debug(f"MRC 모델 파일 경로: {mrc_model_path}, 존재 여부: {os.path.exists(mrc_model_path)}")
                    
                    # 파일 내용 로깅 (디버깅용)
                    try:
                        if os.path.exists(mrc_config_path):
                            with open(mrc_config_path, 'r') as f:
                                config_content = f.read()
                            logger.debug(f"MRC 설정 파일 내용: {config_content[:500]}...")
                    except Exception as e:
                        logger.warning(f"MRC 설정 파일 읽기 실패: {str(e)}")
                    
                    # MRC 모델 디렉토리 확인 및 생성
                    if mrc_config_path and mrc_model_path:
                        os.makedirs(os.path.dirname(mrc_config_path), exist_ok=True)
                        os.makedirs(os.path.dirname(mrc_model_path), exist_ok=True)
                        
                        # MRC 모델 다운로드 설정이 있는 경우
                        config_gdrive_id = self.config.get("mrc", {}).get("model_config_gdrive_id")
                        model_gdrive_id = self.config.get("mrc", {}).get("model_ckpt_gdrive_id")
                        
                        # 설정 파일 체크 및 다운로드
                        if config_gdrive_id and not os.path.exists(mrc_config_path):
                            try:
                                # 여러 경로에서 다운로드 함수 임포트 시도
                                try:
                                    from src.mrc import download_checkpoints
                                    logger.info("다운로드 함수 임포트 성공 (from src.mrc)")
                                except ImportError:
                                    try:
                                        from reranker.src.mrc import download_checkpoints
                                        logger.info("다운로드 함수 임포트 성공 (from reranker.src.mrc)")
                                    except ImportError:
                                        from .src.mrc import download_checkpoints
                                        logger.info("다운로드 함수 임포트 성공 (from .src.mrc)")
                                
                                logger.info(f"MRC 설정 파일 다운로드 중: {mrc_config_path}")
                                download_checkpoints(mrc_config_path, config_gdrive_id)
                            except Exception as e:
                                logger.warning(f"MRC 설정 파일 다운로드 실패: {e}")
                                logger.info(f"설정 파일을 '{mrc_config_path}' 경로에 수동으로 추가해주세요.")
                        
                        # 모델 체크포인트 체크 및 다운로드
                        if model_gdrive_id and not os.path.exists(mrc_model_path):
                            try:
                                # 여러 경로에서 다운로드 함수 임포트 시도
                                try:
                                    from src.mrc import download_checkpoints
                                    logger.info("다운로드 함수 임포트 성공 (from src.mrc)")
                                except ImportError:
                                    try:
                                        from reranker.src.mrc import download_checkpoints
                                        logger.info("다운로드 함수 임포트 성공 (from reranker.src.mrc)")
                                    except ImportError:
                                        from .src.mrc import download_checkpoints
                                        logger.info("다운로드 함수 임포트 성공 (from .src.mrc)")
                                
                                logger.info(f"MRC 모델 파일 다운로드 중: {mrc_model_path}")
                                download_checkpoints(mrc_model_path, model_gdrive_id)
                            except Exception as e:
                                logger.warning(f"MRC 모델 파일 다운로드 실패: {e}")
                                logger.info(f"모델 파일을 '{mrc_model_path}' 경로에 수동으로 추가해주세요.")
                    
                    # 경로 처리 (절대 경로에서 상대 경로로 변환)
                    if not os.path.exists(mrc_config_path) and mrc_config_path.startswith("/reranker/"):
                        relative_config_path = mrc_config_path[10:]  # "/reranker/" 제거
                        if os.path.exists(relative_config_path):
                            logger.info(f"상대 경로로 변환: {mrc_config_path} -> {relative_config_path}")
                            mrc_config_path = relative_config_path
                            
                    if not os.path.exists(mrc_model_path) and mrc_model_path.startswith("/reranker/"):
                        relative_model_path = mrc_model_path[10:]  # "/reranker/" 제거
                        if os.path.exists(relative_model_path):
                            logger.info(f"상대 경로로 변환: {mrc_model_path} -> {relative_model_path}")
                            mrc_model_path = relative_model_path
                
                    # MRC 재랭커 인스턴스 생성
                    logger.debug("MRCReranker.get_instance 호출 시작")
                    try:
                        logger.debug(f"최종 MRC 설정 파일 경로: {mrc_config_path}")
                        logger.debug(f"최종 MRC 모델 파일 경로: {mrc_model_path}")
                        
                        # NumPy 버전 체크 (디버깅용)
                        try:
                            import numpy
                            logger.info(f"NumPy 버전: {numpy.__version__}")
                        except Exception as e:
                            logger.warning(f"NumPy 버전 확인 실패: {str(e)}")
                            
                        # PyTorch 버전 체크 (디버깅용)
                        try:
                            import torch
                            logger.info(f"PyTorch 버전: {torch.__version__}")
                        except Exception as e:
                            logger.warning(f"PyTorch 버전 확인 실패: {str(e)}")
                            
                        # TorchText 버전 체크 (디버깅용)
                        try:
                            import torchtext
                            logger.info(f"TorchText 버전: {torchtext.__version__}")
                        except Exception as e:
                            logger.warning(f"TorchText 버전 확인 실패: {str(e)}")
                            
                        # PyTorch Lightning 버전 체크 (디버깅용)
                        try:
                            import pytorch_lightning
                            logger.info(f"PyTorch Lightning 버전: {pytorch_lightning.__version__}")
                        except Exception as e:
                            logger.warning(f"PyTorch Lightning 버전 확인 실패: {str(e)}")
                            
                        # Munch 버전 체크 (디버깅용)
                        try:
                            import munch
                            logger.info(f"Munch 버전: {munch.__version__ if hasattr(munch, '__version__') else '알 수 없음'}")
                        except Exception as e:
                            logger.warning(f"Munch 버전 확인 실패: {str(e)}")
                        
                        self.mrc_reranker = MRCReranker.get_instance(mrc_config_path, mrc_model_path)
                        logger.info("MRC 재랭커 초기화 완료")
                        logger.debug(f"MRC 재랭커 객체: {self.mrc_reranker}")
                    except Exception as inner_e:
                        logger.error(f"MRCReranker.get_instance 호출 실패: {str(inner_e)}", exc_info=True)
                        raise inner_e
                except Exception as e:
                    logger.error(f"MRC 재랭커 초기화 실패: {str(e)}", exc_info=True)
                    logger.error(f"상세 오류 정보: {type(e).__name__}", exc_info=True)
                    self.mrc_enabled = False
            elif self.mrc_enabled and not MRC_AVAILABLE:
                logger.warning("MRC 모듈을 가져올 수 없어 MRC 재랭킹이 비활성화됩니다")
                logger.warning(f"Python 경로: {sys.path}")
                self.mrc_enabled = False
                
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
            "batch_size": default_batch_size,
            "mrc": {
                "enabled": False
            }
        }
        
        if not config_path:
            logger.warning("No config path provided, using default configuration")
            return default_config
            
        try:
            # 파일 존재 여부 확인
            if not os.path.exists(config_path):
                logger.warning(f"Config file not found at {config_path}")
                
                # 상대 경로 시도
                if config_path.startswith("/reranker/"):
                    relative_path = config_path[10:]  # "/reranker/" 제거
                    if os.path.exists(relative_path):
                        logger.info(f"Using relative path instead: {relative_path}")
                        config_path = relative_path
                    else:
                        logger.warning(f"Config file not found at relative path {relative_path} either")
                        return default_config
                else:
                    return default_config
            
            logger.info(f"Loading config from {config_path}")
            with open(config_path, 'r') as f:
                config = json.load(f)
                logger.debug(f"Loaded config content: {json.dumps(config)}")
                
                # MRC 설정 검증 및 디버그 로깅
                if "mrc" in config:
                    logger.info(f"MRC 설정 확인: {json.dumps(config['mrc'])}")
                    
                    # 파일 경로 변환 (절대 경로에서 상대 경로로)
                    mrc_config = config.get("mrc", {})
                    config_path = mrc_config.get("model_config_path")
                    model_path = mrc_config.get("model_ckpt_path")
                    
                    if config_path and config_path.startswith("/reranker/"):
                        relative_path = config_path[10:]
                        if os.path.exists(relative_path):
                            logger.info(f"MRC 설정 파일 상대 경로로 변환: {config_path} -> {relative_path}")
                            config["mrc"]["model_config_path"] = relative_path
                            
                    if model_path and model_path.startswith("/reranker/"):
                        relative_path = model_path[10:]
                        if os.path.exists(relative_path):
                            logger.info(f"MRC 모델 파일 상대 경로로 변환: {model_path} -> {relative_path}")
                            config["mrc"]["model_ckpt_path"] = relative_path
                
                # GPU 여부에 따라 배치 사이즈 선택
                if isinstance(config.get("batch_size"), dict):
                    mode = "gpu" if torch.cuda.is_available() else "cpu"
                    config["batch_size"] = config["batch_size"].get(mode, default_batch_size[mode])
                elif isinstance(config.get("batch_size"), (int, str)):
                    # 이전 형식의 설정을 위한 하위 호환성 유지
                    config["batch_size"] = int(config["batch_size"])
                    
                # 설정 병합 및 반환
                merged_config = {**default_config, **config}
                logger.debug(f"Final merged config: {json.dumps(merged_config)}")
                return merged_config
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}", exc_info=True)
            logger.warning("Using default configuration due to error")
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
            
            # 캐시 사용하지 않음 (디버깅 및 테스트용으로 비활성화)
            log_step("캐시 사용 안함")
            
            # 재랭킹 메소드 결정 (환경변수로 제어 가능)
            rerank_method = os.getenv("RERANK_METHOD", "auto").lower()
            
            # 'auto' 모드: MRC가 활성화되어 있으면 하이브리드, 아니면 FlashRank만 사용
            if rerank_method == "auto":
                rerank_method = "hybrid" if self.mrc_enabled and self.mrc_reranker else "flashrank"
            
            # 재랭킹 수행
            if rerank_method == "mrc" and self.mrc_enabled and self.mrc_reranker:
                # MRC 방식만 사용
                logger.info("MRC 방식으로 재랭킹 수행")
                return self.mrc_reranker.process_search_results(query, search_result, top_k)
                
            elif rerank_method == "hybrid" and self.mrc_enabled and self.mrc_reranker:
                # 하이브리드 방식 (FlashRank + MRC)
                logger.info("하이브리드 방식으로 재랭킹 수행")
                logger.info(f"MRC 모듈 활성화 상태: {self.mrc_enabled}, MRC 리랭커 객체 존재: {self.mrc_reranker is not None}")
                
                if not self.mrc_enabled or self.mrc_reranker is None:
                    logger.error("MRC 모듈이 비활성화되었거나 초기화되지 않았습니다. 하이브리드 재랭킹을 수행할 수 없습니다.")
                    logger.error("MRC 모델 파일이 올바르게 설치되었는지 확인하세요: /reranker/models/mrc/config.json, /reranker/models/mrc/model.ckpt")
                    # FlashRank 방식으로 폴백
                    logger.info("FlashRank 방식으로 대체 수행합니다.")
                    return self.perform_flashrank_reranking(query, passages, top_k, search_result)
                
                try:
                    # FlashRank 재랭킹 수행
                    logger.debug("FlashRank 재랭킹 시작")
                    flashrank_result = self.perform_flashrank_reranking(query, passages, top_k)
                    flashrank_scores = [p.get("score", 0.0) for p in flashrank_result["results"]]
                    logger.debug(f"FlashRank 재랭킹 완료, 결과 수: {len(flashrank_scores)}")
                    
                    # 하이브리드 재랭킹 수행
                    logger.debug("MRC 하이브리드 재랭킹 시작")
                    hybrid_start_time = time.time()
                    
                    # MRC 리랭커 존재 여부 확인 (불필요한 예외 방지)
                    if not hasattr(self.mrc_reranker, 'hybrid_rerank'):
                        logger.error("MRC 리랭커에 hybrid_rerank 메소드가 없습니다.")
                        raise AttributeError("hybrid_rerank method missing in MRC reranker")
                    
                    reranked_passages, mrc_scores = self.mrc_reranker.hybrid_rerank(
                        query, 
                        flashrank_result["results"], 
                        flashrank_scores, 
                        weight_mrc=self.hybrid_weight_mrc,
                        top_k=top_k,
                        return_mrc_scores=True  # MRC 점수도 함께 반환
                    )
                    mrc_processing_time = time.time() - hybrid_start_time
                    logger.debug(f"MRC 하이브리드 재랭킹 완료, 소요 시간: {mrc_processing_time:.3f}초, 결과 수: {len(reranked_passages)}")
                    
                    # MRC 점수 확인
                    logger.info(f"MRC 점수 샘플 (최대 3개): {mrc_scores[:3]}")
                    
                except Exception as e:
                    logger.error(f"하이브리드 재랭킹 중 오류 발생: {str(e)}", exc_info=True)
                    logger.error("FlashRank 방식으로 대체 수행합니다.")
                    return self.perform_flashrank_reranking(query, passages, top_k, search_result)
                
                # 결과에 세부 점수 추가
                for i, passage in enumerate(reranked_passages):
                    if "metadata" not in passage:
                        passage["metadata"] = {}
                    
                    # 원본 메타데이터 유지하면서 세부 점수 추가
                    metadata = passage.get("metadata", {})
                    metadata.update({
                        "flashrank_score": float(flashrank_scores[i]) if i < len(flashrank_scores) else 0.0,
                        "mrc_score": float(mrc_scores[i]) if i < len(mrc_scores) else 0.0
                    })
                    passage["metadata"] = metadata
                
                # 결과 포맷팅
                result = {
                    "query": query,
                    "results": reranked_passages,
                    "total": len(reranked_passages),
                    "reranked": True,
                    "reranker_type": "hybrid",  # hybrid로 명확하게 표시
                    "processing_time": time.time() - start_time,
                    "flashrank_time": flashrank_result.get("processing_time", 0.0),
                    "mrc_time": mrc_processing_time,
                    "mrc_weight": self.hybrid_weight_mrc
                }
                
                return result
                
            else:
                # 기본 FlashRank 방식
                logger.info("FlashRank 방식으로 재랭킹 수행")
                result = self.perform_flashrank_reranking(query, passages, top_k, search_result)
                
                # 캐시 사용하지 않음 (디버깅 및 테스트용으로 비활성화)
                log_step("캐시 업데이트 안함")
                
                # 최종 단계 로깅
                log_step("최종 결과 준비")
                
                # 최종 결과 및 성능 측정 로그
                detailed_steps = "\n".join([
                    f"  - {step['name']}: {step['time']*1000:.2f}ms ({step['elapsed']*1000:.2f}ms elapsed)"
                    for step in timestamps["steps"]
                ])
                logger.debug(f"Detailed timing:\n{detailed_steps}")
                
                return result
                
        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            # 오류 발생 시 원본 결과 반환
            return search_result
    
    def perform_flashrank_reranking(self, query: str, passages: List[Dict], top_k: int = None, search_result: Dict = None):
        """FlashRank를 사용한 재랭킹 수행"""
        try:
            # 시작 시간 기록
            start_time = time.time()
            
            # 대량 패시지 처리 최적화
            total_passages = len(passages)
            logger.info(f"Reranking {total_passages} passages for query: '{query}'")
            
            # 배치 처리를 위한 최적 크기 계산
            batch_size = min(self.batch_size, total_passages)
            logger.debug(f"Using batch size: {batch_size} for {total_passages} passages")
            
            # 동기화 시간 측정을 위한 변수 초기화
            sync_time = 0
            
            # CUDA 동기화 함수
            def sync_cuda():
                if torch.cuda.is_available():
                    sync_start = time.time()
                    # 모든 CUDA 스트림 동기화 (완전한 동기화 보장)
                    torch.cuda.synchronize()
                    # 메모리 캐시 클리어 (메모리 누수 방지)
                    torch.cuda.empty_cache()
                    nonlocal sync_time
                    sync_time += time.time() - sync_start
            
            # 배치 처리 전 GPU 상태 확인 및 초기화
            log_gpu_memory("배치 처리 전")
            
            # CUDA 완전 초기화 및 동기화
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # 시작 전 캐시 클리어
            sync_cuda()
            
            # 배치 크기 최적화 - 더 작은 배치로 분할 처리
            batch_size = min(16, batch_size)  # 배치 크기 제한
            logger.debug(f"Using optimized batch size: {batch_size} for {total_passages} passages")
            
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
                                    
                                    # Rerank passages (GPU 접근 제한)
                                    with record_function("rerank_call"):
                                        with self._gpu_semaphore:  # 최대 7개 동시 GPU 접근
                                            # GPU 사용량 모니터링
                                            if torch.cuda.is_available():
                                                current_memory = torch.cuda.memory_allocated()/1024**2
                                                reserved_memory = torch.cuda.memory_reserved()/1024**2
                                                total_memory = torch.cuda.get_device_properties(0).total_memory/1024**2
                                                active_count = self.max_gpu_workers - self._gpu_semaphore._value  # 현재 활성 GPU 작업 수
                                                logger.debug(f"[GPU] 활성작업: {active_count}/{self.max_gpu_workers}, 사용메모리: {current_memory:.1f}MB, 예약메모리: {reserved_memory:.1f}MB, 총메모리: {total_memory:.1f}MB")
                                            
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
                                    with self._gpu_semaphore:  # 최대 7개 동시 GPU 접근
                                        # GPU 사용량 모니터링
                                        if torch.cuda.is_available():
                                            current_memory = torch.cuda.memory_allocated()/1024**2
                                            reserved_memory = torch.cuda.memory_reserved()/1024**2
                                            total_memory = torch.cuda.get_device_properties(0).total_memory/1024**2
                                            active_count = self.max_gpu_workers - self._gpu_semaphore._value  # 현재 활성 GPU 작업 수
                                            logger.debug(f"[GPU] 활성작업: {active_count}/{self.max_gpu_workers}, 사용메모리: {current_memory:.1f}MB, 예약메모리: {reserved_memory:.1f}MB, 총메모리: {total_memory:.1f}MB")
                                        
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
                            
                            # Rerank passages (GPU 접근 제한)
                            with self._gpu_semaphore:  # 최대 7개 동시 GPU 접근
                                # GPU 사용량 모니터링
                                if torch.cuda.is_available():
                                    current_memory = torch.cuda.memory_allocated()/1024**2
                                    reserved_memory = torch.cuda.memory_reserved()/1024**2
                                    total_memory = torch.cuda.get_device_properties(0).total_memory/1024**2
                                    active_count = self.max_gpu_workers - self._gpu_semaphore._value  # 현재 활성 GPU 작업 수
                                    logger.debug(f"[GPU] 활성작업: {active_count}/{self.max_gpu_workers}, 사용메모리: {current_memory:.1f}MB, 예약메모리: {reserved_memory:.1f}MB, 총메모리: {total_memory:.1f}MB")
                                
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
                        
                        # Rerank passages (GPU 접근 제한)
                        with self._gpu_semaphore:  # 최대 7개 동시 GPU 접근
                            # GPU 사용량 모니터링
                            if torch.cuda.is_available():
                                current_memory = torch.cuda.memory_allocated()/1024**2
                                reserved_memory = torch.cuda.memory_reserved()/1024**2
                                total_memory = torch.cuda.get_device_properties(0).total_memory/1024**2
                                active_count = self.max_gpu_workers - self._gpu_semaphore._value  # 현재 활성 GPU 작업 수
                                logger.debug(f"[GPU] 활성작업: {active_count}/{self.max_gpu_workers}, 사용메모리: {current_memory:.1f}MB, 예약메모리: {reserved_memory:.1f}MB, 총메모리: {total_memory:.1f}MB")
                            
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
                
                if search_result:
                    return search_result
                else:
                    return passages
            
            # Convert back to original format if needed
            if search_result:
                processed_results = []
                for result in reranked_results:
                    # 기본 필드만 포함 (중요한 데이터 유지)
                    processed_result = {
                        "passage_id": result["id"],
                        "doc_id": result["meta"].get("doc_id", ""),
                        "text": result["text"],
                        "score": float(result["score"])
                    }
                    
                    # 메타데이터에서 original_score만 보존
                    if "original_score" in result["meta"]:
                        metadata = {"original_score": result["meta"]["original_score"]}
                        processed_result["metadata"] = metadata
                    
                    processed_results.append(processed_result)
                
                # Apply top_k if specified
                if top_k is not None:
                    processed_results = processed_results[:top_k]
                
                # 결과 준비 - 핵심 필드만 포함
                result = {
                    "query": query,
                    "results": processed_results,
                    "total": len(processed_results),
                    "reranked": True,
                    "reranker_type": "flashrank",
                    "processing_time": time.time() - start_time,
                    "cached": False
                }
                
                # CUDA 최종 동기화
                sync_cuda()
                
                # 성능 로그 기록
                elapsed_time = time.time() - start_time
                logger.info(f"Reranking completed in {elapsed_time:.3f} seconds for {total_passages} passages")
                logger.info(f"CUDA synchronization overhead: {sync_time:.3f} seconds")
                logger.info(f"Effective throughput: {total_passages / (elapsed_time - sync_time):.1f} passages/second")
                
                # 최종 GPU 메모리 상태 로깅
                log_gpu_memory("재랭킹 완료")
                
                return result
            else:
                # 단순히 재랭크된 패시지 리스트 반환
                if top_k is not None:
                    reranked_results = reranked_results[:top_k]
                return reranked_results
                
        except Exception as e:
            logger.error(f"Error in perform_flashrank_reranking: {str(e)}", exc_info=True)
            if search_result:
                return search_result
            else:
                return passages 