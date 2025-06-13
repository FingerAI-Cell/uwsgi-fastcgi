"""
Reranker FastCGI application with API Gateway functionality
"""

import os
import logging
import requests
import requests_unixsocket
import traceback
from typing import Dict, List, Any, Optional
from flask import Flask, request, Response, jsonify
from pydantic import BaseModel, Field
from urllib.parse import quote_plus
import time
import sys

# 로깅 설정
log_dir = "/var/log/reranker"
if not os.path.exists(log_dir):
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception as e:
        print(f"로그 디렉토리 생성 실패: {str(e)}")
        log_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"대체 로그 디렉토리 사용: {log_dir}")

try:
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # 스트림 핸들러 설정
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # 파일 핸들러 설정
    try:
        log_file = os.path.join(log_dir, 'app.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        print(f"로그 파일 생성 성공: {log_file}")
    except Exception as e:
        print(f"로그 파일 핸들러 설정 실패: {str(e)}")
        # 파일 핸들러 설정 실패시 스트림 핸들러만 사용
except Exception as e:
    print(f"로깅 설정 실패: {str(e)}")
    # 기본 로깅 사용
    import logging
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)

# 더 빠른 JSON 처리를 위해 ujson 사용
try:
    import ujson as json
    logger.info("Using ujson for faster JSON processing")
except ImportError:
    import json
    logger.info("ujson not available, using default json")

# 상대 경로 import 대신 절대 경로 import로 변경
try:
    from service import RerankerService
    logger.info("RerankerService 임포트 성공")
except Exception as e:
    logger.error(f"RerankerService 임포트 실패: {str(e)}")

# 데이터 모델 정의
class PassageModel(BaseModel):
    """Single passage model"""
    passage_id: Optional[Any] = None
    doc_id: Optional[str] = None
    text: str
    score: Optional[float] = None
    position: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class SearchResultModel(BaseModel):
    """Search result containing multiple passages"""
    query: str
    results: List[PassageModel]
    total: Optional[int] = None
    reranked: Optional[bool] = False


class RerankerResponseModel(BaseModel):
    """Response model for reranker API"""
    query: str
    results: List[PassageModel]
    total: int
    reranked: bool = True


# Flask 앱 생성
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# 성능 최적화 설정 추가
app.config['PROPAGATE_EXCEPTIONS'] = True
app.config['PREFERRED_URL_SCHEME'] = 'http'
app.config['JSON_SORT_KEYS'] = False  # JSON 정렬 비활성화로 성능 향상
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 최대 요청 크기
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False  # 압축 JSON 응답
app.config['JSONIFY_MIMETYPE'] = 'application/json; charset=utf-8'  # 명시적 MIME 타입

# 응답 압축 비활성화 - 대용량 응답 처리 시 압축으로 인한 지연 방지
# Flask-Compress 사용하지 않음
logger.info("Response compression disabled for better performance")

# WSGI 응답 버퍼링 비활성화
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['wsgi.response_buffering'] = 'false'

# 스레드 최적화
import threading
threading.stack_size(128 * 1024)  # 스레드 스택 크기 감소

# 응답 속도 최적화를 위한 설정
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TRAP_HTTP_EXCEPTIONS'] = False
app.config['PRESERVE_CONTEXT_ON_EXCEPTION'] = False

# 서비스 인스턴스 생성
reranker_service = None

# RAG 서비스 엔드포인트 설정
RAG_ENDPOINT = os.getenv('RAG_ENDPOINT', 'http://nginx/rag')

# Unix 소켓 세션 생성 및 연결 풀링 최적화
rag_session = requests.Session()
adapter = requests.adapters.HTTPAdapter(
    pool_connections=100,  # 연결 풀 크기
    pool_maxsize=100,      # 최대 연결 수
    max_retries=3,         # 재시도 횟수
    pool_block=False       # 논블로킹 모드
)
rag_session.mount('http://', adapter)
rag_session.mount('https://', adapter)

# FastCGI 응답 헤더 설정
@app.after_request
def add_header(response):
    """응답 헤더 최적화"""
    # 필수 헤더만 설정하여 오버헤드 최소화
    response.headers['X-Content-Type-Options'] = 'nosniff'
    
    # FastCGI 응답 최적화 - 핵심 헤더
    response.headers['X-Accel-Buffering'] = 'no'  # nginx 버퍼링 비활성화
    
    # Transfer-Encoding 최적화
    response.headers.pop('Transfer-Encoding', None)
    
    # 콘텐츠 길이 명시 (chunked 인코딩 방지)
    if response.data and 'Content-Length' not in response.headers:
        response.headers['Content-Length'] = str(len(response.data))
    
    return response


def get_reranker_service():
    """Get reranker service instance"""
    try:
        # 절대 경로로 설정된 환경 변수 확인
        config_path = os.environ.get("RERANKER_CONFIG", "/reranker/config.json")
        
        # 절대 경로에서 파일을 찾지 못한 경우 상대 경로로 시도
        if not os.path.exists(config_path) and config_path.startswith("/reranker/"):
            relative_config_path = config_path[10:]  # "/reranker/" 제거
            if os.path.exists(relative_config_path):
                logger.info(f"환경 변수의 절대 경로를 상대 경로로 변환: {config_path} -> {relative_config_path}")
                config_path = relative_config_path
                
        # 그래도 파일이 없으면 현재 디렉토리에서 config.json 찾기
        if not os.path.exists(config_path):
            if os.path.exists("config.json"):
                logger.info(f"환경 변수 대신 현재 디렉토리의 config.json 사용")
                config_path = "config.json"
                
        logger.info(f"Getting RerankerService with config: {config_path} (exists: {os.path.exists(config_path)})")
        
        # 싱글톤 패턴으로 서비스 인스턴스 가져오기
        from service import RerankerService
        return RerankerService.get_instance(config_path)
    except Exception as e:
        logger.error(f"Failed to get RerankerService: {str(e)}", exc_info=True)
        logger.error("Using dummy reranker for testing")
        return DummyReranker()


class DummyReranker:
    """테스트용 더미 리랭커"""
    def __init__(self):
        """초기화 메서드"""
        self.hybrid_weight_mrc = 0.7  # 기본값으로 설정
        self.mrc_enabled = False
        self.mrc_reranker = None
        
    def process_search_results(self, query: str, search_result: Dict, top_k: int = 5) -> Dict:
        """원본 검색 결과를 그대로 반환"""
        logger.warning("Using dummy reranker - returning original search results")
        return search_result


# 애플리케이션 시작 시 서비스 초기화
@app.before_first_request
def initialize_service():
    """Initialize service before first request"""
    try:
        get_reranker_service()
        logger.info("Service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize service: {str(e)}")
        # 초기화 실패해도 서비스는 계속 실행
        pass


@app.route("/reranker/health")
def health_check():
    """Health check endpoint"""
    return Response(
        json.dumps({"status": "ok", "service": "reranker"}, ensure_ascii=False),
        mimetype='application/json; charset=utf-8'
    )


@app.route("/reranker/enhanced-search", methods=['GET'])
def enhanced_search():
    """
    통합 검색 API: RAG 검색 결과를 Reranker로 순위를 다시 매기는 기능
    """
    try:
        # 전체 요청 처리 시간 측정 시작
        total_start_time = time.time()
        
        # 파라미터 추출
        query_text = request.args.get('query_text')
        top_k = int(request.args.get('top_k', 5))
        raw_results = int(request.args.get('raw_results', 20))
        
        # 필수 파라미터 검증
        if not query_text:
            return jsonify({
                "result_code": "F000001",
                "message": "검색어(query_text)는 필수 입력값입니다.",
                "search_result": None
            }), 400
            
        # Step 1: RAG 서비스에 검색 요청
        search_params = {
            'query_text': query_text,
            'top_k': raw_results
        }
        
        # 선택적 파라미터 추가
        for param in ['domain', 'author', 'start_date', 'end_date', 'title']:
            if request.args.get(param):
                search_params[param] = request.args.get(param)
                
        # RAG 서비스 호출
        logger.info(f"검색 요청: {search_params}")
        rag_response = requests.get(f"{RAG_ENDPOINT}/search", params=search_params)
        
        if rag_response.status_code != 200:
            logger.error(f"RAG 서비스 오류: {rag_response.text}")
            return jsonify({
                "result_code": "F000002",
                "message": f"검색 서비스 오류: {rag_response.status_code}",
                "search_result": None
            }), 500
            
        rag_data = rag_response.json()
        
        # 검색 결과가 없으면 빈 결과 반환
        if not rag_data.get('search_result') or len(rag_data['search_result']) == 0:
            return jsonify({
                "result_code": "F000003",
                "message": "검색 결과가 없습니다.",
                "search_result": []
            }), 200
            
        # Step 2: Reranker 처리를 위한 데이터 준비
        rerank_data = {
            "query": query_text,
            "results": []
        }
        
        # RAG 결과를 Reranker 포맷으로 변환
        for idx, result in enumerate(rag_data['search_result']):
            passage = {
                "passage_id": idx,
                "doc_id": result.get('doc_id'),
                "text": result.get('text'),
                "score": result.get('score'),
                "metadata": {
                    "title": result.get('title'),
                    "author": result.get('author'),
                    "info": result.get('info'),
                    "tags": result.get('tags')
                }
            }
            rerank_data["results"].append(passage)
            
        # Step 3: Reranker 처리
        try:
            search_result = SearchResultModel(**rerank_data)
            reranked = get_reranker_service().process_search_results(
                search_result.query,
                search_result.dict(),
                top_k
            )
            
            # 최종 결과 변환
            final_results = []
            for result in reranked.get('results', [])[:top_k]:
                metadata = result.get('metadata', {})
                final_result = {
                    "doc_id": result.get('doc_id'),
                    "text": result.get('text'),
                    "score": result.get('score'),
                    "title": metadata.get('title'),
                    "author": metadata.get('author'),
                    "info": metadata.get('info'),
                    "tags": metadata.get('tags')
                }
                final_results.append(final_result)
            
            # 전체 요청 처리 시간 계산
            total_elapsed_time = time.time() - total_start_time
            logger.info(f"Total enhanced-search endpoint processing time: {total_elapsed_time:.3f} seconds")
                
            response_data = {
                "result_code": "F000000",
                "message": "검색 및 재랭킹이 성공적으로 완료되었습니다.",
                "search_params": {
                    "query_text": query_text,
                    "top_k": top_k,
                    "filters": {param: search_params[param] for param in search_params if param not in ['query_text', 'top_k']}
                },
                "search_result": final_results,
                "total_processing_time": total_elapsed_time,
                "reranking_time": reranked.get("processing_time", 0)
            }
            
            return Response(json.dumps(response_data, ensure_ascii=False), 
                          content_type="application/json; charset=utf-8")
                          
        except Exception as e:
            logger.error(f"재랭킹 처리 오류: {str(e)}")
            return jsonify({
                "result_code": "F000004",
                "message": f"재랭킹 처리 중 오류가 발생했습니다: {str(e)}",
                "search_result": None
            }), 500
            
    except Exception as e:
        logger.error(f"통합 검색 오류: {str(e)}")
        return jsonify({
            "result_code": "F000005",
            "message": f"통합 검색 중 오류가 발생했습니다: {str(e)}",
            "search_result": None
        }), 500


@app.route("/reranker/rerank", methods=['POST'])
def rerank():
    """
    Rerank passages endpoint
    """
    try:
        # 전체 요청 처리 시간 측정 시작
        total_start_time = time.time()
        
        # Get top_k parameter from query string
        top_k = request.args.get('top_k', type=int)
        
        # Get reranker type parameter (flashrank, mrc, hybrid)
        rerank_type = request.args.get('type', 'auto').lower()
        
        # Set environment variable for reranker method
        os.environ["RERANK_METHOD"] = rerank_type
        
        # Get request body
        data = request.get_json()
        if not data:
            return jsonify({
                "error": "No JSON data provided"
            }), 400
            
        # Validate input
        try:
            search_result = SearchResultModel(**data)
        except Exception as e:
            return jsonify({
                "error": f"Invalid input format: {str(e)}"
            }), 400
            
        # Process reranking
        reranked = get_reranker_service().process_search_results(
            search_result.query,
            search_result.dict(),
            top_k
        )
        
        # 전체 요청 처리 시간 계산
        processing_time = time.time() - total_start_time
        # API 명세에 맞게 processing_time 필드 추가
        reranked["processing_time"] = processing_time
        
        logger.info(f"Total rerank endpoint processing time: {processing_time:.3f} seconds")
        
        # 최적화된 응답 생성
        response_data = json.dumps(reranked, ensure_ascii=False)
        response = Response(
            response_data,
            mimetype='application/json; charset=utf-8'
        )
        
        # FastCGI 응답 지연 해결을 위한 핵심 헤더 설정
        response.headers['X-Accel-Buffering'] = 'no'
        response.headers['Content-Length'] = str(len(response.data))
        
        return response
        
    except Exception as e:
        logger.error(f"Reranking failed: {str(e)}")
        return jsonify({
            "error": f"Reranking failed: {str(e)}"
        }), 500


@app.route("/reranker/mrc-rerank", methods=['POST'])
def mrc_rerank():
    """
    MRC 기반 재랭킹 엔드포인트
    """
    try:
        # 전체 요청 처리 시간 측정 시작
        total_start_time = time.time()
        
        # Get top_k parameter from query string
        top_k = request.args.get('top_k', type=int)
        
        # MRC 방식으로 강제 설정
        os.environ["RERANK_METHOD"] = "mrc"
        
        # Get request body
        data = request.get_json()
        if not data:
            return jsonify({
                "error": "No JSON data provided"
            }), 400
            
        # Validate input
        try:
            search_result = SearchResultModel(**data)
        except Exception as e:
            return jsonify({
                "error": f"Invalid input format: {str(e)}"
            }), 400
            
        # Process reranking
        reranked = get_reranker_service().process_search_results(
            search_result.query,
            search_result.dict(),
            top_k
        )
        
        # 전체 요청 처리 시간 계산
        processing_time = time.time() - total_start_time
        # API 명세에 맞게 processing_time 필드 추가
        reranked["processing_time"] = processing_time
        
        logger.info(f"Total mrc-rerank endpoint processing time: {processing_time:.3f} seconds")
        
        # 최적화된 응답 생성
        response_data = json.dumps(reranked, ensure_ascii=False)
        response = Response(
            response_data,
            mimetype='application/json; charset=utf-8'
        )
        
        # FastCGI 응답 지연 해결을 위한 핵심 헤더 설정
        response.headers['X-Accel-Buffering'] = 'no'
        response.headers['Content-Length'] = str(len(response.data))
        
        return response
        
    except Exception as e:
        logger.error(f"MRC reranking failed: {str(e)}")
        return jsonify({
            "error": f"MRC reranking failed: {str(e)}"
        }), 500


# MRC 설정 확인 함수
def check_mrc_configuration():
    """MRC 모델 설정 및 파일 존재 여부 확인"""
    try:
        service = get_reranker_service()
        mrc_enabled = service.mrc_enabled if hasattr(service, 'mrc_enabled') else False
        mrc_reranker = service.mrc_reranker if hasattr(service, 'mrc_reranker') else None
        
        # 설정 파일 및 모델 파일 경로
        config_path = None
        model_path = None
        
        if hasattr(service, 'config') and isinstance(service.config, dict):
            mrc_config = service.config.get('mrc', {})
            config_path = mrc_config.get('model_config_path')
            model_path = mrc_config.get('model_ckpt_path')
        
        # 파일 존재 여부 확인
        config_exists = False
        model_exists = False
        
        # 절대 경로로 확인
        if config_path:
            config_exists = os.path.exists(config_path)
            logger.debug(f"절대 경로 MRC 설정 파일 확인: {config_path} -> {config_exists}")
        
        if model_path:
            model_exists = os.path.exists(model_path)
            logger.debug(f"절대 경로 MRC 모델 파일 확인: {model_path} -> {model_exists}")
            
        # 절대 경로에서 파일을 찾지 못한 경우 상대 경로로 시도
        if config_path and not config_exists and config_path.startswith("/reranker/"):
            relative_config_path = config_path[10:]  # "/reranker/" 제거
            relative_config_exists = os.path.exists(relative_config_path)
            logger.debug(f"상대 경로 MRC 설정 파일 확인: {relative_config_path} -> {relative_config_exists}")
            if relative_config_exists:
                config_exists = True
                
        if model_path and not model_exists and model_path.startswith("/reranker/"):
            relative_model_path = model_path[10:]  # "/reranker/" 제거
            relative_model_exists = os.path.exists(relative_model_path)
            logger.debug(f"상대 경로 MRC 모델 파일 확인: {relative_model_path} -> {relative_model_exists}")
            if relative_model_exists:
                model_exists = True
        
        return {
            "mrc_enabled": mrc_enabled,
            "mrc_reranker_loaded": mrc_reranker is not None,
            "config_path": config_path,
            "model_path": model_path,
            "config_exists": config_exists,
            "model_exists": model_exists
        }
    except Exception as e:
        logger.error(f"MRC 설정 확인 중 오류 발생: {str(e)}")
        return {
            "error": str(e),
            "mrc_enabled": False,
            "mrc_reranker_loaded": False,
            "config_exists": False,
            "model_exists": False
        }

@app.route("/reranker/mrc-status", methods=['GET'])
def mrc_status():
    """MRC 모듈 상태 확인 API"""
    try:
        # MRC 설정 확인
        mrc_status = check_mrc_configuration()
        
        return jsonify({
            "status": "ok",
            "mrc_configuration": mrc_status,
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"MRC 상태 확인 API 오류: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": time.time()
        }), 500

@app.route("/reranker/hybrid-rerank", methods=['POST'])
def hybrid_rerank():
    """
    하이브리드 재랭킹 엔드포인트 (FlashRank + MRC)
    """
    try:
        # 전체 요청 처리 시간 측정 시작
        total_start_time = time.time()
        
        # Get top_k parameter from query string
        top_k = request.args.get('top_k', type=int)
        
        # Get mrc weight parameter
        mrc_weight = request.args.get('mrc_weight', type=float)
        
        # 하이브리드 방식으로 강제 설정
        os.environ["RERANK_METHOD"] = "hybrid"
        logger.info("하이브리드 재랭킹 모드로 설정됨")
        
        # MRC 설정 확인 및 로깅
        mrc_config = check_mrc_configuration()
        logger.info(f"MRC 설정 상태: 활성화={mrc_config['mrc_enabled']}, 모델 로드됨={mrc_config['mrc_reranker_loaded']}")
        
        # 필요한 파일 존재 확인
        if not mrc_config['config_exists'] or not mrc_config['model_exists']:
            logger.warning(f"MRC 모델 파일 누락: 설정파일={mrc_config['config_exists']}, 모델파일={mrc_config['model_exists']}")
        
        # Get request body
        data = request.get_json()
        if not data:
            return jsonify({
                "error": "No JSON data provided"
            }), 400
            
        # Validate input
        try:
            search_result = SearchResultModel(**data)
            logger.info(f"재랭킹 요청: query='{search_result.query}', 결과 수={len(search_result.results)}")
        except Exception as e:
            logger.error(f"요청 검증 실패: {str(e)}")
            return jsonify({
                "error": f"Invalid input format: {str(e)}"
            }), 400
            
        # Process reranking
        reranker_service = get_reranker_service()
        
        # MRC 가중치 설정
        if mrc_weight is not None:
            logger.info(f"MRC 가중치 변경: {getattr(reranker_service, 'hybrid_weight_mrc', '기본값')} -> {mrc_weight}")
            reranker_service.hybrid_weight_mrc = mrc_weight
            
        # 재랭킹 처리 시작
        logger.info("하이브리드 재랭킹 처리 시작")
        process_start_time = time.time()
        
        reranked = reranker_service.process_search_results(
            search_result.query,
            search_result.dict(),
            top_k
        )
        
        process_time = time.time() - process_start_time
        logger.info(f"하이브리드 재랭킹 처리 완료: {process_time:.3f}초")
        
        # 전체 요청 처리 시간 계산
        processing_time = time.time() - total_start_time
        
        # API 명세에 맞게 processing_time 필드 추가
        reranked["processing_time"] = processing_time
        reranked["mrc_weight"] = reranker_service.hybrid_weight_mrc
        
        # 재랭커 타입 확인 및 로깅
        reranker_type = reranked.get("reranker_type", "unknown")
        if reranker_type != "hybrid":
            logger.warning(f"하이브리드 재랭킹 요청했으나 결과 타입은 '{reranker_type}'입니다. MRC 설정을 확인하세요.")
        else:
            logger.info(f"하이브리드 재랭킹 성공적으로 완료됨")
        
        logger.info(f"Total hybrid-rerank endpoint processing time: {processing_time:.3f} seconds")
        
        # 최적화된 응답 생성
        response_data = json.dumps(reranked, ensure_ascii=False)
        response = Response(
            response_data,
            mimetype='application/json; charset=utf-8'
        )
        
        # FastCGI 응답 지연 해결을 위한 핵심 헤더 설정
        response.headers['X-Accel-Buffering'] = 'no'
        response.headers['Content-Length'] = str(len(response.data))
        
        return response
        
    except Exception as e:
        logger.error(f"하이브리드 재랭킹 실패: {str(e)}", exc_info=True)
        return jsonify({
            "error": f"Hybrid reranking failed: {str(e)}"
        }), 500


@app.route("/reranker/batch_rerank", methods=["POST"])
def batch_rerank():
    """
    Batch rerank multiple queries and their passages
    
    Returns:
        List of reranked results for each query
    """
    try:
        # 전체 요청 처리 시간 측정 시작
        total_start_time = time.time()
        
        data = request.get_json()
        top_k = request.args.get("top_k", type=int)
        
        # Process each query
        results = []
        for query_data in data:
            search_result = SearchResultModel(**query_data)
            reranked = get_reranker_service().process_search_results(
                search_result.query,
                search_result.dict(),
                top_k
            )
            results.append(reranked)
        
        # 전체 요청 처리 시간 계산
        total_elapsed_time = time.time() - total_start_time
        logger.info(f"Total batch_rerank endpoint processing time: {total_elapsed_time:.3f} seconds")
        
        # 배치 처리 결과에 전체 처리 시간 추가
        batch_result = {
            "results": results,
            "total_processing_time": total_elapsed_time,
            "query_count": len(results)
        }
        
        # 최적화된 응답 생성
        response_data = json.dumps(batch_result, ensure_ascii=False)
        response = Response(
            response_data,
            mimetype='application/json; charset=utf-8'
        )
        
        # FastCGI 응답 지연 해결을 위한 핵심 헤더 설정
        response.headers['X-Accel-Buffering'] = 'no'
        response.headers['Content-Length'] = str(len(response.data))
        
        return response
        
    except Exception as e:
        return Response(
            json.dumps({"error": f"Batch reranking failed: {str(e)}"}, ensure_ascii=False),
            status=500,
            mimetype='application/json; charset=utf-8'
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000) 