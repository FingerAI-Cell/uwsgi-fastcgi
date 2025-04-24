"""
Reranker FastCGI application with API Gateway functionality
"""

import os
import json
import logging
import requests
import requests_unixsocket
from typing import Dict, List, Any, Optional
from flask import Flask, request, Response, jsonify
from pydantic import BaseModel, Field
from urllib.parse import quote_plus

# 로깅 설정
log_dir = "/var/log/reranker"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 스트림 핸들러 설정
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)

# 파일 핸들러 설정
file_handler = logging.FileHandler(os.path.join(log_dir, 'app.log'))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# 핸들러 추가
logger.addHandler(stream_handler)
logger.addHandler(file_handler)

# 상대 경로 import 대신 절대 경로 import로 변경
from service import RerankerService

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

# FastCGI 응답 형식 설정
app.config['PROPAGATE_EXCEPTIONS'] = True
app.config['PREFERRED_URL_SCHEME'] = 'http'

# 서비스 인스턴스 생성
reranker_service = None

# RAG 서비스 엔드포인트 설정
RAG_ENDPOINT = os.getenv('RAG_ENDPOINT', 'http://nginx/rag')

# Unix 소켓 세션 생성
rag_session = requests.Session()

# FastCGI 응답 헤더 설정
@app.after_request
def add_header(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response


def get_reranker_service():
    """Get reranker service instance"""
    global reranker_service
    if reranker_service is None:
        try:
            config_path = os.environ.get("RERANKER_CONFIG", "/reranker/config.json")
            logger.info(f"Initializing RerankerService with config: {config_path}")
            logger.info(f"Current working directory: {os.getcwd()}")
            logger.info(f"Config file exists: {os.path.exists(config_path)}")
            reranker_service = RerankerService(config_path)
            logger.info("RerankerService initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RerankerService: {str(e)}", exc_info=True)
            logger.error("Using dummy reranker for testing")
            reranker_service = DummyReranker()
    return reranker_service


class DummyReranker:
    """테스트용 더미 리랭커"""
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
                
            response_data = {
                "result_code": "F000000",
                "message": "검색 및 재랭킹이 성공적으로 완료되었습니다.",
                "search_params": {
                    "query_text": query_text,
                    "top_k": top_k,
                    "filters": {param: search_params[param] for param in search_params if param not in ['query_text', 'top_k']}
                },
                "search_result": final_results
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
        # Get top_k parameter from query string
        top_k = request.args.get('top_k', type=int)
        
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
        
        return Response(
            json.dumps(reranked, ensure_ascii=False),
            mimetype='application/json; charset=utf-8'
        )
        
    except Exception as e:
        logger.error(f"Reranking failed: {str(e)}")
        return jsonify({
            "error": f"Reranking failed: {str(e)}"
        }), 500


@app.route("/reranker/batch_rerank", methods=["POST"])
def batch_rerank():
    """
    Batch rerank multiple queries and their passages
    
    Returns:
        List of reranked results for each query
    """
    try:
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
        
        return Response(
            json.dumps(results, ensure_ascii=False),
            mimetype='application/json; charset=utf-8'
        )
        
    except Exception as e:
        return Response(
            json.dumps({"error": f"Batch reranking failed: {str(e)}"}, ensure_ascii=False),
            status=500,
            mimetype='application/json; charset=utf-8'
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000) 