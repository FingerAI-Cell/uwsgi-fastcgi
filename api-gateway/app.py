from flask import Flask, request, jsonify, Response
import requests
import requests_unixsocket
import json
import logging
import os
from urllib.parse import quote_plus

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler('api-gateway.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# 서비스 엔드포인트 설정
RAG_ENDPOINT = os.getenv('RAG_ENDPOINT', 'http://milvus-rag:5000')
RERANKER_ENDPOINT = os.getenv('RERANKER_ENDPOINT', 'http://milvus-reranker:8000')

# Unix 소켓 세션 생성
if RAG_ENDPOINT.startswith('unix://'):
    rag_session = requests_unixsocket.Session()
else:
    rag_session = requests.Session()

@app.route('/enhanced-search', methods=['GET'])
def enhanced_search():
    """
    통합 검색 API: RAG 검색 결과를 Reranker로 순위를 다시 매기는 기능
    
    Query Parameters:
        query_text: 검색할 텍스트
        top_k: 반환할 상위 결과 수 (기본값: 5)
        raw_results: 전체 검색 결과 수 (기본값: 20)
        domain: 검색할 도메인 (선택 사항)
        author: 작성자 필터 (선택 사항)
        start_date: 시작 날짜 YYYYMMDD (선택 사항)
        end_date: 종료 날짜 YYYYMMDD (선택 사항)
        title: 제목 검색 (선택 사항)
    """
    try:
        # 파라미터 추출
        query_text = request.args.get('query_text')
        top_k = int(request.args.get('top_k', 5))
        raw_results = int(request.args.get('raw_results', 20))
        
        # 필수 파라미터 검증
        if not query_text:
            return jsonify({
                "result_code": "E000001",
                "message": "검색어(query_text)는 필수 입력값입니다.",
                "search_result": None
            }), 400
            
        # Step 1: RAG 서비스에 검색 요청
        search_params = {}
        
        # 기본 파라미터 추가
        search_params['query_text'] = query_text
        search_params['top_k'] = raw_results  # 초기 검색에서는 더 많은 결과 요청
        
        # 선택적 파라미터 추가
        for param in ['domain', 'author', 'start_date', 'end_date', 'title']:
            if request.args.get(param):
                search_params[param] = request.args.get(param)
                
        # RAG 서비스 호출
        logger.info(f"검색 요청: {search_params}")
        if RAG_ENDPOINT.startswith('unix://'):
            logger.info("🟢 Unix socket 모드로 요청을 시도합니다")
            # Unix 소켓을 통한 요청
            socket_path = RAG_ENDPOINT.replace('unix://', '')
            encoded_socket_path = quote_plus(socket_path)
            rag_response = rag_session.get(f'http+unix://{encoded_socket_path}/search', params=search_params)
        else:
            logger.info("🔵 HTTP 모드로 요청을 시도합니다")
            # HTTP를 통한 요청
            rag_response = requests.get(f"{RAG_ENDPOINT}/search", params=search_params)
        
        if rag_response.status_code != 200:
            logger.error(f"RAG 서비스 오류: {rag_response.text}")
            return jsonify({
                "result_code": "E000002",
                "message": f"검색 서비스 오류: {rag_response.status_code}",
                "search_result": None
            }), 500
            
        rag_data = rag_response.json()
        
        # 검색 결과가 없으면 빈 결과 반환
        if not rag_data.get('search_result') or len(rag_data['search_result']) == 0:
            return jsonify({
                "result_code": "E000003",
                "message": "검색 결과가 없습니다.",
                "search_result": []
            }), 200
            
        # Step 2: Reranker 서비스에 결과 전달
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
            
        # Reranker 서비스 호출
        logger.info(f"Reranker 요청: {len(rerank_data['results'])} 결과")
        rerank_response = requests.post(
            f"{RERANKER_ENDPOINT}/reranker/rerank?top_k={top_k}", 
            json=rerank_data
        )
        
        if rerank_response.status_code != 200:
            logger.error(f"Reranker 서비스 오류: {rerank_response.text}")
            # Reranker 실패 시 원본 RAG 결과 사용
            return Response(
                json.dumps({
                    "result_code": "E000004",
                    "message": "재랭킹에 실패했습니다. 원본 검색 결과를 반환합니다.",
                    "search_result": rag_data['search_result'][:top_k],
                    "reranked": False
                }, ensure_ascii=False),
                content_type="application/json; charset=utf-8"
            )
            
        # Step 3: Reranker 결과 처리
        reranked_data = rerank_response.json()
        
        # Reranker 결과를 원본 형식으로 변환
        final_results = []
        for result in reranked_data.get('results', [])[:top_k]:
            # 원본 메타데이터 복원
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
            
        # 최종 응답 반환
        response_data = {
            "result_code": "E000000",
            "message": "검색 및 재랭킹이 성공적으로 완료되었습니다.",
            "search_params": {
                "query_text": query_text,
                "top_k": top_k,
                "filters": {param: search_params[param] for param in search_params if param not in ['query_text', 'top_k']}
            },
            "search_result": final_results,
            "reranked": True
        }
        
        return Response(
            json.dumps(response_data, ensure_ascii=False),
            content_type="application/json; charset=utf-8"
        )
        
    except Exception as e:
        logger.error(f"통합 검색 오류: {str(e)}")
        return jsonify({
            "result_code": "E000005",
            "message": f"통합 검색 중 오류가 발생했습니다: {str(e)}",
            "search_result": None
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """상태 확인 API"""
    return jsonify({"status": "ok", "service": "api-gateway"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True) 