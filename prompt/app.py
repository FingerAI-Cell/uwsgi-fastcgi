from flask import Flask, request, jsonify, Response, stream_with_context
import os
import json
import requests
import logging
from datetime import datetime
from typing import Dict, Any

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/var/log/prompt/app.log") if os.path.exists("/var/log/prompt") else logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger("prompt-backend")

# Flask 앱 초기화
app = Flask(__name__)
app.json.ensure_ascii = False  # 한글 인코딩 처리

config_path = os.environ.get("PROMPT_CONFIG", "/prompt/config.json")

# 환경 변수 설정
RAG_ENDPOINT = os.environ.get("RAG_ENDPOINT", "http://nginx/rag")
RERANKER_ENDPOINT = os.environ.get("RERANKER_ENDPOINT", "http://nginx/reranker")

OLLAMA_ENDPOINT = os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434")

class AgentService:
    def __init__(self, config_path:str=None):
        """
        Initialize the prompt Agent service
        
        Args:
            config_path: Path to config file, if None, use default settings
        """
        try:
            logger.debug("Loading configuration...")
            self.config = self._load_config(config_path)
            self.default_model = self.config.get("default_model")
            self.search_top = self.config.get("search_top")
            self.rerank_top = self.config.get("rerank_top")
            self.rerank_threshold = self.config.get("rerank_threshold")
        
            logger.info(f"Initializing Agent LLM with model: {self.default_model}")
            logger.debug(f"RAG Search Top {self.search_top}")
            logger.debug(f"Reranking Top {self.rerank_top}")
            logger.debug(f"Reranking Threshold {self.rerank_threshold}")
        except Exception as e:
            logger.error(f"Failed to initialize AgentService: {str(e)}")
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
            "search_top": int(os.getenv("RAG_SEARCH_TOP_K", "100")),
            "rerank_top": int(os.getenv("RERANKER_TOP_K", "20")),
            "default_model": os.getenv("DEFAULT_MODEL", "mistral"),
            "rerank_threshold": float(os.getenv("RERANK_THRESHOLD", "0.1"))
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
        
    # 프롬프트 템플릿 로드 함수
    @staticmethod
    def load_prompt_template(template_name):
        template_path = os.path.join(os.path.dirname(__file__), "templates", f"{template_name}.txt")
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"템플릿 파일을 찾을 수 없습니다: {template_path}")
            return None


# 상태 확인 API
@app.route("/prompt/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "service": "prompt-backend"
    })

# 문서 검색 및 요약 API
@app.route("/prompt/summarize", methods=["POST"])
def summarize():
    try:
        data = request.json
        logger.info(f"요청 받음: {json.dumps(data, ensure_ascii=False)}")
        query = data.get("query")
        
        if not query:
            logger.error("쿼리 누락")
            return jsonify({"error": "쿼리가 필요합니다"}), 400
        
        summaryAgent = AgentService(config_path)
        logger.info(f"Agent 초기화 완료: search_top={summaryAgent.search_top}, rerank_top={summaryAgent.rerank_top}")
            
        # 1. RAG 서비스 호출하여 문서 검색
        logger.info(f"RAG 서비스 호출 준비: endpoint={RAG_ENDPOINT}/search")
        search_params = {
            "query_text": query,
            "top_k": summaryAgent.search_top,
            "domains": []  # 기본 빈 도메인 리스트
        }
        
        # 추가 검색 매개변수
        if "domain" in data:  # 단일 도메인 지원
            search_params["domains"] = [data["domain"]]
        elif "domains" in data:  # 복수 도메인 지원
            search_params["domains"] = data["domains"]
            
        for param in ["author", "start_date", "end_date", "title", "info_filter", "tags_filter"]:
            if param in data:
                search_params[param] = data[param]
                logger.info(f"추가 검색 파라미터: {param}={data[param]}")
        
        # curl 형식의 API 호출 로깅
        curl_command = f'''curl -X POST "{RAG_ENDPOINT}/search" \\
  -H "Content-Type: application/json" \\
  -d '{json.dumps(search_params, ensure_ascii=False)}\''''
        logger.info(f"RAG API curl 형식: {curl_command}")
        
        logger.info(f"RAG 검색 요청: params={json.dumps(search_params, ensure_ascii=False)}")
        search_response = requests.post(f"{RAG_ENDPOINT}/search", json=search_params)
        
        logger.info(f"RAG 응답 코드: {search_response.status_code}")
        if search_response.status_code != 200:
            logger.error(f"RAG 검색 오류 응답: {search_response.text}")
            return jsonify({"error": "문서 검색 중 오류가 발생했습니다"}), 500
            
        search_results = search_response.json()
        logger.info("=== RAG 검색 결과 ===")
        logger.info(f"검색된 문서 수: {len(search_results.get('search_result', []))}")
        logger.info(f"RAG 응답 결과: {json.dumps(search_results, ensure_ascii=False, indent=2)}")
        for idx, doc in enumerate(search_results.get("search_result", []), 1):
            logger.info(f"문서 {idx}:")
            logger.info(f"제목: {doc.get('title', '제목 없음')}")
            logger.info(f"내용: {doc.get('text', '')[:100]}...")
            logger.info(f"점수: {doc.get('score', 'N/A')}")
            logger.info("---")
        
        # 2. Reranker 서비스 호출
        logger.info(f"Reranker 서비스 호출 준비: endpoint={RERANKER_ENDPOINT}/rerank")
        rerank_data = {
            "query": query,
            "results": search_results.get("search_result", [])
        }
        
        # curl 형식의 API 호출 로깅
        curl_command = f'''curl -X POST "{RERANKER_ENDPOINT}/rerank?top_k={summaryAgent.rerank_top}" \\
  -H "Content-Type: application/json" \\
  -d '{json.dumps(rerank_data, ensure_ascii=False)}\''''
        logger.info(f"Reranker API curl 형식: {curl_command}")
        
        logger.info(f"Reranker 요청: top_k={summaryAgent.rerank_top}")
        rerank_response = requests.post(
            f"{RERANKER_ENDPOINT}/rerank",
            params={"top_k": summaryAgent.rerank_top},
            json=rerank_data
        )
        
        logger.info(f"Reranker 응답 코드: {rerank_response.status_code}")
        if rerank_response.status_code != 200:
            logger.error(f"Reranker 오류 응답: {rerank_response.text}")
            return jsonify({"error": "문서 재순위화 중 오류가 발생했습니다"}), 500
            
        reranked_results = rerank_response.json()
        logger.info("=== Reranker 결과 ===")
        logger.info(f"재순위화된 문서 수: {len(reranked_results['results'])}")
        logger.info(f"Reranker 응답 결과: {json.dumps(reranked_results, ensure_ascii=False, indent=2)}")
        for idx, doc in enumerate(reranked_results["results"][:summaryAgent.rerank_top], 1):
            logger.info(f"재순위화된 문서 {idx}:")
            logger.info(f"제목: {doc.get('title', '제목 없음')}")
            logger.info(f"내용: {doc.get('text', '')[:100]}...")
            logger.info(f"점수: {doc.get('score', 'N/A')}")
            logger.info("---")
        
        # 3. 프롬프트 템플릿 준비
        logger.info("프롬프트 템플릿 로드 시작")
        template = summaryAgent.load_prompt_template("summarize")
        
        if not template:
            logger.error("프롬프트 템플릿 로드 실패")
            return jsonify({"error": "프롬프트 템플릿을 로드할 수 없습니다"}), 500
            
        # 컨텍스트 생성
        logger.info(f"컨텍스트 생성 시작 (문서 수: {len(reranked_results['results'][:summaryAgent.rerank_top])})")
        context = ""
        for idx, doc in enumerate(reranked_results['results'][:summaryAgent.rerank_top], 1):
            context += f"[문서 {idx}]\n"
            context += f"제목: {doc.get('title', '제목 없음')}\n"
            context += f"내용: {doc.get('text', '')}\n\n"
        
        # 최종 프롬프트 생성
        final_prompt = template.format(
            query=query,
            context=context
        )
        logger.info(f"최종 프롬프트 길이: {len(final_prompt)} 문자")
        logger.info("=== 최종 프롬프트 내용 ===")
        logger.info(f"{final_prompt}")
        logger.info("========================")
        
        # 4. Ollama API 호출
        logger.info(f"Ollama API 호출 준비: endpoint={OLLAMA_ENDPOINT}, model={summaryAgent.default_model}")
        try:
            logger.info("Ollama 요청 시작")
            ollama_response = requests.post(
                f"{OLLAMA_ENDPOINT}/api/generate",
                json={
                    "model": summaryAgent.default_model,
                    "prompt": final_prompt,
                    "stream": False
                },
                timeout=120
            )
            
            logger.info(f"Ollama 응답 코드: {ollama_response.status_code}")
            if ollama_response.status_code != 200:
                logger.error(f"Ollama API 오류 응답: {ollama_response.text}")
                return jsonify({
                    "error": "LLM 요청 중 오류가 발생했습니다",
                    "details": ollama_response.text
                }), 500
                
            summary = ollama_response.json().get("response", "")
            logger.info("=== 최종 요약 결과 ===")
            logger.info(f"쿼리: {query}")
            logger.info(f"요약 길이: {len(summary)} 문자")
            logger.info(f"요약 내용: {summary}")
            logger.info("=== 처리 완료 ===")
            
            return jsonify({
                "query": query,
                "summary": summary,
                "documents_count": len(reranked_results['results']),
                "prompt_length": len(final_prompt)
            })
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama 서비스 연결 오류: {str(e)}", exc_info=True)
            return jsonify({
                "error": "Ollama 서비스에 연결할 수 없습니다",
                "details": str(e)
            }), 503
        
    except Exception as e:
        logger.error(f"처리 중 예외 발생: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# 향상된 검색 API (RAG+Reranker)
@app.route("/prompt/enhanced_search", methods=["POST"])
def enhanced_search():
    try:
        data = request.json
        logger.info(f"향상된 검색 요청 받음: {json.dumps(data, ensure_ascii=False)}")
        
        # 필수 파라미터 확인
        query = data.get("query")
        if not query:
            logger.error("쿼리 누락")
            return jsonify({"error": "쿼리가 필요합니다"}), 400
        
        summaryAgent = AgentService(config_path)
        
        # 사용자 지정 파라미터 또는 기본값 사용
        top_m = data.get("top_m", summaryAgent.search_top)  # RAG 검색 결과 수
        top_n = data.get("top_n", summaryAgent.rerank_top)  # Reranker 결과 수
        threshold = data.get("threshold", summaryAgent.rerank_threshold)  # Reranker 점수 임계치
        mrc_weight = data.get("mrc_weight", 0.7)  # MRC 가중치 (기본값 0.7)
        
        # 파라미터 유효성 검사
        if top_m < top_n:
            logger.warning(f"파라미터 오류: top_m({top_m}) < top_n({top_n}), top_m으로 조정합니다")
            top_n = top_m
            
        logger.info(f"검색 파라미터: query='{query}', top_m={top_m}, top_n={top_n}, threshold={threshold}, mrc_weight={mrc_weight}")
            
        # 1. RAG 서비스 호출하여 문서 검색
        logger.info(f"RAG 서비스 호출 준비: endpoint={RAG_ENDPOINT}/search")
        search_params = {
            "query_text": query,
            "top_k": top_m,
            "domains": []  # 기본 빈 도메인 리스트
        }
        
        # 추가 검색 매개변수
        if "domain" in data:  # 단일 도메인 지원
            search_params["domains"] = [data["domain"]]
        elif "domains" in data:  # 복수 도메인 지원
            search_params["domains"] = data["domains"]
            
        for param in ["author", "start_date", "end_date", "title", "info_filter", "tags_filter"]:
            if param in data:
                search_params[param] = data[param]
                logger.info(f"추가 검색 파라미터: {param}={data[param]}")
        
        logger.info(f"RAG 검색 요청: params={json.dumps(search_params, ensure_ascii=False)}")
        search_response = requests.post(f"{RAG_ENDPOINT}/search", json=search_params)
        
        logger.info(f"RAG 응답 코드: {search_response.status_code}")
        if search_response.status_code != 200:
            logger.error(f"RAG 검색 오류 응답: {search_response.text}")
            return jsonify({"error": "문서 검색 중 오류가 발생했습니다"}), 500
            
        search_results = search_response.json()
        logger.info(f"RAG 검색 결과 수: {len(search_results.get('search_result', []))}")
        logger.info(f"RAG 응답 구조: {json.dumps({k: type(v).__name__ for k, v in search_results.items()}, ensure_ascii=False)}")
        
        # domain_results 확인 로깅
        if "domain_results" in search_results:
            logger.info(f"domain_results 키 존재: {list(search_results['domain_results'].keys())}")
        else:
            logger.warning("domain_results 키가 RAG 응답에 없습니다.")
        
        # 검색 결과가 없는 경우
        if not search_results.get("search_result"):
            logger.warning("검색 결과가 없습니다")
            return jsonify({
                "query": query,
                "top_m": top_m,
                "top_n": top_n,
                "search_count": 0,
                "reranked_count": 0,
                "results": []
            })
        
        # 원본 검색 결과 저장 (doc_id를 키로 사용)
        original_results_by_id = {}
        for item in search_results.get("search_result", []):
            if "doc_id" in item:
                original_results_by_id[item["doc_id"]] = item
                
                # 로깅 (처음 3개만)
                if len(original_results_by_id) <= 3:
                    logger.info(f"원본 결과 매핑: doc_id={item['doc_id']}, fields={list(item.keys())}")
        
        # 검색 결과에 메타데이터 보존 확인 및 처리
        for idx, item in enumerate(search_results.get("search_result", [])):
            # 메타데이터 필드 생성 (없는 경우)
            if "metadata" not in item:
                item["metadata"] = {}
                
            # 메타데이터에 주요 필드 복사
            for field in ["title", "author", "tags", "info", "domain", "doc_id", "raw_doc_id", "passage_id"]:
                if field in item and item[field] is not None:
                    item["metadata"][field] = item[field]
            
            # 원본 점수 저장
            if "score" in item:
                item["metadata"]["original_score"] = item["score"]
                
            # 인덱스 저장
            item["position"] = idx
            
            # 간단한 로깅
            if idx < 3:  # 처음 3개 항목만 로깅
                logger.info(f"검색 결과 {idx}번 메타데이터: {json.dumps(item.get('metadata', {}), ensure_ascii=False)}")
        
        # 2. Reranker 서비스 호출 - 하이브리드 재랭킹 사용
        logger.info(f"Reranker 서비스 호출 준비: endpoint={RERANKER_ENDPOINT}/hybrid-rerank")
        rerank_data = {
            "query": query,
            "results": search_results.get("search_result", [])
        }
        
        # 하이브리드 재랭킹 파라미터 설정
        rerank_params = {
            "top_k": top_n,
            "mrc_weight": mrc_weight
        }
        
        logger.info(f"Reranker 요청: params={json.dumps(rerank_params, ensure_ascii=False)}")
        rerank_response = requests.post(
            f"{RERANKER_ENDPOINT}/hybrid-rerank",
            params=rerank_params,
            json=rerank_data
        )
        
        logger.info(f"Reranker 응답 코드: {rerank_response.status_code}")
        if rerank_response.status_code != 200:
            logger.error(f"Reranker 오류 응답: {rerank_response.text}")
            return jsonify({"error": "문서 재순위화 중 오류가 발생했습니다"}), 500
            
        reranked_results = rerank_response.json()
        logger.info(f"재순위화된 문서 수: {len(reranked_results.get('results', []))}")
        logger.info(f"Reranker 응답 구조: {json.dumps({k: type(v).__name__ for k, v in reranked_results.items()}, ensure_ascii=False)}")
        
        # Reranker 결과 샘플 확인
        if reranked_results.get("results") and len(reranked_results.get("results")) > 0:
            sample_result = reranked_results.get("results")[0]
            logger.info(f"Reranker 결과 샘플: {json.dumps({k: v for k, v in sample_result.items() if k != 'text'}, ensure_ascii=False)}")
            if "metadata" in sample_result:
                logger.info(f"metadata 구조: {json.dumps(sample_result['metadata'], ensure_ascii=False)}")
            else:
                logger.warning("metadata 필드가 Reranker 결과에 없습니다")
        
        # 3. 결과 처리 및 응답 포맷팅
        processed_results = []
        
        # Reranker 결과 처리
        for idx, item in enumerate(reranked_results.get("results", [])):
            # 점수 확인
            rerank_score = item.get("score", 0)
            
            # 임계치 필터링
            if rerank_score < threshold:
                logger.info(f"임계치({threshold}) 미만 결과 필터링: doc_id={item.get('doc_id', 'unknown')}, score={rerank_score}")
                continue
            
            # 결과 아이템 초기화
            result_item = {}
            
            # 1. 원본 검색 결과의 모든 필드 복사 (있는 경우)
            doc_id = item.get("doc_id")
            if doc_id and doc_id in original_results_by_id:
                original_item = original_results_by_id[doc_id]
                # 원본 검색 결과의 모든 필드 복사 (metadata와 점수 관련 필드 제외)
                for key, value in original_item.items():
                    if key not in ["metadata", "score", "flashrank_score", "mrc_score", "hybrid_score"]:
                        result_item[key] = value
                logger.debug(f"원본 검색 결과에서 필드 복사: doc_id={doc_id}")
            
            # 2. 재랭킹 결과의 필드 복사 (원본 덮어쓰기, 메타데이터와 점수 관련 필드 제외)
            for key, value in item.items():
                if key not in ["metadata", "meta", "score", "flashrank_score", "mrc_score", "hybrid_score"]:
                    result_item[key] = value
            
            # 3. 점수 정보 설정 - 모든 점수를 최상위 레벨에 배치
            # 하이브리드 점수를 기본 점수로 사용
            result_item["score"] = item.get("score", 0)
            result_item["rerank_score"] = rerank_score
            result_item["rerank_position"] = idx
            
            # 원본 점수 정보 (있는 경우)
            if doc_id and doc_id in original_results_by_id and "score" in original_results_by_id[doc_id]:
                result_item["original_score"] = original_results_by_id[doc_id]["score"]
            
            # 하이브리드 재랭킹 점수 정보 (있는 경우)
            if "metadata" in item:
                if "flashrank_score" in item["metadata"]:
                    result_item["flashrank_score"] = item["metadata"]["flashrank_score"]
                if "mrc_score" in item["metadata"]:
                    result_item["mrc_score"] = item["metadata"]["mrc_score"]
            
            # id 필드 처리 - 원본 id만 보존하고 새 id는 생성하지 않음
            if "id" in result_item:
                result_item["original_id"] = result_item["id"]
                # id 필드 제거 (rerank_position으로 대체)
                del result_item["id"]
            
            # 4. 메타데이터 구성 - 원본 검색 결과의 모든 메타데이터 포함
            metadata = {}
            
            # 원본 검색 결과의 메타데이터 필드들
            metadata_fields = [
                "title", "author", "domain", "raw_doc_id", "passage_id", 
                "tags", "info", "created_at", "modified_at", "link", "og_image", "og_author"
            ]
            
            # 원본 검색 결과에서 메타데이터 수집
            if doc_id and doc_id in original_results_by_id:
                original_item = original_results_by_id[doc_id]
                
                # 기본 메타데이터 필드 복사
                for field in metadata_fields:
                    if field in original_item:
                        metadata[field] = original_item[field]
                
                # tags와 info는 객체일 수 있으므로 별도 처리
                for nested_field in ["tags", "info"]:
                    if nested_field in original_item and isinstance(original_item[nested_field], dict):
                        metadata[nested_field] = original_item[nested_field]
                
                # 원본 메타데이터가 있으면 병합
                if "metadata" in original_item:
                    for k, v in original_item["metadata"].items():
                        if k not in ["flashrank_score", "mrc_score", "original_score"]:  # 점수 정보 제외
                            metadata[k] = v
            
            # 최소한의 필수 메타데이터 보장
            metadata["doc_id"] = doc_id
            
            # 메타데이터 설정
            result_item["metadata"] = metadata
            
            # 5. meta 필드 제거 (metadata로 통합)
            if "meta" in result_item:
                del result_item["meta"]
            
            # 6. MRC 관련 필드 추가 (있는 경우)
            for mrc_field in ["mrc_answer", "mrc_char_ids"]:
                if mrc_field in item:
                    result_item[mrc_field] = item[mrc_field]
            
            processed_results.append(result_item)
        
        # 4. 최종 결과 반환
        response = {
            "query": query,
            "top_m": top_m,
            "top_n": top_n,
            "threshold": threshold,
            "mrc_weight": reranked_results.get("mrc_weight", mrc_weight),  # 실제 사용된 MRC 가중치
            "search_count": len(search_results.get("search_result", [])),
            "reranked_count": len(reranked_results.get("results", [])),
            "filtered_count": len(processed_results),
            "results": processed_results,
            "processing_time": reranked_results.get("processing_time", 0),
            "reranker_type": reranked_results.get("reranker_type", "hybrid")
        }
        
        logger.info(f"향상된 검색 완료: 검색={response['search_count']}, 재랭킹={response['reranked_count']}, 필터링 후={response['filtered_count']}")
        logger.info(f"결과 첫 항목 샘플: {json.dumps({k: v for k, v in processed_results[0].items() if k != 'text'} if processed_results else {}, ensure_ascii=False)}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"향상된 검색 중 예외 발생: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# 챗봇 API - 단순 질의응답용
@app.route("/prompt/chat", methods=["POST"])
def chat():
    try:
        summaryAgent = AgentService(config_path)
        data = request.json
        query = data.get("query")
        model = data.get("model", summaryAgent.default_model)
        stream = data.get("stream", False)  # 스트리밍 모드 기본값 False
        
        if not query:
            return jsonify({"error": "질문이 필요합니다"}), 400
        
        # 프롬프트 템플릿 없이 사용자 쿼리 직접 사용
        logger.info(f"Ollama API 챗봇 호출 시작: {model}, 스트리밍 모드: {stream}, 쿼리 직접 전달")
        
        try:
            # 스트리밍 모드에 따라 다른 처리
            if stream:
                # 스트리밍 모드로 처리
                def generate():
                    # 응답 누적을 위한 변수
                    accumulated_response = ""
                    
                    # heartbeat 카운터 초기화
                    heartbeat_counter = 0
                    
                    # heartbeat 메시지 전송 (15초마다)
                    def should_send_heartbeat():
                        nonlocal heartbeat_counter
                        heartbeat_counter += 1
                        return heartbeat_counter % 15 == 0
                    
                    with requests.post(
                        f"{OLLAMA_ENDPOINT}/api/generate",
                        json={
                            "model": model,
                            "prompt": query,
                            "stream": True
                        },
                        timeout=60,
                        stream=True
                    ) as ollama_response:
                        if ollama_response.status_code != 200:
                            logger.error(f"Ollama API 오류: {ollama_response.text}")
                            error_response = {
                                "query": query,
                                "model": model,
                                "error": "LLM 요청 중 오류가 발생했습니다",
                                "details": ollama_response.text
                            }
                            yield f"data: {json.dumps(error_response, ensure_ascii=False)}\n\n"
                            return
                        
                        # SSE 형식으로 응답 전송
                        for line in ollama_response.iter_lines():
                            # heartbeat 전송
                            if should_send_heartbeat():
                                yield ":\n\n"  # SSE 주석 형식의 heartbeat
                            
                            if line:
                                try:
                                    response_chunk = json.loads(line)
                                    chunk_text = response_chunk.get("response", "")
                                    if chunk_text:
                                        # 응답 누적
                                        accumulated_response += chunk_text
                                        
                                        # 기존 API 응답 형식으로 구성
                                        stream_response = {
                                            "query": query,
                                            "model": model,
                                            "response": accumulated_response,
                                            "streaming": True
                                        }
                                        
                                        # SSE 형식으로 전송
                                        yield f"data: {json.dumps(stream_response, ensure_ascii=False)}\n\n"
                                except json.JSONDecodeError:
                                    logger.error(f"JSON 디코딩 오류: {line}")
                                    continue
                        
                        # 스트림 종료 응답
                        final_response = {
                            "query": query,
                            "model": model,
                            "response": accumulated_response,
                            "streaming": False,
                            "done": True
                        }
                        yield f"data: {json.dumps(final_response, ensure_ascii=False)}\n\n"
                
                # 스트리밍 응답 헤더 설정 및 반환
                response = Response(stream_with_context(generate()), mimetype='text/event-stream')
                response.headers['Cache-Control'] = 'no-cache, no-transform'
                response.headers['X-Accel-Buffering'] = 'no'  # nginx에서 버퍼링 방지
                response.headers['Connection'] = 'keep-alive'  # 연결 유지
                return response
            else:
                # 기존 방식대로 처리 (스트리밍 없음)
                ollama_response = requests.post(
                    f"{OLLAMA_ENDPOINT}/api/generate",
                    json={
                        "model": model,
                        "prompt": query,  # 사용자 쿼리를 직접 전달
                        "stream": False
                    },
                    timeout=60
                )
                
                if ollama_response.status_code != 200:
                    logger.error(f"Ollama API 오류: {ollama_response.text}")
                    return jsonify({
                        "error": "LLM 요청 중 오류가 발생했습니다",
                        "details": ollama_response.text
                    }), 500
                    
                response_text = ollama_response.json().get("response", "")
                
                return jsonify({
                    "query": query,
                    "model": model,
                    "response": response_text
                })
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama 서비스 연결 오류: {str(e)}")
            return jsonify({
                "error": "Ollama 서비스에 연결할 수 없습니다",
                "details": str(e)
            }), 503
        
    except Exception as e:
        logger.error(f"챗봇 처리 중 오류 발생: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# Ollama 모델 목록 API
@app.route("/prompt/models", methods=["GET"])
def list_models():
    logger.info(f"💬 OLLAMA_ENDPOINT = {OLLAMA_ENDPOINT}")
    logger.info(f"💬 최종 요청 URL = {OLLAMA_ENDPOINT}/api/tags")
    try:
        # Ollama API 호출하여 모델 목록 가져오기
        logger.info("Ollama 모델 목록 요청")
        try:
            models_response = requests.get(
                f"{OLLAMA_ENDPOINT}/api/tags",
                timeout=10
            )
            summaryAgent = AgentService(config_path)
            
            if models_response.status_code != 200:
                logger.error(f"Ollama API 모델 목록 오류: {models_response.text}")
                return jsonify({
                    "error": "모델 목록을 가져오는 중 오류가 발생했습니다",
                    "details": models_response.text
                }), 500
                
            models_data = models_response.json()
            models = [model.get("name") for model in models_data.get("models", [])]
            
            return jsonify({
                "models": models,
                "default_model": summaryAgent.default_model,
                "total": len(models)
            })
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama 서비스 연결 오류: {str(e)}")
            logger.exception(e)  # 전체 traceback도 로그에 남기기
            return jsonify({
                "error": "Ollama 서비스에 연결할 수 없습니다",
                "details": str(e)
            }), 503
            
    except Exception as e:
        logger.error(f"모델 목록 처리 중 오류 발생: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # 개발 환경에서만 사용
    app.run(host="0.0.0.0", port=5000, debug=True) 