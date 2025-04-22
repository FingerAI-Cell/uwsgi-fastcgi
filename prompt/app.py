from flask import Flask, request, jsonify
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
        
            logger.info(f"Initializing Agent LLM with model: {self.default_model}")
            logger.debug(f"RAG Search Top {self.search_top}")
            logger.debug(f"Reranking Top {self.rerank_top}")
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
            "default_model": os.getenv("DEFAULT_MODEL", "mistral")
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
            "top_k": summaryAgent.search_top
        }
        
        # 추가 검색 매개변수
        for param in ["domain", "author", "start_date", "end_date"]:
            if param in data:
                search_params[param] = data[param]
                logger.info(f"추가 검색 파라미터: {param}={data[param]}")
        
        # curl 형식의 API 호출 로깅
        curl_command = f'curl -X GET "{RAG_ENDPOINT}/search'
        query_params = []
        for key, value in search_params.items():
            query_params.append(f"{key}={value}")
        if query_params:
            curl_command += "?" + "&".join(query_params)
        curl_command += '"'
        logger.info(f"RAG API curl 형식: {curl_command}")
        
        logger.info(f"RAG 검색 요청: params={json.dumps(search_params, ensure_ascii=False)}")
        search_response = requests.get(f"{RAG_ENDPOINT}/search", params=search_params)
        
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

# 챗봇 API - 단순 질의응답용
@app.route("/prompt/chat", methods=["POST"])
def chat():
    try:
        summaryAgent = AgentService(config_path)
        data = request.json
        query = data.get("query")
        model = data.get("model", summaryAgent.default_model)
        
        if not query:
            return jsonify({"error": "질문이 필요합니다"}), 400
        
        # 프롬프트 템플릿 로드
        logger.info("챗봇 프롬프트 템플릿 준비")
        template = summaryAgent.load_prompt_template("chat")
        
        if not template:
            # 템플릿이 없으면 기본 프롬프트 사용
            template = "당신은 도움이 되는 AI 어시스턴트입니다. 다음 질문에 정확하고 유용하게 답변해주세요: {query}"
        
        # 템플릿에 변수 채우기
        final_prompt = template.format(query=query)
        
        # Ollama API 호출
        logger.info(f"Ollama API 챗봇 호출 시작: {model}")
        try:
            ollama_response = requests.post(
                f"{OLLAMA_ENDPOINT}/api/generate",
                json={
                    "model": model,
                    "prompt": final_prompt,
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
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama 서비스 연결 오류: {str(e)}")
            return jsonify({
                "error": "Ollama 서비스에 연결할 수 없습니다",
                "details": str(e)
            }), 503
        
        return jsonify({
            "query": query,
            "model": model,
            "response": response_text
        })
        
    except Exception as e:
        logger.error(f"챗봇 처리 중 오류 발생: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# Ollama 모델 목록 API
@app.route("/prompt/models", methods=["GET"])
def list_models():
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