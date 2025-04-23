from flask import Flask, request, jsonify
import os
import logging
import json
import requests
from typing import Optional, Dict, Any
import base64

app = Flask(__name__)

# 로깅 설정
logging.basicConfig(
    filename='/var/log/vision/app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 설정 파일 로드
def load_config():
    try:
        with open('/vision/config.json', 'r') as f:
            config = json.load(f)
            logger.info(f"로드된 설정: {config}")  # 설정 내용 로깅
            return config
    except Exception as e:
        logger.error(f"설정 파일 로드 실패: {str(e)}")
        return {}

config = load_config()
logger.info(f"전역 config 변수 내용: {config}")  # 전역 변수에 저장된 설정 내용 로깅

def call_ollama_api(model: str, prompt: str, image_url: str) -> Optional[Dict[str, Any]]:
    """Ollama API를 호출하여 이미지 분석을 수행합니다."""
    try:
        ollama_url = os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434")
        logger.info(f"Ollama URL: {ollama_url}")
        
        # 이미지 URL에서 이미지를 다운로드하고 base64로 인코딩
        try:
            image_response = requests.get(image_url)
            image_response.raise_for_status()
            image_base64 = base64.b64encode(image_response.content).decode('utf-8')
            logger.info("이미지 다운로드 및 base64 인코딩 완료")
        except Exception as e:
            logger.error(f"이미지 다운로드 실패: {str(e)}")
            return None
        
        # API 요청 데이터
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False  # 스트리밍 비활성화
        }
        logger.info(f"Ollama API 요청 데이터 - model: {model}")  # 모델명 로깅
        logger.info("Ollama API 요청 준비 완료")
        
        # API 호출
        response = requests.post(
            f"{ollama_url}/api/generate",
            json=payload,
            timeout=30  # 30초 타임아웃 설정
        )
        
        # 응답 검증
        if response.status_code != 200:
            logger.error(f"Ollama API 오류 응답: {response.text}")
            return None
            
        result = response.json()
        
        if not result.get('response'):
            logger.warning(f"Ollama API 응답에 'response' 필드가 없습니다: {result}")
            return None
            
        return result
        
    except requests.exceptions.Timeout:
        logger.error("Ollama API 호출 시간 초과")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama API 호출 실패: {str(e)}")
        return None
    except json.JSONDecodeError:
        logger.error("Ollama API 응답을 JSON으로 파싱할 수 없습니다")
        return None
    except Exception as e:
        logger.error(f"예상치 못한 오류 발생: {str(e)}", exc_info=True)
        return None

@app.route('/vision/health', methods=['GET'])
def health_check():
    """서비스 상태를 확인합니다."""
    return jsonify({
        "status": "healthy",
        "service": "vision",
        "default_model": config.get('default_model')
    }), 200

@app.route('/vision/analyze', methods=['POST'])
def analyze_media():
    """이미지 분석을 수행합니다."""
    try:
        logger.info("이미지 분석 요청 시작")
        # 요청 데이터 검증
        data = request.get_json()
        logger.info(f"받은 요청 데이터: {data}")
        
        if not data:
            logger.error("JSON 데이터가 없음")
            return jsonify({"error": "JSON 데이터가 필요합니다"}), 400
            
        if 'url' not in data:
            logger.error("이미지 URL이 없음")
            return jsonify({"error": "이미지 URL이 필요합니다"}), 400
            
        # 파라미터 추출
        image_url = data['url']
        prompt = data.get('prompt', "이 이미지에 대해 설명해주세요")
        model = data.get('model', config.get('default_model', 'llama3.2-vision'))
        
        logger.info(f"분석 파라미터 - URL: {image_url}, Prompt: {prompt}, Model: {model}")
        
        # Ollama API 호출
        logger.info("Ollama API 호출 시작")
        result = call_ollama_api(model, prompt, image_url)
        if not result:
            logger.error("Ollama API 호출 실패")
            return jsonify({"error": "이미지 분석에 실패했습니다"}), 500
            
        logger.info(f"Ollama API 응답: {result}")
            
        # 응답 반환
        response = {
            "description": result.get('response'),
            "image_url": image_url,
            "model": model,
            "total_duration": result.get('total_duration'),
            "load_duration": result.get('load_duration'),
            "prompt_eval_count": result.get('prompt_eval_count'),
            "eval_count": result.get('eval_count'),
            "eval_duration": result.get('eval_duration')
        }
        logger.info(f"최종 응답: {response}")
        return jsonify(response), 200
            
    except Exception as e:
        logger.error(f"미디어 분석 중 오류 발생: {str(e)}", exc_info=True)
        return jsonify({"error": "미디어 분석 중 오류가 발생했습니다"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 